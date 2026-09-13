import Foundation
import Metal
import Tsugumi

// MARK: - QSA selection on the GPU (`--q38-select-check`)
//
// `q38_idx_topk`, `q38_idx_union` and `q38_attn_mask_sel` (qwen38.metal) against the host rule of
// `Qwen38Runner.attention`: per query the top k blocks by score, the lower block first on ties, every
// block when there are at most k. The model's scores rarely tie, so the scores here are synthetic:
// small integers (ties everywhere), mostly zeros (relu), and continuous values.

func runQwen38SelectCheck() throws -> Bool {
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let lib = try MetalContext.moduleLibrary(device: device, module: "qwen38")
    func pso(_ name: String) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    }
    let topk = try pso("q38_idx_topk"), union = try pso("q38_idx_union"), mask = try pso("q38_attn_mask_sel")
    var rng = SystemRandomNumberGenerator()
    var passed = true
    print("Qwen3.8 QSA selection kernels")
    for (label, draw) in [("ties 0..3", { Float(Int.random(in: 0...3, using: &rng)) }),
                          ("90% zeros", { Float.random(in: 0..<1, using: &rng) < 0.9 ? 0 : Float.random(in: 0..<5, using: &rng) }),
                          ("continuous", { Float.random(in: 0..<100, using: &rng) })] as [(String, () -> Float)] {
        for (k, pos0, T) in [(16, 0, 200), (16, 60, 150), (5, 1000, 97), (512, 3000, 300)] {
            let nb = (pos0 + T) / 4
            var scores = [Float](repeating: 0, count: T * nb)
            for i in scores.indices { scores[i] = draw() }
            let sBuf = device.makeBuffer(bytes: scores, length: scores.count * 4, options: .storageModeShared)!
            let thr = device.makeBuffer(length: T * 4, options: .storageModeShared)!
            let cut = device.makeBuffer(length: T * 4, options: .storageModeShared)!
            let cb = queue.makeCommandBuffer()!
            var enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(topk)
            enc.setBuffer(sBuf, offset: 0, index: 0)
            enc.setBuffer(thr, offset: 0, index: 1)
            enc.setBuffer(cut, offset: 0, index: 2)
            var tp = (UInt32(nb), UInt32(k), UInt32(pos0), UInt32(T))
            enc.setBytes(&tp, length: MemoryLayout.size(ofValue: tp), index: 3)
            enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()

            // One sub-batch of 37 queries from the middle: its union and the mask over every token up to it.
            let r0 = T / 3, rows = min(37, T - r0), G = 3
            let lastPos = pos0 + r0 + rows - 1
            let list = (0...lastPos).map(UInt32.init) + [UInt32.max]
            let U = list.count
            let any = device.makeBuffer(length: nb, options: .storageModeShared)!
            let lBuf = device.makeBuffer(bytes: list, length: U * 4, options: .storageModeShared)!
            let mBuf = device.makeBuffer(length: rows * G * U * 4, options: .storageModeShared)!
            enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(union)
            enc.setBuffer(sBuf, offset: r0 * nb * 4, index: 0)
            enc.setBuffer(thr, offset: r0 * 4, index: 1)
            enc.setBuffer(cut, offset: r0 * 4, index: 2)
            enc.setBuffer(any, offset: 0, index: 3)
            var up = (UInt32(nb), UInt32(k), UInt32(pos0 + r0), UInt32(rows))
            enc.setBytes(&up, length: MemoryLayout.size(ofValue: up), index: 4)
            enc.dispatchThreads(MTLSize(width: (lastPos + 1) / 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
            enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(mask)
            enc.setBuffer(mBuf, offset: 0, index: 0)
            enc.setBuffer(lBuf, offset: 0, index: 1)
            enc.setBuffer(sBuf, offset: r0 * nb * 4, index: 2)
            enc.setBuffer(thr, offset: r0 * 4, index: 3)
            enc.setBuffer(cut, offset: r0 * 4, index: 4)
            var mp = (UInt32(G), UInt32(U), UInt32(nb), UInt32(pos0 + r0))
            enc.setBytes(&mp, length: MemoryLayout.size(ofValue: mp), index: 5)
            enc.dispatchThreads(MTLSize(width: U, height: rows * G, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }

            // Host rule.
            let thrP = thr.contents().bindMemory(to: UInt32.self, capacity: T)
            let cutP = cut.contents().bindMemory(to: UInt32.self, capacity: T)
            var selected = [Set<Int>](repeating: [], count: T)
            var bad = 0
            for t in 0..<T {
                let n = (pos0 + t + 1) / 4
                let row = Array(scores[(t * nb)..<(t * nb + n)])
                let want = n <= k ? Set(0..<n)
                    : Set((0..<n).sorted { row[$0] != row[$1] ? row[$0] > row[$1] : $0 < $1 }.prefix(k))
                selected[t] = want
                let got = Set((0..<n).filter { b in
                    let u = row[b].bitPattern
                    return u > thrP[t] || (u == thrP[t] && b <= Int(cutP[t]))
                })
                if got != want { bad += 1 }
            }
            let anyP = any.contents().assumingMemoryBound(to: UInt8.self)
            var badUnion = 0
            for b in 0..<((lastPos + 1) / 4) {
                let want = (r0..<(r0 + rows)).contains { selected[$0].contains(b) }
                if (anyP[b] != 0) != want { badUnion += 1 }
            }
            let mP = mBuf.contents().bindMemory(to: Float.self, capacity: rows * G * U)
            var badMask = 0
            for tl in 0..<rows {
                let t = r0 + tl, at = pos0 + t, n = (at + 1) / 4
                for i in 0..<U {
                    let j = Int(list[i])
                    let keep = list[i] != UInt32.max && j <= at && (j / 4 >= n || selected[t].contains(j / 4))
                    for h in 0..<G where (mP[(tl * G + h) * U + i] == 0) != keep { badMask += 1 }
                }
            }
            let ok = bad == 0 && badUnion == 0 && badMask == 0
            passed = passed && ok
            print("  \(label) k \(k) pos \(pos0) T \(T): top-k rows wrong \(bad)/\(T), union \(badUnion), mask \(badMask)  \(ok ? "ok" : "FAIL")")
        }
    }
    print("  \(passed ? "PASS" : "FAIL")")
    return passed
}
