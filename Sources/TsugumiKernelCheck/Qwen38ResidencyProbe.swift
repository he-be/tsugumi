import Foundation
import Metal
import QuartzCore
import Tsugumi

// MARK: - What the routed buffer's kernel -> GPU start pays for (`--qwen38-residency-probe`)
//
// docs/qwen38/12 §4: the routed command buffers wait 25-34 ms per decode step (48 buffers) between
// `kernelStartTime` and `gpuStartTime`, and 12-13 ms of it stays with every expert already in the page cache
// (docs/qwen38/11 §3). The runner hands the encoder one no-copy view per expert part (`useResources`, about
// 30 per layer at T = 1) and keeps the views. This probe commits one command buffer per arm per round with the
// same kind of views, one dispatch per view reading every page of it, and splits the wait:
//   floor      no expert views
//   v30same    the same 30 views (10 experts x 3 parts) every round, pages cached
//   v30rot     10 experts x 3 parts from 20 sets in turn, pages cached; the first pass over the sets (each
//              view's first command buffer) is reported apart from the later passes
//   v30fresh   new views every round over the same cached pages (a view's first use without a page miss)
//   v30cold    new views over experts none of whose pages were in the page cache at start, each set once
//   big3rot    3 views of 10 contiguous experts each (v30rot's bytes in 3 buffers), 20 sets in turn
//   v300rot    100 experts x 3 parts, 5 sets in turn
// Each arm reads its own layers, so one arm's residency cannot serve another.

func runQwen38ResidencyProbe(gguf: String, rounds: Int) throws {
    setvbuf(stdout, nil, _IOLBF, 0)
    let file = try GGUFFile(url: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath))
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void touch(constant uchar *b [[buffer(0)]], device uint *o [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
        o[0] += b[i * 16384];
    }
    """
    let lib = try device.makeLibrary(source: src, options: nil)
    let pso = try device.makeComputePipelineState(function: lib.makeFunction(name: "touch")!)
    let out = device.makeBuffer(length: 4, options: .storageModeShared)!
    let page = Int(getpagesize())
    func wiredMB() -> Double {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        _ = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
        }
        return Double(stats.wire_count) * Double(getpagesize()) / 1e6
    }
    let wired0 = wiredMB()
    let partNames = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]
    let nExperts = 512

    func expertRange(layer: Int, part: Int, expert: Int, count: Int) throws -> (offset: Int, byteCount: Int) {
        let t = try file.tensor("blk.\(layer).\(partNames[part])")
        let per = t.byteCount / nExperts
        return (t.offset + expert * per, count * per)
    }
    func makeViews(_ spans: [(layer: Int, expert: Int, count: Int)]) throws -> [MTLBuffer] {
        var list: [MTLBuffer] = []
        for part in 0..<3 {
            for s in spans {
                let r = try expertRange(layer: s.layer, part: part, expert: s.expert, count: s.count)
                guard let v = file.noCopyBuffer(device: device, offset: r.offset, byteCount: r.byteCount) else {
                    throw GGUFFile.Error.format("view failed: layer \(s.layer) expert \(s.expert)")
                }
                list.append(v.buffer)
            }
        }
        return list
    }
    func spansBytes(_ spans: [(layer: Int, expert: Int, count: Int)]) throws -> [(offset: Int, byteCount: Int)] {
        var r: [(offset: Int, byteCount: Int)] = []
        for part in 0..<3 { for s in spans { r.append(try expertRange(layer: s.layer, part: part, expert: s.expert, count: s.count)) } }
        return r
    }

    // Cold experts: layers 30..47, experts none of whose three parts has a page in the cache now.
    var cold: [(layer: Int, expert: Int, count: Int)] = []
    for layer in 30..<48 {
        for ex in 0..<nExperts where cold.count < rounds * 10 {
            let rs = try spansBytes([(layer, ex, 1)])
            if rs.allSatisfy({ file.nonResidentBytes(offset: $0.offset, byteCount: $0.byteCount) >= ($0.byteCount / page) * page }) {
                cold.append((layer, ex, 1))
            }
        }
    }
    let coldRounds = cold.count / 10

    // Warm arms: fixed sets over layers 4..20, read ahead with pread.
    func sets(layer: Int, perSet: Int, sets: Int, contiguous: Bool) -> [[(layer: Int, expert: Int, count: Int)]] {
        (0..<sets).map { s in contiguous ? [(layer, s * perSet, perSet)] : (0..<perSet).map { (layer, s * perSet + $0, 1) } }
    }
    let warmSpans: [String: [[(layer: Int, expert: Int, count: Int)]]] = [
        "v30same": sets(layer: 8, perSet: 10, sets: 1, contiguous: false),
        "v30rot": sets(layer: 4, perSet: 10, sets: 20, contiguous: false),
        "v30fresh": sets(layer: 16, perSet: 10, sets: 1, contiguous: false),
        "big3rot": sets(layer: 12, perSet: 10, sets: 20, contiguous: true),
        "v300rot": sets(layer: 20, perSet: 100, sets: 5, contiguous: false),
    ]
    var ranges: [(offset: Int, byteCount: Int)] = []
    for (_, ss) in warmSpans { for s in ss { ranges += try spansBytes(s) } }
    let total = ranges.reduce(0) { $0 + $1.byteCount }
    let tr = CFAbsoluteTimeGetCurrent()
    file.preadRanges(ranges, threads: 4)
    print(String(format: "Qwen3.8 residency probe: %d rounds (cold %d), read ahead %.0f MB in %.2f s, cold experts %d (%.0f MB)",
                 rounds, coldRounds, Double(total) / 1e6, CFAbsoluteTimeGetCurrent() - tr, cold.count,
                 Double(try spansBytes(cold).reduce(0) { $0 + $1.byteCount }) / 1e6))
    let wiredRead = wiredMB()
    var samples: [String: [(Double, Double, Double, Double)]] = [:]
    var viewCount: [String: Int] = [:]
    var mb: [String: Double] = [:]
    var wiredUsed = 0.0
    let armNames = ["floor", "v30same", "v30rot", "v30fresh", "v30cold", "big3rot", "v300rot"]
    try autoreleasepool { () throws -> Void in
        var kept: [String: [[MTLBuffer]]] = [:]
        for name in ["v30same", "v30rot", "big3rot", "v300rot"] { kept[name] = try warmSpans[name]!.map { try makeViews($0) } }

        // samples: label -> (commit -> kernel start, kernel start -> GPU start, GPU, commit -> wait returns) in ms
        for round in -2..<rounds {   // two warm-up rounds, not kept
            for k in 0..<armNames.count { try autoreleasepool { () throws -> Void in
                let name = armNames[(k + max(round, 0)) % armNames.count]   // rotate which arm goes first
                var label = name
                var set: [MTLBuffer] = []
                switch name {
                case "floor": break
                case "v30fresh": set = try makeViews(warmSpans[name]![0])
                case "v30cold":
                    guard round >= 0 && round < coldRounds else { return }
                    set = try makeViews(Array(cold[(round * 10)..<(round * 10 + 10)]))
                default:
                    let all = kept[name]!
                    let r = round + 2
                    set = all[r % all.count]
                    if all.count > 1 { label = name + (r < all.count ? "-first" : "-later") }
                }
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pso)
                enc.setBuffer(out, offset: 0, index: 1)
                if set.isEmpty {
                    enc.setBuffer(out, offset: 0, index: 0)
                    enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                } else {
                    enc.useResources(set, usage: .read)
                    for v in set {
                        enc.setBuffer(v, offset: 0, index: 0)
                        enc.dispatchThreads(MTLSize(width: v.length / 16384, height: 1, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
                    }
                }
                enc.endEncoding()
                let c = CACurrentMediaTime()
                cb.commit()
                cb.waitUntilCompleted()
                let w = CACurrentMediaTime()
                guard round >= 0 else { return }
                samples[label, default: []].append(((cb.kernelStartTime - c) * 1000, (cb.gpuStartTime - cb.kernelStartTime) * 1000,
                                                    (cb.gpuEndTime - cb.gpuStartTime) * 1000, (w - c) * 1000))
                viewCount[label] = set.count
                mb[label] = Double(set.reduce(0) { $0 + $1.length }) / 1e6
            } }
        }
        wiredUsed = wiredMB()
    }
    do {   // one more command buffer after the views are gone, then let the driver settle
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(out, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    Thread.sleep(forTimeInterval: 2)
    print(String(format: "wired MB: start %.0f, after read-ahead %.0f, after the rounds (kept views %.0f MB) %.0f, views released %.0f",
                 wired0, wiredRead, 15.4 + 15.4 * 20 + 15.0 * 20 + 153.9 * 5, wiredUsed, wiredMB()))
    func q(_ xs: [Double], _ p: Double) -> Double {
        let s = xs.sorted()
        return s[min(s.count - 1, Int(Double(s.count - 1) * p + 0.5))]
    }
    print("arm              n views  MB/cb | commit>kernel p50 | kernel>gpu p50 p10 p90 | gpu p50 p90 | wall p50 p90")
    for label in samples.keys.sorted() {
        let s = samples[label]!
        let kg = s.map { $0.1 }, g = s.map { $0.2 }, wl = s.map { $0.3 }
        print(String(format: "%-15@ %3d %5d %6.1f | %17.3f | %14.3f %.3f %.3f | %7.3f %.3f | %8.3f %.3f", label as NSString,
                     s.count, viewCount[label]!, mb[label]!, q(s.map { $0.0 }, 0.5), q(kg, 0.5), q(kg, 0.1), q(kg, 0.9),
                     q(g, 0.5), q(g, 0.9), q(wl, 0.5), q(wl, 0.9)))
    }
}

// MARK: - Does a buffer paging in block another queue (`--qwen38-residency-probe --q38-queues`)
//
// docs/qwen38/13 §6: the page-in happens between `kernelStartTime` and `gpuStartTime`. Reordering rows (the MTP
// verify rows staggered by half a layer) can only hide it if a buffer paging in does not hold up buffers of another
// command queue. Per round, fresh views of 10 cold experts (30 views) and 30 kept views over cached pages:
//   warm       the cached buffer alone
//   coldA+warmA the cold buffer on queue A, then the cached buffer on queue A (same queue: queued behind it)
//   coldA+warmB the cold buffer on queue A, then the cached buffer on queue B, committed right after
//   coldA+coldB two cold buffers (different experts) on queues A and B, committed back to back
//   cold2A     the same two cold buffers' worth of views in one buffer on queue A (the T=2 union shape)

func runQwen38QueueProbe(gguf: String, rounds: Int) throws {
    setvbuf(stdout, nil, _IOLBF, 0)
    let file = try GGUFFile(url: URL(fileURLWithPath: (gguf as NSString).expandingTildeInPath))
    let device = MTLCreateSystemDefaultDevice()!
    let queueA = device.makeCommandQueue()!, queueB = device.makeCommandQueue()!
    let src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void touch(constant uchar *b [[buffer(0)]], device uint *o [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
        o[0] += b[i * 16384];
    }
    """
    let lib = try device.makeLibrary(source: src, options: nil)
    let pso = try device.makeComputePipelineState(function: lib.makeFunction(name: "touch")!)
    let out = device.makeBuffer(length: 4, options: .storageModeShared)!
    let page = Int(getpagesize())
    let partNames = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]
    func ranges(_ layer: Int, _ experts: [Int]) throws -> [(offset: Int, byteCount: Int)] {
        var r: [(offset: Int, byteCount: Int)] = []
        for part in partNames {
            let t = try file.tensor("blk.\(layer).\(part)")
            let per = t.byteCount / 512
            for ex in experts { r.append((t.offset + ex * per, per)) }
        }
        return r
    }
    func views(_ rs: [(offset: Int, byteCount: Int)]) throws -> [MTLBuffer] {
        try rs.map {
            guard let v = file.noCopyBuffer(device: device, offset: $0.offset, byteCount: $0.byteCount) else {
                throw GGUFFile.Error.format("view failed")
            }
            return v.buffer
        }
    }
    func buffer(_ q: MTLCommandQueue, _ set: [MTLBuffer]) -> MTLCommandBuffer {
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.useResources(set, usage: .read)
        for v in set {
            enc.setBuffer(v, offset: 0, index: 0)
            enc.dispatchThreads(MTLSize(width: v.length / 16384, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 1, depth: 1))
        }
        enc.endEncoding()
        return cb
    }

    // Cold experts (layers 30..47, no page of any part cached), 6 groups of 10 per round, each group used once.
    var cold: [(Int, Int)] = []
    outer: for layer in 30..<48 {
        for ex in 0..<512 {
            if cold.count >= rounds * 60 { break outer }
            if try ranges(layer, [ex]).allSatisfy({ file.nonResidentBytes(offset: $0.offset, byteCount: $0.byteCount) >= ($0.byteCount / page) * page }) {
                cold.append((layer, ex))
            }
        }
    }
    let usable = cold.count / 60
    var nextGroup = 0
    func coldViews() throws -> [MTLBuffer] {
        var list: [MTLBuffer] = []
        for (layer, ex) in cold[(nextGroup * 10)..<(nextGroup * 10 + 10)] { list += try views(try ranges(layer, [ex])) }
        nextGroup += 1
        return list
    }
    let warmRanges = try ranges(8, Array(0..<10))
    file.preadRanges(warmRanges, threads: 4)
    let warm = try views(warmRanges)
    do { let cb = buffer(queueA, warm); cb.commit(); cb.waitUntilCompleted() }
    do { let cb = buffer(queueB, warm); cb.commit(); cb.waitUntilCompleted() }
    print("Qwen3.8 queue probe: \(usable) rounds of cold experts")

    // label -> [(kernel>gpu ms of the first buffer, of the second, commit of the first -> both done)]
    var samples: [String: [(Double, Double, Double)]] = [:]
    let arms = ["warm", "coldA+warmA", "coldA+warmB", "coldA+coldB", "cold2A"]
    for round in 0..<usable {
        for k in 0..<arms.count {
            let arm = arms[(k + round) % arms.count]
            try autoreleasepool { () throws -> Void in
                let first: MTLCommandBuffer, second: MTLCommandBuffer?
                switch arm {
                case "warm": first = buffer(queueA, warm); second = nil
                case "coldA+warmA": first = buffer(queueA, try coldViews()); second = buffer(queueA, warm)
                case "coldA+warmB": first = buffer(queueA, try coldViews()); second = buffer(queueB, warm)
                case "coldA+coldB": first = buffer(queueA, try coldViews()); second = buffer(queueB, try coldViews())
                default: first = buffer(queueA, try coldViews() + coldViews()); second = nil
                }
                let c = CACurrentMediaTime()
                first.commit()
                second?.commit()
                first.waitUntilCompleted()
                second?.waitUntilCompleted()
                let w = CACurrentMediaTime()
                samples[arm, default: []].append(((first.gpuStartTime - first.kernelStartTime) * 1000,
                                                  second.map { ($0.gpuStartTime - $0.kernelStartTime) * 1000 } ?? .nan,
                                                  (w - c) * 1000))
            }
        }
    }
    func q(_ xs: [Double], _ p: Double) -> Double {
        let s = xs.filter { !$0.isNaN }.sorted()
        return s.isEmpty ? .nan : s[min(s.count - 1, Int(Double(s.count - 1) * p + 0.5))]
    }
    print("arm           n | first kernel>gpu p50 p90 | second kernel>gpu p50 p90 | both done p50 p90")
    for arm in arms {
        let s = samples[arm] ?? []
        print(String(format: "%-12@ %3d | %9.3f %.3f | %10.3f %.3f | %8.3f %.3f", arm as NSString, s.count,
                     q(s.map { $0.0 }, 0.5), q(s.map { $0.0 }, 0.9), q(s.map { $0.1 }, 0.5), q(s.map { $0.1 }, 0.9),
                     q(s.map { $0.2 }, 0.5), q(s.map { $0.2 }, 0.9)))
    }
}
