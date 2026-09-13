import Foundation
import Metal
import Tsugumi

// MARK: - `--q38-small-bench [iterations]`: GPU ms of the non-GEMV decode kernels
//
// Each sample is one command buffer with 48 dispatches of one kernel at the
// decode shape (the per-token count for kernels that run once per layer),
// dispatched the way `Qwen38Runner.run` does. Inputs are random; only time is read.
func runQwen38SmallKernelBench(iterations: Int) throws {
    let context = try MetalContext()
    let device = context.device
    let lib = try MetalContext.moduleLibrary(device: device, module: "qwen38")
    func pso(_ name: String) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
    }
    func buf(_ n: Int) -> MTLBuffer {
        let b = device.makeBuffer(length: n * 4, options: .storageModeShared)!
        let p = b.contents().bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { p[i] = Float.random(in: -1...1) }
        return b
    }
    let e = 2560, hc = 4
    let x = buf(hc * e), gamma = buf(hc * e), out = buf(hc * e), gate = buf(hc * e), mixed = buf(e)
    let small = buf(320), inj = buf(hc)
    struct Case { let name: String; let pso: MTLComputePipelineState; let n: MTLSize; let setup: (MTLComputeCommandEncoder) -> Void }
    var cases: [Case] = []
    let scale = buf(64)
    cases.append(Case(name: "rms_scale 4x2560", pso: try pso("q38_group_rms_scale"), n: MTLSize(width: hc, height: 1, depth: 1)) { enc in
        enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(scale, offset: 0, index: 1)
        var p = (UInt32(hc), UInt32(e), Float(1e-6)); enc.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 2)
    })
    cases.append(Case(name: "rms_apply 10240", pso: try pso("q38_rms_apply"), n: MTLSize(width: hc * e, height: 1, depth: 1)) { enc in
        enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(gamma, offset: 0, index: 1); enc.setBuffer(out, offset: 0, index: 2)
        var p = (UInt32(hc), UInt32(e), Float(1e-6)); enc.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 3)
        var w = UInt32(1); enc.setBytes(&w, length: 4, index: 4)
        enc.setBuffer(scale, offset: 0, index: 5)
    })
    cases.append(Case(name: "unary 320", pso: try pso("q38_unary"), n: MTLSize(width: 320, height: 1, depth: 1)) { enc in
        enc.setBuffer(small, offset: 0, index: 0); enc.setBuffer(small, offset: 0, index: 1)
        var p = (UInt32(0), Float(0.25), Float(1)); enc.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 2)
    })
    cases.append(Case(name: "unary 4", pso: try pso("q38_unary"), n: MTLSize(width: 4, height: 1, depth: 1)) { enc in
        enc.setBuffer(inj, offset: 0, index: 0); enc.setBuffer(inj, offset: 0, index: 1)
        var p = (UInt32(1), Float(0.25), Float(2)); enc.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 2)
    })
    cases.append(Case(name: "hc_mix 2560", pso: try pso("q38_hc_mix"), n: MTLSize(width: e, height: 1, depth: 1)) { enc in
        enc.setBuffer(out, offset: 0, index: 0); enc.setBuffer(gate, offset: 0, index: 1); enc.setBuffer(mixed, offset: 0, index: 2)
        var p = (UInt32(hc), UInt32(e)); enc.setBytes(&p, length: 8, index: 3)
    })
    cases.append(Case(name: "hc_combine 10240", pso: try pso("q38_hc_combine"), n: MTLSize(width: hc * e, height: 1, depth: 1)) { enc in
        enc.setBuffer(gamma, offset: 0, index: 0); enc.setBuffer(mixed, offset: 0, index: 1); enc.setBuffer(inj, offset: 0, index: 2)
        var p = (UInt32(hc), UInt32(e)); enc.setBytes(&p, length: 8, index: 3)
    })
    // GDN (per linear layer) and attention (per full layer) kernels.
    let Hk = 16, Hv = 48, Dl = 128, C = 2 * Hk * Dl + Hv * Dl
    let conv = buf(C), qkv = buf(C), hist = buf(3 * C), cw = buf(C * 4)
    let ga = buf(Hv), gb = buf(Hv), A = buf(Hv), dt = buf(Hv), S = buf(Hv * Dl * Dl), lo = buf(Hv * Dl), z = buf(Hv * Dl), nw = buf(Dl)
    let Aneg = A.contents().bindMemory(to: Float.self, capacity: Hv)
    for i in 0..<Hv { Aneg[i] = -abs(Aneg[i]) }
    var gp = (UInt32(Hk), UInt32(Hv), UInt32(Dl))
    cases.append(Case(name: "gdn_conv 10240", pso: try pso("q38_gdn_conv"), n: MTLSize(width: C, height: 1, depth: 1)) { enc in
        enc.setBuffer(qkv, offset: 0, index: 0); enc.setBuffer(hist, offset: 0, index: 1); enc.setBuffer(cw, offset: 0, index: 2)
        enc.setBuffer(conv, offset: 0, index: 3); var c = UInt32(C), k = UInt32(4)
        enc.setBytes(&c, length: 4, index: 4); enc.setBytes(&k, length: 4, index: 5)
    })
    cases.append(Case(name: "gdn_qk_norm 32x128", pso: try pso("q38_gdn_qk_norm"), n: MTLSize(width: 2 * Hk, height: 1, depth: 1)) { enc in
        enc.setBuffer(conv, offset: 0, index: 0); enc.setBytes(&gp, length: 12, index: 1)
    })
    cases.append(Case(name: "gdn_step 128x48", pso: try pso("q38_gdn_step"), n: MTLSize(width: Dl, height: Hv, depth: 1)) { enc in
        enc.setBuffer(conv, offset: 0, index: 0); enc.setBuffer(ga, offset: 0, index: 1); enc.setBuffer(gb, offset: 0, index: 2)
        enc.setBuffer(A, offset: 0, index: 3); enc.setBuffer(dt, offset: 0, index: 4); enc.setBuffer(S, offset: 0, index: 5)
        enc.setBuffer(lo, offset: 0, index: 6); enc.setBytes(&gp, length: 12, index: 7)
    })
    cases.append(Case(name: "gdn_norm_gate 48x128", pso: try pso("q38_gdn_norm_gate"), n: MTLSize(width: Hv, height: 1, depth: 1)) { enc in
        enc.setBuffer(lo, offset: 0, index: 0); enc.setBuffer(z, offset: 0, index: 1); enc.setBuffer(nw, offset: 0, index: 2)
        var d = UInt32(Dl), ep = Float(1e-6); enc.setBytes(&d, length: 4, index: 3); enc.setBytes(&ep, length: 4, index: 4)
    })
    let H = 24, Hkv = 2, D = 256, cap = 2048
    let qg = buf(2 * H * D), kc = buf(cap * Hkv * D), vc = buf(cap * Hkv * D), qn = buf(D), kn = buf(D)
    let freq = buf(32), q = buf(H * D), qgate = buf(H * D), ao = buf(H * D), sel = buf(cap)
    var ap = (UInt32(H), UInt32(Hkv), UInt32(D), UInt32(64), UInt32(9), Float(1e-6))
    cases.append(Case(name: "attn_prep 26", pso: try pso("q38_attn_prep"), n: MTLSize(width: H + Hkv, height: 1, depth: 1)) { enc in
        enc.setBuffer(qg, offset: 0, index: 0); enc.setBuffer(kc, offset: 0, index: 1); enc.setBuffer(qn, offset: 0, index: 2)
        enc.setBuffer(kn, offset: 0, index: 3); enc.setBuffer(freq, offset: 0, index: 4); enc.setBuffer(q, offset: 0, index: 5)
        enc.setBuffer(qgate, offset: 0, index: 6); enc.setBytes(&ap, length: MemoryLayout.size(ofValue: ap), index: 7)
    })
    // Five-pass attention (`q38_attn_score` ... `q38_attn_mix`), one layer = 5 dispatches.
    for n in [10, 2048] {
        var pp = (UInt32(H), UInt32(Hkv), UInt32(D), UInt32(n), UInt32(0))
        let len = MemoryLayout.size(ofValue: pp)
        let scores = buf(H * cap), mxb = buf(H), smb = buf(H)
        let score = try pso("q38_attn_score"), stat = try pso("q38_attn_stat"), weight = try pso("q38_attn_weight"), mix = try pso("q38_attn_mix")
        var times: [Double] = []
        for it in 0...iterations {
            let cb = context.queue.makeCommandBuffer()!
            for _ in 0..<48 {
                func lanes(_ p: MTLComputePipelineState, _ g: MTLSize, _ f: (MTLComputeCommandEncoder) -> Void) {
                    let enc = cb.makeComputeCommandEncoder()!
                    enc.setComputePipelineState(p); f(enc)
                    enc.dispatchThreadgroups(g, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    enc.endEncoding()
                }
                lanes(score, MTLSize(width: n, height: H, depth: 1)) { enc in
                    enc.setBuffer(q, offset: 0, index: 0); enc.setBuffer(kc, offset: 0, index: 1)
                    enc.setBuffer(scores, offset: 0, index: 2); enc.setBuffer(sel, offset: 0, index: 3)
                    enc.setBytes(&pp, length: len, index: 4)
                }
                for op in [UInt32(0), 1] {
                    lanes(stat, MTLSize(width: H, height: 1, depth: 1)) { enc in
                        enc.setBuffer(scores, offset: 0, index: 0); enc.setBuffer(mxb, offset: 0, index: 1)
                        enc.setBuffer(smb, offset: 0, index: 2); enc.setBytes(&pp, length: len, index: 3)
                        var o = op; enc.setBytes(&o, length: 4, index: 4)
                    }
                }
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(weight)
                enc.setBuffer(scores, offset: 0, index: 0); enc.setBuffer(mxb, offset: 0, index: 1)
                enc.setBuffer(smb, offset: 0, index: 2); enc.setBytes(&pp, length: len, index: 3)
                enc.dispatchThreads(MTLSize(width: n, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
                enc.endEncoding()
                lanes(mix, MTLSize(width: D, height: H, depth: 1)) { enc in
                    enc.setBuffer(scores, offset: 0, index: 0); enc.setBuffer(vc, offset: 0, index: 1)
                    enc.setBuffer(qgate, offset: 0, index: 2); enc.setBuffer(ao, offset: 0, index: 3)
                    enc.setBuffer(sel, offset: 0, index: 4); enc.setBytes(&pp, length: len, index: 5)
                }
            }
            cb.commit()
            cb.waitUntilCompleted()
            if it > 0 { times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000) }
        }
        print(String(format: "  %-20@ %7.2f", "attn 5-pass n \(n)" as NSString, times.sorted()[times.count / 2]))
    }
    let shg = buf(640), shu = buf(640), acc = buf(e), sg = buf(1)
    cases.append(Case(name: "silu_mul 640", pso: try pso("q38_silu_mul"), n: MTLSize(width: 640, height: 1, depth: 1)) { enc in
        enc.setBuffer(shg, offset: 0, index: 0); enc.setBuffer(shu, offset: 0, index: 1)
    })
    cases.append(Case(name: "add_scaled 2560", pso: try pso("q38_add_scaled"), n: MTLSize(width: e, height: 1, depth: 1)) { enc in
        enc.setBuffer(acc, offset: 0, index: 0); enc.setBuffer(mixed, offset: 0, index: 1); enc.setBuffer(sg, offset: 0, index: 2)
        var op = UInt32(1); enc.setBytes(&op, length: 4, index: 3)
    })
    print("Qwen3.8 small kernels: 48 dispatches per sample, median of \(iterations) (GPU ms)")
    for c in cases {
        var times: [Double] = []
        for i in 0...iterations {
            let cb = context.queue.makeCommandBuffer()!
            for _ in 0..<48 {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(c.pso)
                c.setup(enc)
                if c.name.hasPrefix("rms_scale") {
                    enc.dispatchThreadgroups(c.n, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                } else {
                    let w = min(c.pso.threadExecutionWidth, max(c.n.width, 1))
                    let h = min(max(c.pso.maxTotalThreadsPerThreadgroup / w, 1), max(c.n.height, 1))
                    enc.dispatchThreads(c.n, threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
                }
                enc.endEncoding()
            }
            cb.commit()
            cb.waitUntilCompleted()
            if i > 0 { times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000) }
        }
        print(String(format: "  %-20@ %7.2f", c.name as NSString, times.sorted()[times.count / 2]))
    }
}
