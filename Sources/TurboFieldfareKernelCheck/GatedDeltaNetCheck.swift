import Foundation
import Metal
import TurboFieldfare

// MARK: - qwen_delta_rule — Gated DeltaNet の数値検査
//
// `docs/qwen35moe/04-PHASES.md` Phase 2。**カーネルのバグと丸め誤差を分離する**
// のが目的なので、同じ入力に対して 3 通り走らせる:
//
//   - GPU (fp16 の q/k/v、fp32 の状態、レジスタ常駐)
//   - CPU float32  … 「桁の落ち方」の床。GPU と同じ精度で、同じ順序で足す
//   - CPU double   … 真値の代わり
//
// 合格は「GPU の誤差が float32 の床と同じ桁」であること。床そのものが大きいなら
// それはカーネルの罪ではない (PLAN_VISION §6 の教訓)。
//
// **Phase 2 の出口は T=1 では閉じない。**再帰状態は 1 トークンでは誤差が積まれず、
// `qwen_delta_rule` の蓄積の間違いが見えない。2048 トークン流したあとの状態を
// 見るのが本題で、`--gdn-tokens` の既定が 2048 なのはそのため。

private enum GDNCheckError: Error {
    case noCommandBuffer
    case dispatchFailed(String)
}

private struct GDNShape {
    var seqLen: Int
    var numKHeads = 16      // linear_num_key_heads
    var numVHeads = 32      // linear_num_value_heads
    var keyHeadDim = 128    // linear_key_head_dim
    var valueHeadDim = 128  // linear_value_head_dim

    var qkCount: Int { seqLen * numKHeads * keyHeadDim }
    var vCount: Int { seqLen * numVHeads * valueHeadDim }
    var gateCount: Int { seqLen * numVHeads }
    var stateCount: Int { numVHeads * valueHeadDim * keyHeadDim }
}

/// Deterministic inputs in the shapes the model actually produces:
/// `k` is L2-normalised, `q` is L2-normalised and scaled by `Dk^-0.5`
/// (the asymmetric pair `docs/qwen35moe/14-REFERENCE.md` §3 confirmed),
/// `beta = sigmoid(·)` lives in (0,1) and `g = exp(-exp(A_log)·softplus(·))`
/// sits just under 1 — the measured α median is 0.78–0.99
/// (`docs/qwen35moe/10-MLX4BIT-AUDIT.md` §5).
private struct GDNInputs {
    var q: [Float16]
    var k: [Float16]
    var v: [Float16]
    var g: [Float]
    var beta: [Float]
    var state0: [Float]
}

private struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [0,1).
    mutating func uniform() -> Double { Double(next() >> 11) * (1.0 / 9007199254740992.0) }
    /// Box–Muller, one at a time (this runs once per case; speed is irrelevant).
    mutating func normal() -> Double {
        let u1 = Swift.max(uniform(), 1e-12)
        let u2 = uniform()
        return (-2.0 * Foundation.log(u1)).squareRoot() * Foundation.cos(2.0 * .pi * u2)
    }
}

private func makeGDNInputs(_ shape: GDNShape, seed: UInt64) -> GDNInputs {
    var rng = SplitMix64(state: seed)
    let dk = shape.keyHeadDim
    let dv = shape.valueHeadDim

    func normalisedRows(count: Int, dim: Int, scale: Double) -> [Float16] {
        var out = [Float16](repeating: 0, count: count * dim)
        var row = [Double](repeating: 0, count: dim)
        for r in 0..<count {
            var norm = 0.0
            for i in 0..<dim { row[i] = rng.normal(); norm += row[i] * row[i] }
            let inv = scale / Swift.max(norm.squareRoot(), 1e-12)
            for i in 0..<dim { out[r * dim + i] = Float16(row[i] * inv) }
        }
        return out
    }

    let q = normalisedRows(count: shape.seqLen * shape.numKHeads, dim: dk,
                           scale: 1.0 / Double(dk).squareRoot())
    let k = normalisedRows(count: shape.seqLen * shape.numKHeads, dim: dk, scale: 1.0)
    var v = [Float16](repeating: 0, count: shape.vCount)
    for i in 0..<v.count { v[i] = Float16(rng.normal() * 0.5) }

    var g = [Float](repeating: 0, count: shape.gateCount)
    var beta = [Float](repeating: 0, count: shape.gateCount)
    for i in 0..<g.count {
        // A_log ∈ [-4, 4] と softplus(N(0,1)+dt_bias) を模す (10 §5 の実測値域)。
        let aLog = -4.0 + 8.0 * rng.uniform()
        let dt = Foundation.log(1.0 + Foundation.exp(rng.normal() - 1.5))
        g[i] = Float(Foundation.exp(-Foundation.exp(aLog) * dt))
        beta[i] = Float(1.0 / (1.0 + Foundation.exp(-rng.normal())))
    }
    // 状態は「途中から始める」ことも見たいので 0 にしない。
    var state0 = [Float](repeating: 0, count: shape.stateCount)
    for i in 0..<state0.count { state0[i] = Float(rng.normal() * 0.05) }
    _ = dv
    return GDNInputs(q: q, k: k, v: v, g: g, beta: beta, state0: state0)
}

/// The recurrence itself, generic over the accumulator type so the same source
/// produces the float32 floor and the double truth.
private func gdnReference<T: BinaryFloatingPoint>(_ inputs: GDNInputs,
                                                  _ shape: GDNShape,
                                                  as: T.Type,
                                                  decayAfterContraction: Bool = false)
    -> (y: [T], state: [T]) {
    let dk = shape.keyHeadDim
    let dv = shape.valueHeadDim
    let hv = shape.numVHeads
    let hk = shape.numKHeads
    let repeatFactor = hv / hk
    var state = inputs.state0.map { T($0) }
    var y = [T](repeating: 0, count: shape.vCount)
    var kv = [T](repeating: 0, count: dv)

    for t in 0..<shape.seqLen {
        for h in 0..<hv {
            let kh = h / repeatFactor
            let qkBase = (t * hk + kh) * dk
            let vBase = (t * hv + h) * dv
            let sBase = h * dv * dk
            let gt = T(inputs.g[t * hv + h])
            let bt = T(inputs.beta[t * hv + h])

            for d in 0..<dv {
                var acc: T = 0
                let row = sBase + d * dk
                for i in 0..<dk {
                    if !decayAfterContraction { state[row + i] *= gt }
                    acc += state[row + i] * T(inputs.k[qkBase + i])
                }
                // 検出力用の「もっともらしい間違い」: 減衰を縮約のあとに掛ける。
                if decayAfterContraction {
                    for i in 0..<dk { state[row + i] *= gt }
                }
                kv[d] = acc
            }
            for d in 0..<dv {
                let delta = (T(inputs.v[vBase + d]) - kv[d]) * bt
                let row = sBase + d * dk
                var out: T = 0
                for i in 0..<dk {
                    state[row + i] += T(inputs.k[qkBase + i]) * delta
                    out += state[row + i] * T(inputs.q[qkBase + i])
                }
                y[vBase + d] = out
            }
        }
    }
    return (y, state)
}

private func relative(_ actual: [Double], _ reference: [Double]) -> Double {
    var maxDiff = 0.0
    var refNorm = 0.0
    for i in 0..<actual.count {
        precondition(reference[i].isFinite, "reference is not finite — harness bug")
        refNorm = Swift.max(refNorm, abs(reference[i]))
        if !actual[i].isFinite { return .infinity }
        maxDiff = Swift.max(maxDiff, abs(actual[i] - reference[i]))
    }
    precondition(refNorm > 1e-4, "reference has no signal — harness bug")
    return maxDiff / refNorm
}

private func buffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer {
    values.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                          options: .storageModeShared)!
    }
}

private func read<T>(_ buffer: MTLBuffer, count: Int, as: T.Type) -> [T] {
    Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: T.self, capacity: count),
                              count: count))
}

/// One GPU run. `chunk` splits the sequence into that many tokens per dispatch,
/// carrying the state across dispatches — the shape decode and chunked prefill
/// actually use.
private func runGDNOnGPU(_ context: MetalContext,
                         _ kernel: GatedDeltaNet,
                         _ inputs: GDNInputs,
                         _ shape: GDNShape,
                         timeBlock: GatedDeltaNet.TimeBlock,
                         chunk: Int? = nil) throws -> (y: [Float], state: [Float]) {
    let device = context.device
    let q = buffer(device, inputs.q)
    let k = buffer(device, inputs.k)
    let v = buffer(device, inputs.v)
    let g = buffer(device, inputs.g)
    let beta = buffer(device, inputs.beta)
    let state = buffer(device, inputs.state0)
    let y = device.makeBuffer(length: shape.vCount * MemoryLayout<Float16>.size,
                              options: .storageModeShared)!

    let step = chunk ?? shape.seqLen
    var start = 0
    while start < shape.seqLen {
        let tokens = Swift.min(step, shape.seqLen - start)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw GDNCheckError.noCommandBuffer
        }
        let encoded = kernel.encode(
            commandBuffer: commandBuffer,
            q: q, qOffset: start * shape.numKHeads * shape.keyHeadDim * MemoryLayout<Float16>.size,
            k: k, kOffset: start * shape.numKHeads * shape.keyHeadDim * MemoryLayout<Float16>.size,
            v: v, vOffset: start * shape.numVHeads * shape.valueHeadDim * MemoryLayout<Float16>.size,
            g: g, gOffset: start * shape.numVHeads * MemoryLayout<Float>.size,
            beta: beta, betaOffset: start * shape.numVHeads * MemoryLayout<Float>.size,
            stateIn: state,
            y: y, yOffset: start * shape.numVHeads * shape.valueHeadDim * MemoryLayout<Float16>.size,
            stateOut: state,
            seqLen: tokens,
            numKHeads: shape.numKHeads,
            numVHeads: shape.numVHeads,
            keyHeadDim: shape.keyHeadDim,
            valueHeadDim: shape.valueHeadDim,
            timeBlock: timeBlock)
        precondition(encoded, "qwen_delta_rule refused the dispatch — harness bug")
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw GDNCheckError.dispatchFailed("\(error)")
        }
        start += tokens
    }
    let yHalf = read(y, count: shape.vCount, as: Float16.self)
    return (yHalf.map { Float($0) }, read(state, count: shape.stateCount, as: Float.self))
}

func runGatedDeltaNetCheck(tokens: Int) throws -> [CaseResult] {
    let context = try MetalContext()
    let kernel = try GatedDeltaNet(context: context)
    var results: [CaseResult] = []

    for seqLen in [1, tokens] {
        let shape = GDNShape(seqLen: seqLen)
        let inputs = makeGDNInputs(shape, seed: 0x51D3_5F00 &+ UInt64(seqLen))
        let truth = gdnReference(inputs, shape, as: Double.self)
        let floor32 = gdnReference(inputs, shape, as: Float.self)
        let floorY = relative(floor32.y.map { Double($0) }, truth.y.map { Double($0) })
        let floorState = relative(floor32.state.map { Double($0) }, truth.state.map { Double($0) })
        let label = seqLen == 1 ? "decode" : "prefill \(seqLen)"

        for block in GatedDeltaNet.TimeBlock.allCases {
            let got = try runGDNOnGPU(context, kernel, inputs, shape, timeBlock: block)
            let relY = relative(got.y.map { Double($0) }, truth.y.map { Double($0) })
            let relState = relative(got.state.map { Double($0) }, truth.state.map { Double($0) })
            // 出力は fp16 で書き戻すので、床は float32 の床ではなく fp16 の刻み
            // (2^-11 ≈ 4.9e-4) が支配する。状態は fp32 のままなので床は float32 側。
            results.append(result("qwen_delta_rule y \(label) TB=\(block.rawValue)",
                                  groupSize: 0, rel: relY, tolerance: 4e-3,
                                  detail: String(format: "float32 の床 %.2e", floorY)))
            results.append(result("qwen_delta_rule state \(label) TB=\(block.rawValue)",
                                  groupSize: 0, rel: relState,
                                  tolerance: Swift.max(20 * floorState, 1e-5),
                                  detail: String(format: "float32 の床 %.2e", floorState)))
        }

        // 検出力: 減衰を縮約のあとに掛ける「もっともらしい間違い」(この算式で
        // いちばん起こしやすい取り違え) に対しては、同じ GPU 出力が外れること。
        // 床は y の許容 4e-3 の 10 倍に置く — **正しい参照が通った同じ物差しで、
        // 間違った参照が桁違いに落ちる**ことを言うため。α の中央値が 0.9 前後
        // (10 §5) なので 1 ステップの差は小さく、それでも 19 倍離れている。
        let wrong = gdnReference(inputs, shape, as: Double.self, decayAfterContraction: true)
        let got = try runGDNOnGPU(context, kernel, inputs, shape, timeBlock: .tb32)
        results.append(detectionResult("qwen_delta_rule 検出力 (減衰の位置) \(label)",
                                       groupSize: 0,
                                       rel: relative(got.y.map { Double($0) },
                                                     wrong.y.map { Double($0) }),
                                       floor: 4e-2))
    }

    // 状態の持ち越し: 1 回で流したものと、32 トークンずつに切って状態を渡したものが
    // **ビット一致**すること。時間ブロックは算術の順序を変えないので、ここは
    // 誤差の話ではない (`docs/qwen35moe/14-REFERENCE.md` §3 の 3 本目の検査の GPU 版)。
    let shape = GDNShape(seqLen: tokens)
    let inputs = makeGDNInputs(shape, seed: 0x51D3_5F00 &+ UInt64(tokens))
    let single = try runGDNOnGPU(context, kernel, inputs, shape, timeBlock: .tb32)
    let carried = try runGDNOnGPU(context, kernel, inputs, shape, timeBlock: .tb32, chunk: 32)
    var stateMismatch = 0
    for i in 0..<single.state.count where single.state[i] != carried.state[i] { stateMismatch += 1 }
    var yMismatch = 0
    for i in 0..<single.y.count where single.y[i] != carried.y[i] { yMismatch += 1 }
    results.append(result("qwen_delta_rule 状態の持ち越し (1 回 対 32 ずつ)",
                          groupSize: 0,
                          rel: Double(stateMismatch + yMismatch),
                          tolerance: 0,
                          detail: "ビット一致を要求。state \(stateMismatch) / y \(yMismatch) 本が不一致"))
    return results
}

// MARK: - 時間 (`docs/qwen35moe/05-RISKS.md` §2 #2 の中止線)
//
// 線形注意は 40 層のうち 30 層にあり、prefill の 1 チャンクでその 30 層ぶんが
// 走る。中止線は **30 層合計で 150 ms**。ここはカーネル 1 本だけのマイクロベンチ
// なので、`bench.sh` の作法 (temp / クールダウン) の対象ではない — 数字は
// 「このカーネルが 1 層に何 ms 使うか」であって、モデルの速度ではない。

func runGatedDeltaNetBench(tokens: Int, iterations: Int) throws {
    let context = try MetalContext()
    let kernel = try GatedDeltaNet(context: context)
    let device = context.device
    print("# qwen_delta_rule マイクロベンチ  \(iterations) 回の中央値  "
          + "Hk=16 Hv=32 Dk=Dv=128")
    print("       T      TB   1 層 (ms)    30 層 (ms)   GB/s (状態)")

    for seqLen in [1, tokens] {
        let shape = GDNShape(seqLen: seqLen)
        let inputs = makeGDNInputs(shape, seed: 0x1234)
        let q = buffer(device, inputs.q), k = buffer(device, inputs.k)
        let v = buffer(device, inputs.v), g = buffer(device, inputs.g)
        let beta = buffer(device, inputs.beta), state = buffer(device, inputs.state0)
        let y = device.makeBuffer(length: shape.vCount * MemoryLayout<Float16>.size,
                                  options: .storageModeShared)!
        for block in GatedDeltaNet.TimeBlock.allCases {
            var samples: [Double] = []
            for _ in 0..<iterations {
                guard let commandBuffer = context.queue.makeCommandBuffer() else {
                    throw GDNCheckError.noCommandBuffer
                }
                _ = kernel.encode(commandBuffer: commandBuffer,
                                  q: q, k: k, v: v, g: g, beta: beta,
                                  stateIn: state, y: y, stateOut: state,
                                  seqLen: seqLen,
                                  numKHeads: shape.numKHeads,
                                  numVHeads: shape.numVHeads,
                                  keyHeadDim: shape.keyHeadDim,
                                  valueHeadDim: shape.valueHeadDim,
                                  timeBlock: block)
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                if let error = commandBuffer.error {
                    throw GDNCheckError.dispatchFailed("\(error)")
                }
                samples.append((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1e3)
            }
            samples.sort()
            let median = samples[samples.count / 2]
            // 状態の往復だけを数えたバイト (読み + 書き)。q/k/v/y はこれより小さい。
            let stateBytes = Double(shape.stateCount * MemoryLayout<Float>.size * 2)
            print("\(String(format: "%8d", seqLen))  \(String(format: "%6d", block.rawValue))"
                  + "  \(String(format: "%10.3f", median))"
                  + "  \(String(format: "%12.1f", median * 30))"
                  + "  \(String(format: "%12.1f", stateBytes / (median * 1e-3) / 1e9))")
        }
    }
}

// MARK: - 投機デコード幅 2 の再帰状態の実費 (Phase 7 の M0)
//
// `docs/qwen35moe/33-MTP-ACCEPTANCE.md` §3-4 の 2 番目。**カーネルを 1 本も
// 書かずに**、幅 2 の検証パスが再帰状態に払う固定費を出す。
//
// 33 §3-3 が「符号を握るのは受理率ではなくパスあたりの固定費」と書いた、
// その固定費の主犯が本節の対象である。NVMAI は同型のモデルで、和集合だけの
// 予測 0.993 に対し実測 0.85 — **差の 16% が固定費**だった
// (`32-NVMAI-ADOPT.md` §1-4)。当方の固定費がそれより軽いかどうかが論点。
//
// 測るのは 3 つ:
//
//   1. **restore** — 棄却時に影バッファを本物へ書き戻す blit。30 層ぶんの
//      state (FP32) と conv (FP16) をまとめて 1 回。`RecurrentStateManager` は
//      連続 2 本なので blit も 2 回で済む (`32-NVMAI-ADOPT.md` §1-2)
//   2. **snapshot** — 確定行の直後の状態を取る手。NVMAI はカーネルに第 2 の
//      状態出力を足しているが、**カーネルを触らない実装もある**: T=2 を
//      T=1 の 2 回に割れば、境目の状態はそのまま本物のバッファに立つ。
//      その代償 `2×T1 − T2` をここで測る
//   3. **幅 2 の再帰層そのもの** — T=1 と T=2 の差。33 §3-2 は「パス全体が
//      エキスパート和集合比 (1.554〜1.585 倍) で伸びる」という悲観で組んで
//      あるが、**再帰層はエキスパートを 1 枚も読まない**ので、その仮定が
//      どれだけ悲観かがここで分かる
//
// 腕は交互に取る (`32-NVMAI-ADOPT.md` §5-1 の interleaved A/B)。逐次スイープは
// 前の腕が温めたものを次の腕の手柄にする。
func runQwenSpeculativeStateBench(iterations: Int, cooldownSeconds: Double) throws {
    let context = try MetalContext()
    let kernel = try GatedDeltaNet(context: context)
    let device = context.device

    // `scratch/ornith-oq4e-g64.gturbo/manifest.json` の linearAttention と同じ:
    // layerCount 30 / numValueHeads 32 / valueHeadDim 128 / keyHeadDim 128 /
    // numKeyHeads 16 / convKernelDim 4。`RecurrentStateManager` の算式をそのまま。
    let layerCount = 30
    let numKHeads = 16, numVHeads = 32, keyHeadDim = 128, valueHeadDim = 128
    let convKernelDim = 4
    let stateBytesPerLayer = numVHeads * valueHeadDim * keyHeadDim * MemoryLayout<Float>.size
    let convChannels = 2 * numKHeads * keyHeadDim + numVHeads * valueHeadDim
    let convBytesPerLayer = (convKernelDim - 1) * convChannels * MemoryLayout<Float16>.size
    let stateBytes = stateBytesPerLayer * layerCount
    let convBytes = convBytesPerLayer * layerCount
    let totalBytes = stateBytes + convBytes

    print("# 投機デコード幅 2 の再帰状態の実費 (測定、n=\(iterations) の中央値、交互)")
    print("  30 層 / Hk=\(numKHeads) Hv=\(numVHeads) Dk=Dv=\(keyHeadDim) / conv 窓 \(convKernelDim)")
    print(String(format: "  state %d B (%.1f MiB) + conv %d B (%.2f MiB) = **%d B (%.1f MiB)**",
                 stateBytes, Double(stateBytes) / 1048576.0,
                 convBytes, Double(convBytes) / 1048576.0,
                 totalBytes, Double(totalBytes) / 1048576.0))
    print(String(format: "  クールダウン %.0f 秒", cooldownSeconds))

    guard let stateSrc = device.makeBuffer(length: stateBytes, options: .storageModeShared),
          let stateDst = device.makeBuffer(length: stateBytes, options: .storageModeShared),
          let convSrc = device.makeBuffer(length: convBytes, options: .storageModeShared),
          let convDst = device.makeBuffer(length: convBytes, options: .storageModeShared) else {
        throw GDNCheckError.dispatchFailed("影バッファを確保できない")
    }
    memset(stateSrc.contents(), 0x3F, stateBytes)
    memset(convSrc.contents(), 0x3C, convBytes)

    // T=1 (通常の decode) と T=2 (検証パス) を同じ入力の形で用意する。
    let shape1 = GDNShape(seqLen: 1), shape2 = GDNShape(seqLen: 2)
    let in1 = makeGDNInputs(shape1, seed: 0x1234), in2 = makeGDNInputs(shape2, seed: 0x1234)
    let q1 = buffer(device, in1.q), k1 = buffer(device, in1.k), v1 = buffer(device, in1.v)
    let g1 = buffer(device, in1.g), b1 = buffer(device, in1.beta), s1 = buffer(device, in1.state0)
    let y1 = device.makeBuffer(length: shape1.vCount * MemoryLayout<Float16>.size,
                               options: .storageModeShared)!
    let q2 = buffer(device, in2.q), k2 = buffer(device, in2.k), v2 = buffer(device, in2.v)
    let g2 = buffer(device, in2.g), b2 = buffer(device, in2.beta), s2 = buffer(device, in2.state0)
    let y2 = device.makeBuffer(length: shape2.vCount * MemoryLayout<Float16>.size,
                               options: .storageModeShared)!

    func timeGPU(_ body: (MTLCommandBuffer) throws -> Void) throws -> Double {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw GDNCheckError.noCommandBuffer
        }
        try body(commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw GDNCheckError.dispatchFailed("\(error)") }
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1e3
    }

    func blit(_ commandBuffer: MTLCommandBuffer) throws {
        guard let encoder = commandBuffer.makeBlitCommandEncoder() else {
            throw GDNCheckError.dispatchFailed("blit encoder が取れない")
        }
        encoder.copy(from: stateSrc, sourceOffset: 0, to: stateDst,
                     destinationOffset: 0, size: stateBytes)
        encoder.copy(from: convSrc, sourceOffset: 0, to: convDst,
                     destinationOffset: 0, size: convBytes)
        encoder.endEncoding()
    }

    func deltaRule(_ commandBuffer: MTLCommandBuffer, seqLen: Int,
                   _ block: GatedDeltaNet.TimeBlock, dispatches: Int) {
        for _ in 0..<dispatches {
            _ = kernel.encode(commandBuffer: commandBuffer,
                              q: seqLen == 1 ? q1 : q2, k: seqLen == 1 ? k1 : k2,
                              v: seqLen == 1 ? v1 : v2, g: seqLen == 1 ? g1 : g2,
                              beta: seqLen == 1 ? b1 : b2,
                              stateIn: seqLen == 1 ? s1 : s2,
                              y: seqLen == 1 ? y1 : y2,
                              stateOut: seqLen == 1 ? s1 : s2,
                              seqLen: seqLen,
                              numKHeads: numKHeads, numVHeads: numVHeads,
                              keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim,
                              timeBlock: block)
        }
    }

    var blitSamples: [Double] = []
    var memcpySamples: [Double] = []
    var t1: [GatedDeltaNet.TimeBlock: [Double]] = [:]
    var t2: [GatedDeltaNet.TimeBlock: [Double]] = [:]
    var t1x2: [GatedDeltaNet.TimeBlock: [Double]] = [:]

    for iteration in 0..<iterations {
        if iteration > 0, cooldownSeconds > 0 { Thread.sleep(forTimeInterval: cooldownSeconds) }
        blitSamples.append(try timeGPU(blit))

        // storageModeShared なので CPU からも同じバイトを写せる。GPU を 1 度も
        // 起こさない道があるかどうかは、棄却のたびに払う費用として意味がある。
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        memcpy(stateDst.contents(), stateSrc.contents(), stateBytes)
        memcpy(convDst.contents(), convSrc.contents(), convBytes)
        memcpySamples.append(Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started) / 1e6)

        for block in GatedDeltaNet.TimeBlock.allCases {
            t1[block, default: []].append(
                try timeGPU { deltaRule($0, seqLen: 1, block, dispatches: layerCount) })
            t2[block, default: []].append(
                try timeGPU { deltaRule($0, seqLen: 2, block, dispatches: layerCount) })
            t1x2[block, default: []].append(
                try timeGPU { deltaRule($0, seqLen: 1, block, dispatches: layerCount * 2) })
        }
    }

    func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    let blitMs = median(blitSamples)
    let memcpyMs = median(memcpySamples)
    print("")
    print("## 1. 影バッファの写し (30 層ぶんを 1 回)")
    print("  手                     ms      実効 GB/s (読み+書き)")
    print(String(format: "  blit (GPU)      %9.3f  %12.1f", blitMs,
                 Double(totalBytes * 2) / (blitMs * 1e-3) / 1e9))
    print(String(format: "  memcpy (CPU)    %9.3f  %12.1f", memcpyMs,
                 Double(totalBytes * 2) / (memcpyMs * 1e-3) / 1e9))

    print("")
    print("## 2. 再帰層 30 層 — 幅 1 と幅 2")
    print("      TB     T=1 (ms)    T=2 (ms)   T=2/T=1   T=1×2 (ms)   割る代償 (ms)")
    for block in GatedDeltaNet.TimeBlock.allCases {
        let a = median(t1[block] ?? [0]), b = median(t2[block] ?? [0])
        let c = median(t1x2[block] ?? [0])
        print("\(String(format: "%8d", block.rawValue))"
              + "\(String(format: "%13.3f", a))\(String(format: "%12.3f", b))"
              + "\(String(format: "%10.3f", b / a))"
              + "\(String(format: "%13.3f", c))\(String(format: "%16.3f", c - b))")
    }

    // 期待値は 33 §2-1 の受理率で重み付けする。棄却 = 1 − P1。
    let rejectRate = 1.0 - 0.7870
    let decodeMsPerToken = 49.98   // `27-PHASE6-THROUGHPUT.md` §2 の重ね腕 (m.json)
    let snapshotBlit = blitMs
    let restore = rejectRate * blitMs
    print("")
    print("## 3. 期待値 (**導出**) — 1 パスあたり")
    print("  受理率 P1 = 78.70% (`33-MTP-ACCEPTANCE.md` §2-1)、棄却 "
          + String(format: "%.1f%%", rejectRate * 100))
    print("  decode 1 パス = " + String(format: "%.2f ms/tok", decodeMsPerToken)
          + " (`27-PHASE6-THROUGHPUT.md` §2)")
    print("")
    print("  実装                          snapshot   restore     合計    パス比")
    print(String(format: "  blit で写す                 %8.3f  %8.3f %8.3f  %7.2f%%",
                 snapshotBlit, restore, snapshotBlit + restore,
                 100.0 * (snapshotBlit + restore) / decodeMsPerToken))
    for block in GatedDeltaNet.TimeBlock.allCases {
        let split = median(t1x2[block] ?? [0]) - median(t2[block] ?? [0])
        print(String(format: "  T=2 を T=1×2 に割る (TB%2d)  %8.3f  %8.3f %8.3f  %7.2f%%",
                     block.rawValue, split, restore, split + restore,
                     100.0 * (split + restore) / decodeMsPerToken))
    }
    print("")
    print("  比較対象: NVMAI の固定費は和集合予測との差 **16%** "
          + "(`32-NVMAI-ADOPT.md` §1-4 / `33-MTP-ACCEPTANCE.md` §3-3)")
    print("  **これは再帰状態ぶんだけ**である。KV カーソル・幅 2 の頭・"
          + "占有率低下は別勘定 (33 §3-4 の 1 と 3)")
}
