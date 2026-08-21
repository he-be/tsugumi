import Foundation
import Metal
import TurboFieldfare

// MARK: - qwen.metal の数値検査 (`docs/qwen35moe/04-PHASES.md` Phase 2 の残り)
//
// `GatedDeltaNetCheck.swift` と同じ形で立てる: **チェックポイントも fixtures も
// 開かず**、実物と同じ形の合成入力に対して 3 通り走らせる。
//
//   - GPU
//   - CPU float32  … 「桁の落ち方」の床
//   - CPU double   … 真値の代わり
//
// カーネルのバグと丸め誤差を分離するのが目的なので、床より上に出た誤差だけを
// カーネルの罪として数える (PLAN_VISION §6 の教訓)。
//
// **各カーネルに 1 本ずつ検出力の負例を付ける。**このモデルで静かに壊れる道は
// 事前に分かっている (docs/qwen35moe/03-DESIGN.md §2、01-MODEL.md §3-1):
//
//   partial RoPE   Gemma の組 (i, HD/2+i) と分母 HD を流用する
//   conv1d         `[C, K]` の K 軸を逆順に読む (上流 bf16 と MLX で軸が違う)
//   l2norm         和ではなく平均に eps を足す (すぐ隣の RMSNorm と同じ形なので紛れる)
//   RMSNormGated   `1 + w` を足す (30 層ぶんの norm でここだけ足さない)
//
// これらの「もっともらしい間違い」を参照側に作って、**同じ GPU 出力が桁違いに
// 落ちる**ことを確かめる。正例が通るだけでは物差しが働いている証明にならない。

private enum QwenCheckError: Error {
    case noCommandBuffer
    case dispatchFailed(String)
    case encodeRefused(String)
}

// MARK: - 汎用スカラー
//
// 参照を float32 と double の 2 通りで走らせるために、指数関数だけ抽象化する。

private protocol QwenScalar: BinaryFloatingPoint {
    static func qExp(_ x: Self) -> Self
    static func qLog1p(_ x: Self) -> Self
    static func qPow(_ x: Self, _ y: Self) -> Self
    static func qCos(_ x: Self) -> Self
    static func qSin(_ x: Self) -> Self
}

extension Float: QwenScalar {
    static func qExp(_ x: Float) -> Float { Foundation.expf(x) }
    static func qLog1p(_ x: Float) -> Float { Foundation.log1p(x) }
    static func qPow(_ x: Float, _ y: Float) -> Float { Foundation.powf(x, y) }
    static func qCos(_ x: Float) -> Float { Foundation.cosf(x) }
    static func qSin(_ x: Float) -> Float { Foundation.sinf(x) }
}

extension Double: QwenScalar {
    static func qExp(_ x: Double) -> Double { Foundation.exp(x) }
    static func qLog1p(_ x: Double) -> Double { Foundation.log1p(x) }
    static func qPow(_ x: Double, _ y: Double) -> Double { Foundation.pow(x, y) }
    static func qCos(_ x: Double) -> Double { Foundation.cos(x) }
    static func qSin(_ x: Double) -> Double { Foundation.sin(x) }
}

private func qSigmoid<T: QwenScalar>(_ x: T) -> T { 1 / (1 + T.qExp(-x)) }
private func qSilu<T: QwenScalar>(_ x: T) -> T { x * qSigmoid(x) }
private func qSoftplus<T: QwenScalar>(_ x: T) -> T {
    Swift.max(x, 0) + T.qLog1p(T.qExp(-abs(x)))
}

// MARK: - 道具
//
// 乱数は `GatedDeltaNetCheck.swift` と同じ SplitMix64 + Box–Muller。
// `TurboFieldfareValidationSupport` の同名の型は `normal()` を持たないので、
// 検査どうしで同じ流れを再現できるようこちらにも置く (種から再現できること)。

private struct QwenRNG {
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
    mutating func normal() -> Double {
        let u1 = Swift.max(uniform(), 1e-12)
        let u2 = uniform()
        return (-2.0 * Foundation.log(u1)).squareRoot() * Foundation.cos(2.0 * .pi * u2)
    }
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

private func relative(_ actual: [Double], _ reference: [Double]) -> Double {
    precondition(actual.count == reference.count, "shape mismatch — harness bug")
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

private func doubles<T: BinaryFloatingPoint>(_ values: [T]) -> [Double] {
    values.map { Double($0) }
}

/// BF16 の学習済みベクトル。GPU に渡すビット列と、参照が使う**まったく同じ値**の
/// float を一緒に作る (丸めの差をカーネルの誤差に混ぜないため)。
private struct BF16Vector {
    var bits: [UInt16]
    var values: [Float]

    init(count: Int, scale: Double, rng: inout QwenRNG) {
        var bits: [UInt16] = []
        var values: [Float] = []
        bits.reserveCapacity(count)
        values.reserveCapacity(count)
        for _ in 0..<count {
            let raw = Quantization.bf16Bits(Float(rng.normal() * scale))
            bits.append(raw)
            values.append(Quantization.bf16ToFloat(raw))
        }
        self.bits = bits
        self.values = values
    }
}

// MARK: - 1. qwen_delta_qkv_prepare
//
// 因果 depthwise conv1d + SiLU + q/k の l2norm。線形注意 30 層の入口。

private struct PrepareShape {
    var seqLen: Int
    var numKHeads = 16
    var numVHeads = 32
    var headDim = 128
    var convKernel = 4

    var convDim: Int { (2 * numKHeads + numVHeads) * headDim }   // 8192
    var qkCount: Int { seqLen * numKHeads * headDim }
    var vCount: Int { seqLen * numVHeads * headDim }
    var stateCount: Int { (convKernel - 1) * convDim }
}

private struct PrepareInputs {
    var qkv: [Float16]        // [T, C]
    var convWeight: BF16Vector  // [C, K]
    var state0: [Float16]     // [K-1, C]
}

private enum PrepareVariant {
    case correct
    /// `[C, K]` の K 軸を逆順に読む。MLX は `[8192, 4, 1]`、上流 bf16 は
    /// `[8192, 1, 4]` で、squeeze 後の並びを取り違えるとこうなる。
    case reversedTaps
    /// l2norm の eps を**和ではなく平均**に足す (隣の RMSNorm と同じ形にしてしまう)。
    case meanNormalisation
}

private func makePrepareInputs(_ shape: PrepareShape, seed: UInt64) -> PrepareInputs {
    var rng = QwenRNG(state: seed)
    var qkv = [Float16](repeating: 0, count: shape.seqLen * shape.convDim)
    for i in 0..<qkv.count { qkv[i] = Float16(rng.normal() * 0.6) }
    let weight = BF16Vector(count: shape.convDim * shape.convKernel, scale: 0.4, rng: &rng)
    var state0 = [Float16](repeating: 0, count: shape.stateCount)
    for i in 0..<state0.count { state0[i] = Float16(rng.normal() * 0.6) }
    return PrepareInputs(qkv: qkv, convWeight: weight, state0: state0)
}

private func prepareReference<T: QwenScalar>(_ inputs: PrepareInputs,
                                             _ shape: PrepareShape,
                                             as: T.Type,
                                             variant: PrepareVariant = .correct)
    -> (q: [T], k: [T], v: [T], state: [T]) {
    let C = shape.convDim
    let K = shape.convKernel
    let HD = shape.headDim
    let Hk = shape.numKHeads
    let Hv = shape.numVHeads
    let eps = T(QwenKernels.l2Eps)
    let qScale = T(1.0 / Double(HD).squareRoot())

    var history = inputs.state0.map { T($0) }          // [K-1, C]
    var conv = [T](repeating: 0, count: C)
    var q = [T](repeating: 0, count: shape.qkCount)
    var k = [T](repeating: 0, count: shape.qkCount)
    var v = [T](repeating: 0, count: shape.vCount)

    for t in 0..<shape.seqLen {
        for c in 0..<C {
            let x = T(inputs.qkv[t * C + c])
            // カーネルと同じ足す順 (現在のタップが先、過去が後) にする。
            // 順序を変えると float32 の床が動いてしまう。
            let currentTap = variant == .reversedTaps ? 0 : K - 1
            var acc = x * T(inputs.convWeight.values[c * K + currentTap])
            for j in 0..<(K - 1) {
                let tap = variant == .reversedTaps ? (K - 1 - j) : j
                acc += history[j * C + c] * T(inputs.convWeight.values[c * K + tap])
            }
            conv[c] = qSilu(acc)
        }
        for j in 0..<(K - 2) {
            for c in 0..<C { history[j * C + c] = history[(j + 1) * C + c] }
        }
        for c in 0..<C { history[(K - 2) * C + c] = T(inputs.qkv[t * C + c]) }

        for block in 0..<(2 * Hk + Hv) {
            let isQ = block < Hk
            let isK = !isQ && block < 2 * Hk
            let head = isQ ? block : (isK ? block - Hk : block - 2 * Hk)
            if isQ || isK {
                var sum: T = 0
                for d in 0..<HD {
                    let value = conv[block * HD + d]
                    sum += value * value
                }
                let denominator = variant == .meanNormalisation
                    ? sum / T(HD) + eps
                    : sum + eps
                let inv = 1 / denominator.squareRoot()
                for d in 0..<HD {
                    let scaled = conv[block * HD + d] * inv * (isQ ? qScale : 1)
                    let index = (t * Hk + head) * HD + d
                    if isQ { q[index] = scaled } else { k[index] = scaled }
                }
            } else {
                for d in 0..<HD {
                    v[(t * Hv + head) * HD + d] = conv[block * HD + d]
                }
            }
        }
    }
    return (q, k, v, history)
}

private func runPrepareOnGPU(_ context: MetalContext,
                             _ kernels: QwenKernels,
                             _ inputs: PrepareInputs,
                             _ shape: PrepareShape,
                             chunk: Int? = nil,
                             tokensPerGroup: Int = QwenKernels.tokensPerGroup) throws
    -> (q: [Float16], k: [Float16], v: [Float16], state: [Float16]) {
    let device = context.device
    let qkv = buffer(device, inputs.qkv)
    let weight = buffer(device, inputs.convWeight.bits)
    // 状態は 2 面。カーネルは同じバッファを許さない (トークンブロックが 2 個以上
    // あると書き手と読み手が別の threadgroup になるため) ので、ここでも交互に使う。
    var state = buffer(device, inputs.state0)
    var stateBack = buffer(device, inputs.state0)
    let q = device.makeBuffer(length: shape.qkCount * MemoryLayout<Float16>.size,
                              options: .storageModeShared)!
    let k = device.makeBuffer(length: shape.qkCount * MemoryLayout<Float16>.size,
                              options: .storageModeShared)!
    let v = device.makeBuffer(length: shape.vCount * MemoryLayout<Float16>.size,
                              options: .storageModeShared)!

    let step = chunk ?? shape.seqLen
    var start = 0
    while start < shape.seqLen {
        let tokens = Swift.min(step, shape.seqLen - start)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw QwenCheckError.noCommandBuffer
        }
        let encoded = kernels.encodeDeltaQKVPrepare(
            commandBuffer: commandBuffer,
            qkv: qkv, qkvOffset: start * shape.convDim * MemoryLayout<Float16>.size,
            convWeight: weight,
            stateIn: state, stateOut: stateBack,
            q: q, qOffset: start * shape.numKHeads * shape.headDim * MemoryLayout<Float16>.size,
            k: k, kOffset: start * shape.numKHeads * shape.headDim * MemoryLayout<Float16>.size,
            v: v, vOffset: start * shape.numVHeads * shape.headDim * MemoryLayout<Float16>.size,
            seqLen: tokens,
            numKHeads: shape.numKHeads,
            numVHeads: shape.numVHeads,
            headDim: shape.headDim,
            tokensPerGroup: tokensPerGroup)
        guard encoded else { throw QwenCheckError.encodeRefused("qwen_delta_qkv_prepare") }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw QwenCheckError.dispatchFailed("\(error)") }
        swap(&state, &stateBack)
        start += tokens
    }
    return (read(q, count: shape.qkCount, as: Float16.self),
            read(k, count: shape.qkCount, as: Float16.self),
            read(v, count: shape.vCount, as: Float16.self),
            read(state, count: shape.stateCount, as: Float16.self))
}

// MARK: - 2. qwen_delta_gates

private func gatesReference<T: QwenScalar>(a: [Float16], b: [Float16],
                                           aLog: [Float], dtBias: [Float],
                                           seqLen: Int, heads: Int,
                                           as: T.Type,
                                           dropBias: Bool = false) -> (g: [T], beta: [T]) {
    var g = [T](repeating: 0, count: seqLen * heads)
    var beta = [T](repeating: 0, count: seqLen * heads)
    for i in 0..<(seqLen * heads) {
        let h = i % heads
        let bias = dropBias ? T(0) : T(dtBias[h])
        g[i] = T.qExp(-T.qExp(T(aLog[h])) * qSoftplus(T(a[i]) + bias))
        beta[i] = qSigmoid(T(b[i]))
    }
    return (g, beta)
}

// MARK: - 3. qwen_delta_norm_gate

private func normGateReference<T: QwenScalar>(o: [Float16], z: [Float16], weight: [Float],
                                              seqLen: Int, heads: Int, headDim: Int,
                                              eps: Float,
                                              as: T.Type,
                                              plusOneWeight: Bool = false) -> [T] {
    var out = [T](repeating: 0, count: seqLen * heads * headDim)
    for t in 0..<seqLen {
        for h in 0..<heads {
            let base = (t * heads + h) * headDim
            var sum: T = 0
            for d in 0..<headDim {
                let value = T(o[base + d])
                sum += value * value
            }
            let inv = 1 / (sum / T(headDim) + T(eps)).squareRoot()
            for d in 0..<headDim {
                // `Qwen3_5MoeRMSNormGated` は **`1 + w` ではない**。30 層ぶんの
                // norm のうちここだけが例外 (01-MODEL.md §3-1)。
                let w = plusOneWeight ? 1 + T(weight[d]) : T(weight[d])
                out[base + d] = T(o[base + d]) * inv * w * qSilu(T(z[base + d]))
            }
        }
    }
    return out
}

// MARK: - 4. qwen_qkv_epilogue

private struct AttnShape {
    var seqLen: Int
    var numQHeads = 16
    var numKVHeads = 2
    var headDim = 256
    var rotaryDim = 64
    var theta: Float = 10_000_000
    var position = 0

    var qCount: Int { seqLen * numQHeads * 2 * headDim }
    var kCount: Int { seqLen * numKVHeads * headDim }
}

private enum RopeVariant {
    case qwen
    /// Gemma の `fused_rope_neox_pair`: head 全体を回し、組は `(i, HD/2+i)`、
    /// 周波数の分母は `head_dim`。**流用すると静かに間違う**の実演。
    case gemma
}

/// `q` は 2 倍幅 `[T, NQ, 2*HD]` のまま返す (gate 側が無傷であることも見たいので)。
private func epilogueReference<T: QwenScalar>(q: [Float16], k: [Float16],
                                              qWeight: [Float], kWeight: [Float],
                                              _ shape: AttnShape,
                                              eps: Float,
                                              as: T.Type,
                                              variant: RopeVariant = .qwen) -> (q: [T], k: [T]) {
    let HD = shape.headDim
    var qOut = q.map { T($0) }
    var kOut = k.map { T($0) }

    func process(_ data: inout [T], base: Int, weight: [Float], position: Int) {
        var sum: T = 0
        for d in 0..<HD {
            let value = data[base + d]
            sum += value * value
        }
        let inv = 1 / (sum / T(HD) + T(eps)).squareRoot()
        var head = [T](repeating: 0, count: HD)
        for d in 0..<HD { head[d] = data[base + d] * inv * T(weight[d]) }

        let rotated = variant == .qwen ? shape.rotaryDim : HD
        let half = rotated / 2
        for i in 0..<half {
            let exponent = T(-Double(2 * i) / Double(rotated))
            let angle = T(position) * T.qPow(T(shape.theta), exponent)
            let c = T.qCos(angle)
            let s = T.qSin(angle)
            let x0 = head[i]
            let x1 = head[half + i]
            data[base + i] = x0 * c - x1 * s
            data[base + half + i] = x1 * c + x0 * s
        }
        for d in rotated..<HD { data[base + d] = head[d] }
    }

    for t in 0..<shape.seqLen {
        let position = shape.position + t
        for h in 0..<shape.numQHeads {
            process(&qOut, base: (t * shape.numQHeads + h) * 2 * HD,
                    weight: qWeight, position: position)
        }
        for h in 0..<shape.numKVHeads {
            process(&kOut, base: (t * shape.numKVHeads + h) * HD,
                    weight: kWeight, position: position)
        }
    }
    return (qOut, kOut)
}

/// 2 倍幅の `q` から q だけ (または gate だけ) を抜き出す。
private func slice<T>(_ values: [T], shape: AttnShape, gate: Bool) -> [T] {
    let HD = shape.headDim
    var out: [T] = []
    out.reserveCapacity(shape.seqLen * shape.numQHeads * HD)
    for t in 0..<shape.seqLen {
        for h in 0..<shape.numQHeads {
            let base = (t * shape.numQHeads + h) * 2 * HD + (gate ? HD : 0)
            out.append(contentsOf: values[base..<(base + HD)])
        }
    }
    return out
}

// MARK: - 検査本体

func runQwenKernelCheck(tokens: Int) throws -> [CaseResult] {
    let context = try MetalContext()
    let kernels = try QwenKernels(context: context)
    let device = context.device
    var results: [CaseResult] = []

    // ---- 1. qwen_delta_qkv_prepare ------------------------------------------
    for seqLen in [1, tokens] {
        let shape = PrepareShape(seqLen: seqLen)
        let inputs = makePrepareInputs(shape, seed: 0x9E11_0001 &+ UInt64(seqLen))
        let truth = prepareReference(inputs, shape, as: Double.self)
        let floor32 = prepareReference(inputs, shape, as: Float.self)
        let got = try runPrepareOnGPU(context, kernels, inputs, shape)
        let label = seqLen == 1 ? "decode" : "prefill \(seqLen)"

        for (name, gpu, reference, floor) in [
            ("q", got.q, truth.q, floor32.q),
            ("k", got.k, truth.k, floor32.k),
            ("v", got.v, truth.v, floor32.v),
            ("conv 状態", got.state, truth.state, floor32.state),
        ] {
            let floorRel = relative(doubles(floor), doubles(reference))
            results.append(result("qwen_delta_qkv_prepare \(name) \(label)",
                                  groupSize: 0,
                                  rel: relative(doubles(gpu), doubles(reference)),
                                  tolerance: 4e-3,
                                  detail: String(format: "float32 の床 %.2e", floorRel)))
        }

        // 検出力 2 本。どちらも「正しい参照が 4e-3 で通った同じ物差し」で測る。
        let reversed = prepareReference(inputs, shape, as: Double.self, variant: .reversedTaps)
        results.append(detectionResult("qwen_delta_qkv_prepare 検出力 (conv タップの向き) \(label)",
                                       groupSize: 0,
                                       rel: relative(doubles(got.v), doubles(reversed.v)),
                                       floor: 4e-2))
        let meanNorm = prepareReference(inputs, shape, as: Double.self,
                                        variant: .meanNormalisation)
        results.append(detectionResult("qwen_delta_qkv_prepare 検出力 (l2norm の eps) \(label)",
                                       groupSize: 0,
                                       rel: relative(doubles(got.q), doubles(meanNorm.q)),
                                       floor: 4e-2))
    }

    // 状態の持ち越し: 1 回で流したものと 32 トークンずつに切ったものが**ビット一致**。
    // conv の窓は 4 なので、状態の書き戻しがずれていれば境界のトークンで必ず出る。
    do {
        let shape = PrepareShape(seqLen: tokens)
        let inputs = makePrepareInputs(shape, seed: 0x9E11_0001 &+ UInt64(tokens))
        let single = try runPrepareOnGPU(context, kernels, inputs, shape)
        let carried = try runPrepareOnGPU(context, kernels, inputs, shape, chunk: 32)
        var mismatch = 0
        for i in 0..<single.q.count where single.q[i] != carried.q[i] { mismatch += 1 }
        for i in 0..<single.k.count where single.k[i] != carried.k[i] { mismatch += 1 }
        for i in 0..<single.v.count where single.v[i] != carried.v[i] { mismatch += 1 }
        for i in 0..<single.state.count where single.state[i] != carried.state[i] { mismatch += 1 }
        results.append(result("qwen_delta_qkv_prepare 状態の持ち越し (1 回 対 32 ずつ)",
                              groupSize: 0, rel: Double(mismatch), tolerance: 0,
                              detail: "ビット一致を要求。\(mismatch) 本が不一致"))
    }

    // ---- 2. qwen_delta_gates ------------------------------------------------
    do {
        let heads = 32
        let seqLen = tokens
        var rng = QwenRNG(state: 0x9E11_0002)
        var a = [Float16](repeating: 0, count: seqLen * heads)
        var b = [Float16](repeating: 0, count: seqLen * heads)
        for i in 0..<a.count {
            a[i] = Float16(rng.normal())
            b[i] = Float16(rng.normal())
        }
        // A_log ∈ [-4, 4]、dt_bias は負に寄る (10 §5 の実測値域)。
        var aLogBits: [UInt16] = []
        var dtBiasBits: [UInt16] = []
        var aLog: [Float] = []
        var dtBias: [Float] = []
        for _ in 0..<heads {
            let a1 = Quantization.bf16Bits(Float(-4.0 + 8.0 * rng.uniform()))
            let d1 = Quantization.bf16Bits(Float(rng.normal() - 1.5))
            aLogBits.append(a1); aLog.append(Quantization.bf16ToFloat(a1))
            dtBiasBits.append(d1); dtBias.append(Quantization.bf16ToFloat(d1))
        }

        let truth = gatesReference(a: a, b: b, aLog: aLog, dtBias: dtBias,
                                   seqLen: seqLen, heads: heads, as: Double.self)
        let floor32 = gatesReference(a: a, b: b, aLog: aLog, dtBias: dtBias,
                                     seqLen: seqLen, heads: heads, as: Float.self)

        let gBuffer = device.makeBuffer(length: seqLen * heads * MemoryLayout<Float>.size,
                                        options: .storageModeShared)!
        let betaBuffer = device.makeBuffer(length: seqLen * heads * MemoryLayout<Float>.size,
                                           options: .storageModeShared)!
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw QwenCheckError.noCommandBuffer
        }
        guard kernels.encodeDeltaGates(commandBuffer: commandBuffer,
                                       a: buffer(device, a), b: buffer(device, b),
                                       aLog: buffer(device, aLogBits),
                                       dtBias: buffer(device, dtBiasBits),
                                       g: gBuffer, beta: betaBuffer,
                                       seqLen: seqLen, numVHeads: heads) else {
            throw QwenCheckError.encodeRefused("qwen_delta_gates")
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw QwenCheckError.dispatchFailed("\(error)") }

        let g = read(gBuffer, count: seqLen * heads, as: Float.self)
        let beta = read(betaBuffer, count: seqLen * heads, as: Float.self)
        // 出力は FP32 なので床は FP32 のまま。tolerance は床の 20 倍。
        let gFloor = relative(doubles(floor32.g), doubles(truth.g))
        let betaFloor = relative(doubles(floor32.beta), doubles(truth.beta))
        results.append(result("qwen_delta_gates g", groupSize: 0,
                              rel: relative(doubles(g), doubles(truth.g)),
                              tolerance: Swift.max(20 * gFloor, 1e-6),
                              detail: String(format: "float32 の床 %.2e", gFloor)))
        results.append(result("qwen_delta_gates beta", groupSize: 0,
                              rel: relative(doubles(beta), doubles(truth.beta)),
                              tolerance: Swift.max(20 * betaFloor, 1e-6),
                              detail: String(format: "float32 の床 %.2e", betaFloor)))
        let wrong = gatesReference(a: a, b: b, aLog: aLog, dtBias: dtBias,
                                   seqLen: seqLen, heads: heads, as: Double.self,
                                   dropBias: true)
        results.append(detectionResult("qwen_delta_gates 検出力 (dt_bias の脱落)",
                                       groupSize: 0,
                                       rel: relative(doubles(g), doubles(wrong.g)),
                                       floor: 1e-3))
    }

    // ---- 3. qwen_delta_norm_gate --------------------------------------------
    do {
        let heads = 32, headDim = 128
        let seqLen = tokens
        var rng = QwenRNG(state: 0x9E11_0003)
        var o = [Float16](repeating: 0, count: seqLen * heads * headDim)
        var z = [Float16](repeating: 0, count: seqLen * heads * headDim)
        for i in 0..<o.count {
            o[i] = Float16(rng.normal() * 0.8)
            z[i] = Float16(rng.normal())
        }
        let weight = BF16Vector(count: headDim, scale: 0.5, rng: &rng)
        let eps = QwenKernels.rmsEps

        let truth = normGateReference(o: o, z: z, weight: weight.values,
                                      seqLen: seqLen, heads: heads, headDim: headDim,
                                      eps: eps, as: Double.self)
        let floor32 = normGateReference(o: o, z: z, weight: weight.values,
                                        seqLen: seqLen, heads: heads, headDim: headDim,
                                        eps: eps, as: Float.self)
        let out = device.makeBuffer(length: o.count * MemoryLayout<Float16>.size,
                                    options: .storageModeShared)!
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw QwenCheckError.noCommandBuffer
        }
        guard kernels.encodeDeltaNormGate(commandBuffer: commandBuffer,
                                          o: buffer(device, o), z: buffer(device, z),
                                          weight: buffer(device, weight.bits),
                                          out: out,
                                          seqLen: seqLen, numVHeads: heads,
                                          headDim: headDim) else {
            throw QwenCheckError.encodeRefused("qwen_delta_norm_gate")
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw QwenCheckError.dispatchFailed("\(error)") }

        let got = read(out, count: o.count, as: Float16.self)
        let floorRel = relative(doubles(floor32), doubles(truth))
        results.append(result("qwen_delta_norm_gate", groupSize: 0,
                              rel: relative(doubles(got), doubles(truth)),
                              tolerance: 4e-3,
                              detail: String(format: "float32 の床 %.2e", floorRel)))
        let wrong = normGateReference(o: o, z: z, weight: weight.values,
                                      seqLen: seqLen, heads: heads, headDim: headDim,
                                      eps: eps, as: Double.self, plusOneWeight: true)
        results.append(detectionResult("qwen_delta_norm_gate 検出力 (`1+w` を足す)",
                                       groupSize: 0,
                                       rel: relative(doubles(got), doubles(wrong)),
                                       floor: 4e-2))
    }

    // ---- 4. qwen_qkv_epilogue -----------------------------------------------
    for seqLen in [1, tokens] {
        // 位置 0 から始めない: RoPE の角度が 0 だと回転が恒等になり、組の取り違えが
        // 見えなくなる。
        let shape = AttnShape(seqLen: seqLen, position: 137)
        var rng = QwenRNG(state: 0x9E11_0004 &+ UInt64(seqLen))
        var q = [Float16](repeating: 0, count: shape.qCount)
        var k = [Float16](repeating: 0, count: shape.kCount)
        for i in 0..<q.count { q[i] = Float16(rng.normal() * 0.7) }
        for i in 0..<k.count { k[i] = Float16(rng.normal() * 0.7) }
        // q_norm には `head_dim^-0.5 = 1/16` が焼き込まれている (12 §5)。
        let qWeight = BF16Vector(count: shape.headDim, scale: 1.0 / 16.0, rng: &rng)
        let kWeight = BF16Vector(count: shape.headDim, scale: 0.6, rng: &rng)
        let eps = QwenKernels.rmsEps
        let label = seqLen == 1 ? "decode" : "prefill \(seqLen)"

        let truth = epilogueReference(q: q, k: k, qWeight: qWeight.values,
                                      kWeight: kWeight.values, shape, eps: eps,
                                      as: Double.self)
        let floor32 = epilogueReference(q: q, k: k, qWeight: qWeight.values,
                                        kWeight: kWeight.values, shape, eps: eps,
                                        as: Float.self)

        let qBuffer = buffer(device, q)
        let kBuffer = buffer(device, k)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw QwenCheckError.noCommandBuffer
        }
        guard kernels.encodeQKVEpilogue(commandBuffer: commandBuffer,
                                        q: qBuffer, k: kBuffer,
                                        qWeight: buffer(device, qWeight.bits),
                                        kWeight: buffer(device, kWeight.bits),
                                        seqLen: seqLen,
                                        numQHeads: shape.numQHeads,
                                        numKVHeads: shape.numKVHeads,
                                        headDim: shape.headDim,
                                        rotaryDim: shape.rotaryDim,
                                        position: shape.position,
                                        theta: shape.theta) else {
            throw QwenCheckError.encodeRefused("qwen_qkv_epilogue")
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw QwenCheckError.dispatchFailed("\(error)") }

        let gotQ = read(qBuffer, count: shape.qCount, as: Float16.self)
        let gotK = read(kBuffer, count: shape.kCount, as: Float16.self)
        let qFloor = relative(doubles(slice(floor32.q, shape: shape, gate: false)),
                              doubles(slice(truth.q, shape: shape, gate: false)))
        results.append(result("qwen_qkv_epilogue q \(label)", groupSize: 0,
                              rel: relative(doubles(slice(gotQ, shape: shape, gate: false)),
                                            doubles(slice(truth.q, shape: shape, gate: false))),
                              tolerance: 4e-3,
                              detail: String(format: "float32 の床 %.2e", qFloor)))
        let kFloor = relative(doubles(floor32.k), doubles(truth.k))
        results.append(result("qwen_qkv_epilogue k \(label)", groupSize: 0,
                              rel: relative(doubles(gotK), doubles(truth.k)),
                              tolerance: 4e-3,
                              detail: String(format: "float32 の床 %.2e", kFloor)))

        // gate (後半 HD) は無傷であること。`attn_output_gate` が後で読むので、
        // ここで norm も RoPE も掛かってはいけない。**ビット一致**を要求する。
        let gateBefore = slice(q, shape: shape, gate: true)
        let gateAfter = slice(gotQ, shape: shape, gate: true)
        var gateMismatch = 0
        for i in 0..<gateBefore.count where gateBefore[i] != gateAfter[i] { gateMismatch += 1 }
        results.append(result("qwen_qkv_epilogue gate 側が無傷 \(label)", groupSize: 0,
                              rel: Double(gateMismatch), tolerance: 0,
                              detail: "ビット一致を要求。\(gateMismatch) 本が不一致"))

        // 検出力: Gemma の組と分母を流用した参照から桁違いに離れること。
        let gemma = epilogueReference(q: q, k: k, qWeight: qWeight.values,
                                      kWeight: kWeight.values, shape, eps: eps,
                                      as: Double.self, variant: .gemma)
        results.append(detectionResult("qwen_qkv_epilogue 検出力 (Gemma の組 (i, HD/2+i)) \(label)",
                                       groupSize: 0,
                                       rel: relative(doubles(gotK), doubles(gemma.k)),
                                       floor: 4e-2))
    }

    // ---- 5. qwen_attn_output_gate -------------------------------------------
    do {
        let shape = AttnShape(seqLen: tokens)
        var rng = QwenRNG(state: 0x9E11_0005)
        var o = [Float16](repeating: 0, count: shape.seqLen * shape.numQHeads * shape.headDim)
        var qg = [Float16](repeating: 0, count: shape.qCount)
        for i in 0..<o.count { o[i] = Float16(rng.normal()) }
        for i in 0..<qg.count { qg[i] = Float16(rng.normal()) }

        func reference<T: QwenScalar>(as: T.Type) -> [T] {
            var out = [T](repeating: 0, count: o.count)
            for t in 0..<shape.seqLen {
                for h in 0..<shape.numQHeads {
                    for d in 0..<shape.headDim {
                        let index = (t * shape.numQHeads + h) * shape.headDim + d
                        let gate = (t * shape.numQHeads + h) * 2 * shape.headDim
                            + shape.headDim + d
                        out[index] = T(o[index]) * qSigmoid(T(qg[gate]))
                    }
                }
            }
            return out
        }
        let truth = reference(as: Double.self)
        let floor32 = reference(as: Float.self)

        let oBuffer = buffer(device, o)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw QwenCheckError.noCommandBuffer
        }
        guard kernels.encodeAttnOutputGate(commandBuffer: commandBuffer,
                                           o: oBuffer, qGate: buffer(device, qg),
                                           seqLen: shape.seqLen,
                                           numQHeads: shape.numQHeads,
                                           headDim: shape.headDim) else {
            throw QwenCheckError.encodeRefused("qwen_attn_output_gate")
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw QwenCheckError.dispatchFailed("\(error)") }
        let got = read(oBuffer, count: o.count, as: Float16.self)
        results.append(result("qwen_attn_output_gate", groupSize: 0,
                              rel: relative(doubles(got), doubles(truth)),
                              tolerance: 4e-3,
                              detail: String(format: "float32 の床 %.2e",
                                             relative(doubles(floor32), doubles(truth)))))
    }

    // ---- 6. qwen_moe_shared_gate / 7. qwen_silu_mul --------------------------
    do {
        let hidden = 2048
        var rng = QwenRNG(state: 0x9E11_0006)
        var y = [Float16](repeating: 0, count: hidden)
        var x = [Float16](repeating: 0, count: hidden)
        for i in 0..<hidden {
            y[i] = Float16(rng.normal())
            // 内積が飽和しないよう 1/sqrt(D) で割る (実物の logit の値域に近い)。
            x[i] = Float16(rng.normal() / Double(hidden).squareRoot())
        }
        let weight = BF16Vector(count: hidden, scale: 1.0, rng: &rng)

        func sharedReference<T: QwenScalar>(as: T.Type) -> [T] {
            var acc: T = 0
            for i in 0..<hidden { acc += T(x[i]) * T(weight.values[i]) }
            let scale = qSigmoid(acc)
            return (0..<hidden).map { T(y[$0]) * scale }
        }
        let truth = sharedReference(as: Double.self)
        let floor32 = sharedReference(as: Float.self)

        let yBuffer = buffer(device, y)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw QwenCheckError.noCommandBuffer
        }
        guard kernels.encodeMoESharedGate(commandBuffer: commandBuffer,
                                          y: yBuffer, x: buffer(device, x),
                                          weight: buffer(device, weight.bits),
                                          hiddenSize: hidden) else {
            throw QwenCheckError.encodeRefused("qwen_moe_shared_gate")
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw QwenCheckError.dispatchFailed("\(error)") }
        let got = read(yBuffer, count: hidden, as: Float16.self)
        results.append(result("qwen_moe_shared_gate", groupSize: 0,
                              rel: relative(doubles(got), doubles(truth)),
                              tolerance: 4e-3,
                              detail: String(format: "float32 の床 %.2e",
                                             relative(doubles(floor32), doubles(truth)))))

        // silu_mul
        let count = 4096
        var gate = [Float16](repeating: 0, count: count)
        var up = [Float16](repeating: 0, count: count)
        for i in 0..<count {
            gate[i] = Float16(rng.normal() * 1.5)
            up[i] = Float16(rng.normal())
        }
        func siluReference<T: QwenScalar>(as: T.Type) -> [T] {
            (0..<count).map { qSilu(T(gate[$0])) * T(up[$0]) }
        }
        let siluTruth = siluReference(as: Double.self)
        let siluFloor = siluReference(as: Float.self)
        let out = device.makeBuffer(length: count * MemoryLayout<Float16>.size,
                                    options: .storageModeShared)!
        guard let siluCommandBuffer = context.queue.makeCommandBuffer() else {
            throw QwenCheckError.noCommandBuffer
        }
        guard kernels.encodeSiluMul(commandBuffer: siluCommandBuffer,
                                    gate: buffer(device, gate), up: buffer(device, up),
                                    out: out, count: count) else {
            throw QwenCheckError.encodeRefused("qwen_silu_mul")
        }
        siluCommandBuffer.commit()
        siluCommandBuffer.waitUntilCompleted()
        if let error = siluCommandBuffer.error {
            throw QwenCheckError.dispatchFailed("\(error)")
        }
        let siluGot = read(out, count: count, as: Float16.self)
        results.append(result("qwen_silu_mul", groupSize: 0,
                              rel: relative(doubles(siluGot), doubles(siluTruth)),
                              tolerance: 4e-3,
                              detail: String(format: "float32 の床 %.2e",
                                             relative(doubles(siluFloor), doubles(siluTruth)))))
    }

    return results
}

// MARK: - 時間 (`docs/qwen35moe/05-RISKS.md` §2 の中止線に足す分)
//
// `qwen_delta_rule` の 30 層 125.7 ms は線形注意の**本体だけ**の数字で、その
// 前後の `qwen_delta_qkv_prepare` と `qwen_delta_norm_gate` は入っていない。
// prefill のチャンク 1 本には両方とも 30 層ぶん走るので、中止線 150 ms に対する
// 残りを知るにはここも測る。`qwen_qkv_epilogue` は full attention の 10 層。
//
// カーネル 1 本ずつのマイクロベンチなので `bench.sh` の作法 (temp / クールダウン)
// の対象ではない。数字は「このカーネルが 1 層に何 ms 使うか」であって、モデルの
// 速度ではない。

func runQwenKernelBench(tokens: Int, iterations: Int) throws {
    let context = try MetalContext()
    let kernels = try QwenKernels(context: context)
    let device = context.device
    print("# qwen.metal マイクロベンチ  \(iterations) 回の中央値")
    print("  カーネル                           T      1 層 (ms)   層数ぶん (ms)")

    func measure(_ label: String, layers: Int, seqLen: Int,
                 _ encode: (MTLCommandBuffer) -> Bool) throws {
        var samples: [Double] = []
        for _ in 0..<iterations {
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw QwenCheckError.noCommandBuffer
            }
            guard encode(commandBuffer) else {
                throw QwenCheckError.encodeRefused(label)
            }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw QwenCheckError.dispatchFailed("\(error)")
            }
            samples.append((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1e3)
        }
        samples.sort()
        let median = samples[samples.count / 2]
        print("  \(label.padding(toLength: 30, withPad: " ", startingAt: 0))"
              + "\(String(format: "%6d", seqLen))"
              + "\(String(format: "%12.3f", median))"
              + "\(String(format: "%14.1f", median * Double(layers)))")
    }

    for seqLen in [1, tokens] {
        let shape = PrepareShape(seqLen: seqLen)
        let inputs = makePrepareInputs(shape, seed: 0x1234)
        let qkv = buffer(device, inputs.qkv)
        let weight = buffer(device, inputs.convWeight.bits)
        let state = buffer(device, inputs.state0)
        let stateBack = buffer(device, inputs.state0)
        let q = device.makeBuffer(length: shape.qkCount * MemoryLayout<Float16>.size,
                                  options: .storageModeShared)!
        let k = device.makeBuffer(length: shape.qkCount * MemoryLayout<Float16>.size,
                                  options: .storageModeShared)!
        let v = device.makeBuffer(length: shape.vCount * MemoryLayout<Float16>.size,
                                  options: .storageModeShared)!
        // R (1 threadgroup が持つトークン数) を振る。窓 4 の読み直しが増える代わりに
        // threadgroup 数が増える — どちらが効くかは測らないと分からない。
        for tokensPerGroup in [16, 32, 64, 128] where tokensPerGroup <= max(seqLen, 16) {
            try measure("qwen_delta_qkv_prepare R=\(tokensPerGroup)",
                        layers: 30, seqLen: seqLen) { commandBuffer in
                kernels.encodeDeltaQKVPrepare(commandBuffer: commandBuffer,
                                              qkv: qkv, convWeight: weight,
                                              stateIn: state, stateOut: stateBack,
                                              q: q, k: k, v: v,
                                              seqLen: seqLen,
                                              numKHeads: shape.numKHeads,
                                              numVHeads: shape.numVHeads,
                                              headDim: shape.headDim,
                                              tokensPerGroup: tokensPerGroup)
            }
        }

        let heads = 32, headDim = 128
        var rng = QwenRNG(state: 0x5678)
        var o = [Float16](repeating: 0, count: seqLen * heads * headDim)
        for i in 0..<o.count { o[i] = Float16(rng.normal()) }
        let oBuffer = buffer(device, o)
        let zBuffer = buffer(device, o)
        let normWeight = BF16Vector(count: headDim, scale: 0.5, rng: &rng)
        let normOut = device.makeBuffer(length: o.count * MemoryLayout<Float16>.size,
                                        options: .storageModeShared)!
        try measure("qwen_delta_norm_gate", layers: 30, seqLen: seqLen) { commandBuffer in
            kernels.encodeDeltaNormGate(commandBuffer: commandBuffer,
                                        o: oBuffer, z: zBuffer,
                                        weight: buffer(device, normWeight.bits),
                                        out: normOut,
                                        seqLen: seqLen, numVHeads: heads, headDim: headDim)
        }

        let attn = AttnShape(seqLen: seqLen, position: 137)
        var qFull = [Float16](repeating: 0, count: attn.qCount)
        var kFull = [Float16](repeating: 0, count: attn.kCount)
        for i in 0..<qFull.count { qFull[i] = Float16(rng.normal()) }
        for i in 0..<kFull.count { kFull[i] = Float16(rng.normal()) }
        let qWeight = BF16Vector(count: attn.headDim, scale: 1.0 / 16.0, rng: &rng)
        let kWeight = BF16Vector(count: attn.headDim, scale: 0.6, rng: &rng)
        let qBuffer = buffer(device, qFull)
        let kBuffer = buffer(device, kFull)
        let qWeightBuffer = buffer(device, qWeight.bits)
        let kWeightBuffer = buffer(device, kWeight.bits)
        try measure("qwen_qkv_epilogue", layers: 10, seqLen: seqLen) { commandBuffer in
            kernels.encodeQKVEpilogue(commandBuffer: commandBuffer,
                                      q: qBuffer, k: kBuffer,
                                      qWeight: qWeightBuffer, kWeight: kWeightBuffer,
                                      seqLen: seqLen,
                                      numQHeads: attn.numQHeads,
                                      numKVHeads: attn.numKVHeads,
                                      headDim: attn.headDim,
                                      rotaryDim: attn.rotaryDim,
                                      position: attn.position,
                                      theta: attn.theta)
        }

        var a = [Float16](repeating: 0, count: seqLen * heads)
        for i in 0..<a.count { a[i] = Float16(rng.normal()) }
        let aBuffer = buffer(device, a)
        let aLog = BF16Vector(count: heads, scale: 2.0, rng: &rng)
        let dtBias = BF16Vector(count: heads, scale: 1.0, rng: &rng)
        let aLogBuffer = buffer(device, aLog.bits)
        let dtBiasBuffer = buffer(device, dtBias.bits)
        let gBuffer = device.makeBuffer(length: seqLen * heads * MemoryLayout<Float>.size,
                                        options: .storageModeShared)!
        let betaBuffer = device.makeBuffer(length: seqLen * heads * MemoryLayout<Float>.size,
                                           options: .storageModeShared)!
        try measure("qwen_delta_gates", layers: 30, seqLen: seqLen) { commandBuffer in
            kernels.encodeDeltaGates(commandBuffer: commandBuffer,
                                     a: aBuffer, b: aBuffer,
                                     aLog: aLogBuffer, dtBias: dtBiasBuffer,
                                     g: gBuffer, beta: betaBuffer,
                                     seqLen: seqLen, numVHeads: heads)
        }
    }
}
