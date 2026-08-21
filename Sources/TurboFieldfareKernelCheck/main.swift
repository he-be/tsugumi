import Foundation
import Metal
import TurboFieldfare
import TurboFieldfareValidationSupport

// Numeric self-check for the affine-INT4 kernels at a chosen group size.
//
// Why this exists as an executable rather than a test: `swift test` cannot run
// in this environment (the swift-testing module is unavailable), and the bug
// class this guards against is invisible to the end-to-end checks. The
// vectorized INT4 GEMV bodies split each SIMD group's 128-byte block by a
// hardcoded "8 lanes per group" assumption that is only true at group 64
// (see PLAN_QAT §3-1-a). Getting that wrong does not crash and does not
// produce NaN — it reads the wrong per-group scale and yields plausible but
// degraded output, which is exactly what a quality-motivated checkpoint swap
// cannot afford to confuse with the checkpoint itself.
//
// Every case runs at both group sizes. Group 64 asserts the existing behavior
// is untouched; group 32 is the new path.
//
//   swift run -c release TurboFieldfareKernelCheck
//   swift run -c release TurboFieldfareKernelCheck --group-size 32

// MARK: - Result plumbing

struct CaseResult {
    let name: String
    let groupSize: Int
    let relativeError: Double
    let tolerance: Double
    let detail: String
    /// A negative control: the comparison is expected to *fail*, and the case
    /// passes when the error clears the bar instead of staying under it.
    let inverted: Bool

    var passed: Bool { inverted ? relativeError >= tolerance : relativeError <= tolerance }
}

func result(_ name: String, groupSize: Int, rel: Double, tolerance: Double,
            detail: String = "") -> CaseResult {
    CaseResult(name: name, groupSize: groupSize,
               relativeError: rel, tolerance: tolerance, detail: detail,
               inverted: false)
}

/// A case that proves a comparison has detection power: the same GPU output is
/// scored against a deliberately wrong reference, and passing means the score
/// is at least `floor` — comfortably clear of the tolerance the positive case
/// just met. A check that has never been seen to fail is not evidence
/// (`PLAN_VISION.md` §6-3).
func detectionResult(_ name: String, groupSize: Int, rel: Double, floor: Double,
                     detail: String = "") -> CaseResult {
    CaseResult(name: name, groupSize: groupSize,
               relativeError: rel, tolerance: floor,
               detail: detail.isEmpty ? "must exceed" : "must exceed; " + detail,
               inverted: true)
}

// MARK: - Comparison

/// NaN-safe relative error.
///
/// `RelError.compute` cannot be used here: it folds the per-element diff with
/// `max(_:_:)`, and `max(0, .nan)` returns 0 in Swift, so an output full of
/// NaN scores a *perfect* relative error of zero. A broken kernel that reads
/// past its row produces exactly that, which would turn the case this harness
/// exists for into a silent PASS. Non-finite output is a failure here, and a
/// reference with no signal in it is a harness bug rather than a result.
func relativeError(actual: [Float], reference: [Float]) -> Double {
    precondition(actual.count == reference.count, "length mismatch")
    var maxDiff = 0.0
    var refNorm = 0.0
    var nonFinite = 0
    for i in 0..<actual.count {
        let a = Double(actual[i])
        let r = Double(reference[i])
        precondition(r.isFinite, "reference is not finite at \(i) — harness bug")
        refNorm = Swift.max(refNorm, abs(r))
        if !a.isFinite { nonFinite += 1; continue }
        maxDiff = Swift.max(maxDiff, abs(a - r))
    }
    precondition(refNorm > 1e-4,
                 "reference has no signal (max |ref| = \(refNorm)) — harness bug")
    if nonFinite > 0 { return .infinity }
    return maxDiff / refNorm
}

/// A command buffer that failed left its output buffer untouched, so the
/// comparison downstream would be measuring stale memory rather than the
/// kernel. Surface it instead.
func waitAndCheck(_ commandBuffer: MTLCommandBuffer, _ label: String) {
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        fatalError("\(label): command buffer failed — \(error)")
    }
}

// MARK: - Shared helpers

/// Flattens per-row affine tensors into the (weights, scales, biases) triple
/// the kernels bind, matching the packed sub-tensor layout.
func packRows(_ rows: [Quantization.Int4AffineRow])
    -> (packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
    (rows.flatMap(\.packed), rows.flatMap(\.scales), rows.flatMap(\.biases))
}

/// The affine scheme every case in this run builds fixtures for and compiles the
/// library with. `sym` drops the bias array from the packed model
/// (`docs/mtp/44-W1-WEIGHT-DIET.md`), so the *same* cases have to pass with the
/// kernels deriving `-8 * scale` instead of loading it. The fixture still
/// carries the bias array and the reference is still `dequantizeInt4Affine`, so
/// a kernel that got the derivation wrong fails against a bias it can see but
/// no longer reads.
///
/// `nonisolated(unsafe)`: the driver is a single-threaded command-line pass that
/// sets this before building each group of cases.
nonisolated(unsafe) var affineScheme: Quantization.AffineScheme = .affine

func quantizedRows(count: Int, n: Int, groupSize: Int,
                   rng: inout SplitMix64) -> [Quantization.Int4AffineRow] {
    var rows: [Quantization.Int4AffineRow] = []
    rows.reserveCapacity(count)
    for _ in 0..<count {
        let raw = (0..<n).map { _ in rng.uniform(-0.5, 0.5) }
        rows.append(affineScheme == .sym
            ? Quantization.quantizeInt4Symmetric(raw, groupSize: groupSize)
            : Quantization.quantizeInt4Affine(raw, groupSize: groupSize))
    }
    return rows
}

func makeContext(groupSize: Int) throws -> MetalContext {
    let context = try MetalContext()
    try context.setAffineGroupSize(groupSize)
    try context.setAffineScheme(affineScheme)
    return context
}

// MARK: - Case 1/2 — dequant_int4_gemv_simd

/// `m` rows of `n` columns through the decode INT4 GEMV, against the vDSP
/// reference. `n = 2112` and `n = 704` are the shapes whose group count is not
/// a whole number of vectorized blocks, so they exercise the scalar tail.
func checkInt4GEMV(m: Int, n: Int, groupSize: Int, seed: UInt64) throws -> CaseResult {
    var rng = SeedTree(seed).key("int4-gemv-m\(m)-n\(n)-g\(groupSize)")
    let rows = quantizedRows(count: m, n: n, groupSize: groupSize, rng: &rng)
    let (packed, scales, biases) = packRows(rows)

    let xFp16 = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
    let xRef = xFp16.map { Float($0) }

    let context = try makeContext(groupSize: groupSize)
    let kernel = try DequantInt4GEMV(context: context)

    guard let wBuf = context.device.makeBuffer(bytes: packed, length: packed.count,
                                               options: .storageModeShared),
          let sBuf = context.device.makeBuffer(
            bytes: scales, length: scales.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared),
          let bBuf = context.device.makeBuffer(
            bytes: biases, length: biases.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared),
          let xBuf = Fp16Buffer.make(context.device, halves: xFp16),
          let yBuf = Fp16Buffer.make(context.device, count: m),
          let cmd = context.queue.makeCommandBuffer() else {
        fatalError("buffer allocation failed")
    }

    kernel.encode(commandBuffer: cmd,
                  weights: wBuf, scales: sBuf, biases: bBuf,
                  x: xBuf, y: yBuf,
                  m: UInt32(m), n: UInt32(n))
    waitAndCheck(cmd, "int4-gemv m=\(m) n=\(n) g=\(groupSize)")

    let reference = DequantInt4GemvRef.apply(weightRows: rows, x: xRef, n: n,
                                             groupSize: groupSize)
    let actual = Fp16Buffer.read(yBuf, count: m)
    let groups = n / groupSize
    let blockWidth = 256 / groupSize
    let tail = groups % blockWidth
    return result("int4-gemv m=\(m) n=\(n)", groupSize: groupSize,
           rel: relativeError(actual: actual, reference: reference),
           tolerance: Double(Tolerance.fp16Reduction),
           detail: "groups=\(groups) blocks=\(groups / blockWidth) tail=\(tail)")
}

// MARK: - Case 3 — routed MoE (gate/up at N=D, down at N=F)

/// Drives the production two-phase routed path at the real decode shapes.
/// D=2816 covers the block-only case; F=704 covers the tail at both group
/// sizes (11 groups → 2 blocks + 3 at group 64, 22 groups → 2 blocks + 6 at
/// group 32).
func checkRoutedMoE(d: Int, f: Int, groupSize: Int, seed: UInt64) throws -> CaseResult {
    let topK = 8
    var rng = SeedTree(seed).key("routed-moe-d\(d)-f\(f)-g\(groupSize)")

    func matrix(rows: Int, columns: Int) -> [[Float]] {
        (0..<rows).map { _ in (0..<columns).map { _ in rng.uniform(-0.4, 0.4) } }
    }

    var gates: [[[Float]]] = []
    var ups: [[[Float]]] = []
    var downs: [[[Float]]] = []
    for _ in 0..<topK {
        gates.append(matrix(rows: f, columns: d))
        ups.append(matrix(rows: f, columns: d))
        downs.append(matrix(rows: d, columns: f))
    }
    let x: [Float] = (0..<d).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
    let residual: [Float] = (0..<d).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
    let routingWeights: [Float] = (0..<topK).map { Float(Float16(0.04 + Float($0) * 0.015)) }

    func quantize(_ matrices: [[[Float]]]) -> [[Quantization.Int4AffineRow]] {
        matrices.map { rows in
            rows.map {
                affineScheme == .sym
                    ? Quantization.quantizeInt4Symmetric($0, groupSize: groupSize)
                    : Quantization.quantizeInt4Affine($0, groupSize: groupSize)
            }
        }
    }
    let quantGates = quantize(gates)
    let quantUps = quantize(ups)
    let quantDowns = quantize(downs)

    let expected = MoeRef.applyStreamedRouted(
        x: x, residual: residual,
        routedGate: quantGates, routedUp: quantUps, routedDown: quantDowns,
        indices: Array(0..<topK), routingWeights: routingWeights,
        d: d, f: f, groupSize: groupSize)

    // One contiguous blob per expert, sub-tensors in the packed order the
    // argument buffer expects.
    var blobBytes: [[UInt8]] = []
    var offsets: MoEExpertOffsets?
    for expert in 0..<topK {
        var bytes: [UInt8] = []
        func appendBytes(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func appendHalves(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        let g = packRows(quantGates[expert])
        let u = packRows(quantUps[expert])
        let dn = packRows(quantDowns[expert])
        let gateW = UInt32(bytes.count); appendBytes(g.packed)
        let gateS = UInt32(bytes.count); appendHalves(g.scales)
        let gateB = UInt32(bytes.count); appendHalves(g.biases)
        let upW = UInt32(bytes.count); appendBytes(u.packed)
        let upS = UInt32(bytes.count); appendHalves(u.scales)
        let upB = UInt32(bytes.count); appendHalves(u.biases)
        let downW = UInt32(bytes.count); appendBytes(dn.packed)
        let downS = UInt32(bytes.count); appendHalves(dn.scales)
        let downB = UInt32(bytes.count); appendHalves(dn.biases)
        blobBytes.append(bytes)
        offsets = MoEExpertOffsets(gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                                   upWOff: upW, upSOff: upS, upBOff: upB,
                                   downWOff: downW, downSOff: downS, downBOff: downB)
    }
    guard let routedOffsets = offsets else { fatalError("no experts") }

    let context = try makeContext(groupSize: groupSize)
    let kernel = try MoE(context: context)
    let routedBuffers = blobBytes.compactMap {
        context.device.makeBuffer(bytes: $0, length: $0.count,
                                  options: .storageModeShared)
    }
    guard routedBuffers.count == topK,
          let xBuffer = Fp16Buffer.make(context.device, values: x),
          let residualBuffer = Fp16Buffer.make(context.device, values: residual),
          let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
          let acts = Fp16Buffer.make(context.device, count: topK * f),
          let output = Fp16Buffer.make(context.device, count: d),
          let argumentBuffer = kernel.makeRoutedArgumentBuffer(
            routedBlobs: routedBuffers, topK: UInt32(topK)),
          let cmd = context.queue.makeCommandBuffer() else {
        fatalError("buffer allocation failed")
    }

    kernel.encodeRoutedPersistentPhase1U16Load(
        commandBuffer: cmd,
        routedArgBuffer: argumentBuffer, routedBlobs: routedBuffers,
        routedOffsets: routedOffsets,
        x: xBuffer, acts: acts,
        d: UInt32(d), f: UInt32(f), topK: UInt32(topK))
    kernel.encodeRoutedPersistentPhase2Reduce(
        commandBuffer: cmd,
        routedArgBuffer: argumentBuffer, routedBlobs: routedBuffers,
        routedOffsets: routedOffsets,
        acts: acts, routingWeights: routingBuffer, residual: residualBuffer,
        y: output,
        d: UInt32(d), f: UInt32(f), topK: UInt32(topK))
    waitAndCheck(cmd, "routed-moe d=\(d) f=\(f) g=\(groupSize)")

    let actual = Fp16Buffer.read(output, count: d)
    return result("routed-moe d=\(d) f=\(f) topK=\(topK)", groupSize: groupSize,
           rel: relativeError(actual: actual, reference: expected),
           tolerance: Double(Tolerance.fp16ChainedReduction),
           detail: "gate/up groups=\(d / groupSize) down groups=\(f / groupSize)")
}

// MARK: - Case 4 — fused greedy LM head

/// The LM head carries its own copy of the vectorized block loop
/// (`logit.metal`), so a fix applied to the other copies can be forgotten
/// here. It only exposes an argmax, which is a coarser signal than a relative
/// error, so this runs several independent draws: a kernel reading the wrong
/// per-group scales has to agree with the reference on every one of them.
func checkLMHeadGreedy(d: Int, vocab: Int, draws: Int, groupSize: Int,
                       seed: UInt64) throws -> CaseResult {
    let context = try makeContext(groupSize: groupSize)
    let chain = try LMHeadChainInt4(context: context, maxD: d, maxVocab: vocab)
    var mismatches: [String] = []

    for draw in 0..<draws {
        var rng = SeedTree(seed &+ UInt64(draw)).key("lm-head-d\(d)-v\(vocab)-g\(groupSize)")
        let rows = quantizedRows(count: vocab, n: d, groupSize: groupSize, rng: &rng)
        let (packed, scales, biases) = packRows(rows)

        let hiddenFp16 = (0..<d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let normFp32 = (0..<d).map { _ in rng.uniform(0.5, 1.5) }
        let normBF16 = normFp32.map { Quantization.bf16Bits($0) }

        guard let hidden = context.device.makeBuffer(
                bytes: hiddenFp16, length: hiddenFp16.count * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let norm = context.device.makeBuffer(
                bytes: normBF16, length: normBF16.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared),
              let weights = context.device.makeBuffer(
                bytes: packed, length: packed.count, options: .storageModeShared),
              let scaleBuffer = context.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared),
              let biasBuffer = context.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared),
              let outToken = context.device.makeBuffer(
                length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let cmd = context.queue.makeCommandBuffer() else {
            fatalError("buffer allocation failed")
        }

        chain.encodeGreedyDecode(commandBuffer: cmd,
                                 hidden: hidden, normWeight: norm,
                                 weights: weights, scales: scaleBuffer,
                                 biases: biasBuffer, outToken: outToken,
                                 d: UInt32(d), vocab: UInt32(vocab))
        waitAndCheck(cmd, "lm-head d=\(d) v=\(vocab) g=\(groupSize) draw \(draw)")
        let actual = outToken.contents().load(as: UInt32.self)

        // CPU reference: the same RMSNorm the chain applies, then argmax over
        // the dequantized rows.
        let hiddenFp32 = hiddenFp16.map { Float($0) }
        let normed = RmsNormRef.apply(x: hiddenFp32,
                                      weight: normFp32.map { Quantization.bf16ToFloat(Quantization.bf16Bits($0)) },
                                      eps: 1e-6)
        let logits = DequantInt4GemvRef.apply(weightRows: rows, x: normed, n: d,
                                              groupSize: groupSize)
        var best = 0
        for i in 1..<logits.count where logits[i] > logits[best] { best = i }
        if actual != UInt32(best) {
            mismatches.append("draw \(draw): gpu=\(actual) cpu=\(best)")
        }
    }

    // Encoded as a relative error so it shares the reporting path: 0 when every
    // draw agrees, 1 otherwise.
    return result("lm-head-greedy d=\(d) v=\(vocab) draws=\(draws)", groupSize: groupSize,
           rel: mismatches.isEmpty ? 0 : 1, tolerance: 0,
           detail: mismatches.isEmpty
               ? "argmax agrees on all \(draws) draws"
               : mismatches.joined(separator: "; "))
}

// MARK: - Case 5 — decode router (INT8 affine and BF16)

/// CPU reference for the router: logits, top-k with the kernel's tie-break,
/// softmax over the survivors, then the per-expert gain. Plain loops — this
/// must not share structure with the kernel it checks.
func routerReference(weightRows: [[Float]], x: [Float], numExperts: Int, topK: Int,
                     perExpertScale: [Float]) -> (indices: [Int], weights: [Float]) {
    var logits = [Float](repeating: 0, count: numExperts)
    for e in 0..<numExperts {
        var acc: Float = 0
        for i in 0..<x.count { acc += weightRows[e][i] * x[i] }
        logits[e] = acc
    }
    // Descending by score, ties broken by the lower expert index.
    let order = (0..<numExperts).sorted {
        logits[$0] == logits[$1] ? $0 < $1 : logits[$0] > logits[$1]
    }
    let chosen = Array(order.prefix(topK))
    let maxScore = logits[chosen[0]]
    let exps = chosen.map { expf(logits[$0] - maxScore) }
    let sum = exps.reduce(0, +)
    let weights = zip(chosen, exps).map { ($1 / sum) * perExpertScale[$0] }
    return (chosen, weights)
}

/// The router GEMV walks a row in fixed 64-element steps, so at group 32 two
/// affine groups share one step — the same class of geometry assumption as
/// §3-1-a, in a kernel that check missed. The BF16 variant has no group
/// structure at all; it runs here to prove the QAT router path is right before
/// any checkpoint exists to test it against.
func checkRouter(weightBits: Int, d: Int, numExperts: Int, draws: Int,
                 groupSize: Int, seed: UInt64) throws -> CaseResult {
    let topK = 8
    let label = "router-\(weightBits == 16 ? "bf16" : "int8") d=\(d) experts=\(numExperts)"
    let context = try makeContext(groupSize: groupSize)
    let kernel = try MoE(context: context, routerWeightBits: weightBits)

    var worstRel = 0.0
    var mismatches: [String] = []

    for draw in 0..<draws {
        var rng = SeedTree(seed &+ UInt64(draw)).key("\(label)-g\(groupSize)")
        let raw = (0..<numExperts).map { _ in (0..<d).map { _ in rng.uniform(-0.2, 0.2) } }

        // Weight bytes plus, for INT8, the affine companions.
        var weightBytes: [UInt8] = []
        var scaleBits: [UInt16] = []
        var biasBits: [UInt16] = []
        var dequantized: [[Float]] = []
        for row in raw {
            if weightBits == 16 {
                let bits = row.map { Quantization.bf16Bits($0) }
                for value in bits {
                    weightBytes.append(UInt8(truncatingIfNeeded: value))
                    weightBytes.append(UInt8(truncatingIfNeeded: value >> 8))
                }
                dequantized.append(bits.map { Quantization.bf16ToFloat($0) })
            } else {
                let quantized = Quantization.quantizeInt8Affine(row, groupSize: groupSize)
                weightBytes.append(contentsOf: quantized.packed)
                scaleBits.append(contentsOf: quantized.scales)
                biasBits.append(contentsOf: quantized.biases)
                dequantized.append(Quantization.dequantizeInt8Affine(
                    quantized, n: d, groupSize: groupSize))
            }
        }

        let hiddenFp16 = (0..<d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let effectiveScaleF = (0..<d).map { _ in rng.uniform(0.5, 1.5) }
        let effectiveScaleBits = effectiveScaleF.map { Quantization.bf16Bits($0) }
        let perExpertScaleF = (0..<numExperts).map { _ in rng.uniform(0.8, 1.2) }
        let perExpertScaleBits = perExpertScaleF.map { Quantization.bf16Bits($0) }

        // The kernel folds effective_scale into the activation, not the weight.
        let x = (0..<d).map {
            Float(hiddenFp16[$0]) * Quantization.bf16ToFloat(effectiveScaleBits[$0])
        }
        let expected = routerReference(
            weightRows: dequantized, x: x, numExperts: numExperts, topK: topK,
            perExpertScale: perExpertScaleBits.map { Quantization.bf16ToFloat($0) })

        func halfBuffer(_ values: [UInt16]) -> MTLBuffer? {
            context.device.makeBuffer(bytes: values,
                                      length: values.count * MemoryLayout<UInt16>.stride,
                                      options: .storageModeShared)
        }
        guard let wBuf = context.device.makeBuffer(
                bytes: weightBytes, length: weightBytes.count,
                options: .storageModeShared),
              let esBuf = halfBuffer(effectiveScaleBits),
              let pesBuf = halfBuffer(perExpertScaleBits),
              let hidden = Fp16Buffer.make(context.device, halves: hiddenFp16),
              let outIndices = context.device.makeBuffer(
                length: topK * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let outWeights = Fp16Buffer.make(context.device, count: topK),
              let cmd = context.queue.makeCommandBuffer() else {
            fatalError("buffer allocation failed")
        }

        if weightBits == 16 {
            kernel.encodeRouterGemma4BF16(
                commandBuffer: cmd, weights: wBuf, hidden: hidden,
                effectiveScale: esBuf, perExpertScale: pesBuf,
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(numExperts), d: UInt32(d), topK: UInt32(topK))
        } else {
            guard let sBuf = halfBuffer(scaleBits), let bBuf = halfBuffer(biasBits) else {
                fatalError("buffer allocation failed")
            }
            kernel.encodeRouterGemma4(
                commandBuffer: cmd, weights: wBuf, scales: sBuf, biases: bBuf,
                hidden: hidden, effectiveScale: esBuf, perExpertScale: pesBuf,
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(numExperts), d: UInt32(d), topK: UInt32(topK))
        }
        waitAndCheck(cmd, "\(label) g=\(groupSize) draw \(draw)")

        let gpuIndices = (0..<topK).map {
            Int(outIndices.contents().load(fromByteOffset: $0 * MemoryLayout<UInt32>.stride,
                                           as: UInt32.self))
        }
        if gpuIndices != expected.indices {
            mismatches.append("draw \(draw): gpu=\(gpuIndices) cpu=\(expected.indices)")
            continue
        }
        worstRel = Swift.max(worstRel, relativeError(
            actual: Fp16Buffer.read(outWeights, count: topK),
            reference: expected.weights))
    }

    let tolerance = Double(Tolerance.fp16Reduction)
    if !mismatches.isEmpty {
        return result(label, groupSize: groupSize, rel: .infinity, tolerance: tolerance,
                      detail: "expert selection differs — " + mismatches.joined(separator: "; "))
    }
    return result(label, groupSize: groupSize, rel: worstRel, tolerance: tolerance,
                  detail: "top-\(topK) agrees on all \(draws) draws"
                      + (weightBits == 16 ? "" : ", groups=\(d / groupSize)"))
}

// MARK: - Case 6 — INT8 kernels

/// The INT8 GEMV and the fused INT8 shared-expert carry the same fixed-64
/// -element step as the router did (§3-1-a-2). Neither is reachable from
/// either checkpoint we run — the current pin is group 64, and the QAT one is
/// 4-bit throughout — which is precisely why they went unchecked twice. They
/// are on the harness now because they *can* run at group 32, not because
/// something runs them.
func packInt8Rows(_ rows: [Quantization.Int8AffineRow])
    -> (packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
    (rows.flatMap(\.packed), rows.flatMap(\.scales), rows.flatMap(\.biases))
}

func quantizedInt8Rows(count: Int, n: Int, groupSize: Int,
                       rng: inout SplitMix64) -> [Quantization.Int8AffineRow] {
    (0..<count).map { _ in
        Quantization.quantizeInt8Affine((0..<n).map { _ in rng.uniform(-0.5, 0.5) },
                                        groupSize: groupSize)
    }
}

func checkInt8GEMV(m: Int, n: Int, groupSize: Int, seed: UInt64) throws -> CaseResult {
    var rng = SeedTree(seed).key("int8-gemv-m\(m)-n\(n)-g\(groupSize)")
    let rows = quantizedInt8Rows(count: m, n: n, groupSize: groupSize, rng: &rng)
    let (packed, scales, biases) = packInt8Rows(rows)

    let xFp16 = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
    let context = try makeContext(groupSize: groupSize)
    let kernel = try DequantInt8GEMV(context: context)

    guard let wBuf = context.device.makeBuffer(bytes: packed, length: packed.count,
                                               options: .storageModeShared),
          let sBuf = context.device.makeBuffer(
            bytes: scales, length: scales.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared),
          let bBuf = context.device.makeBuffer(
            bytes: biases, length: biases.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared),
          let xBuf = Fp16Buffer.make(context.device, halves: xFp16),
          let yBuf = Fp16Buffer.make(context.device, count: m),
          let cmd = context.queue.makeCommandBuffer() else {
        fatalError("buffer allocation failed")
    }

    kernel.encode(commandBuffer: cmd, weights: wBuf, scales: sBuf, biases: bBuf,
                  x: xBuf, y: yBuf, m: UInt32(m), n: UInt32(n))
    waitAndCheck(cmd, "int8-gemv m=\(m) n=\(n) g=\(groupSize)")

    let reference = DequantInt8GemvRef.apply(weightRows: rows,
                                             x: xFp16.map { Float($0) },
                                             n: n, groupSize: groupSize)
    return result("int8-gemv m=\(m) n=\(n)", groupSize: groupSize,
           rel: relativeError(actual: Fp16Buffer.read(yBuf, count: m),
                              reference: reference),
           tolerance: Double(Tolerance.fp16Reduction),
           detail: "groups=\(n / groupSize) steps=\(n / 64)")
}

/// Fused INT8 gate/up/GELU plus the INT8 down GEMV, at the production shared
/// -expert shape.
func checkSharedExpertInt8(d: Int, f: Int, groupSize: Int,
                           seed: UInt64) throws -> CaseResult {
    var rng = SeedTree(seed).key("shared-int8-d\(d)-f\(f)-g\(groupSize)")
    let gateRows = quantizedInt8Rows(count: f, n: d, groupSize: groupSize, rng: &rng)
    let upRows = quantizedInt8Rows(count: f, n: d, groupSize: groupSize, rng: &rng)
    let downRows = quantizedInt8Rows(count: d, n: f, groupSize: groupSize, rng: &rng)
    let xFp16 = (0..<d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

    let context = try makeContext(groupSize: groupSize)
    let runtime = try SharedExpertRuntime(context: context, weightBits: 8)

    func projection(_ rows: [Quantization.Int8AffineRow],
                    rowCount: Int, columns: Int) -> SharedExpertProjection {
        let (packed, scales, biases) = packInt8Rows(rows)
        guard let w = context.device.makeBuffer(bytes: packed, length: packed.count,
                                                options: .storageModeShared),
              let s = context.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared),
              let b = context.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared) else {
            fatalError("buffer allocation failed")
        }
        return SharedExpertProjection(weights: w, scales: s, biases: b,
                                      rows: UInt32(rowCount), cols: UInt32(columns))
    }

    guard let xBuf = Fp16Buffer.make(context.device, halves: xFp16),
          let yBuf = Fp16Buffer.make(context.device, count: d),
          let actBuf = Fp16Buffer.make(context.device, count: f),
          let cmd = context.queue.makeCommandBuffer() else {
        fatalError("buffer allocation failed")
    }

    try runtime.encode(commandBuffer: cmd, x: xBuf,
                       gate: projection(gateRows, rowCount: f, columns: d),
                       up: projection(upRows, rowCount: f, columns: d),
                       down: projection(downRows, rowCount: d, columns: f),
                       y: yBuf,
                       scratchGate: actBuf, scratchUp: actBuf, scratchAct: actBuf)
    waitAndCheck(cmd, "shared-expert-int8 d=\(d) f=\(f) g=\(groupSize)")

    let reference = MoeRef.runFFNInt8(gateRows: gateRows, upRows: upRows,
                                      downRows: downRows,
                                      x: xFp16.map { Float($0) },
                                      d: d, f: f, groupSize: groupSize)
    return result("shared-expert-int8 d=\(d) f=\(f)", groupSize: groupSize,
           rel: relativeError(actual: Fp16Buffer.read(yBuf, count: d),
                              reference: reference),
           tolerance: Double(Tolerance.fp16ChainedReduction),
           detail: "gate/up groups=\(d / groupSize) down groups=\(f / groupSize)")
}

// MARK: - Case 10 — prefill INT4 QMM (simdgroup_matrix tiles)

/// `Y[t, n] = X[t, k] * W[n, k]^T` through the prefill QMM, which serves both
/// the Q/K/V/O projections and the batched shared MLP.
///
/// The tiled kernel is the one place where a K tile (32) can be narrower than
/// the affine group (64), so the scale/bias index has to come from the global
/// K position rather than the tile counter — the failure mode that killed the
/// MPP path. `k = 2816` covers it: 88 tiles over 44 groups at group 64, and
/// 88 tiles over 88 groups at group 32.
///
/// Shapes that are not whole tiles (`t = 131`, `n = 100`) check the masked
/// edges: out-of-range rows must contribute zero rather than garbage, and
/// out-of-range outputs must not be written at all.
func checkPrefillInt4QMM(t: Int, n: Int, k: Int, groupSize: Int,
                         seed: UInt64) throws -> CaseResult {
    var rng = SeedTree(seed).key("prefill-qmm-t\(t)-n\(n)-k\(k)-g\(groupSize)")
    let rows = quantizedRows(count: n, n: k, groupSize: groupSize, rng: &rng)
    let (packed, scales, biases) = packRows(rows)

    let xFp16 = (0..<(t * k)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

    let context = try makeContext(groupSize: groupSize)
    let kernel = try PrefillInt4QMM(context: context)

    // A sentinel in the output catches a tile that skips its masked store.
    let sentinel = [Float16](repeating: Float16(-7.0), count: t * n)
    guard let wBuf = context.device.makeBuffer(bytes: packed, length: packed.count,
                                               options: .storageModeShared),
          let sBuf = context.device.makeBuffer(
            bytes: scales, length: scales.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared),
          let bBuf = context.device.makeBuffer(
            bytes: biases, length: biases.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared),
          let xBuf = Fp16Buffer.make(context.device, halves: xFp16),
          let yBuf = Fp16Buffer.make(context.device, halves: sentinel),
          let cmd = context.queue.makeCommandBuffer() else {
        fatalError("buffer allocation failed")
    }

    let path = kernel.encode(commandBuffer: cmd,
                             weights: wBuf, scales: sBuf, biases: bBuf,
                             x: xBuf, y: yBuf,
                             t: t, n: n, k: k)
    waitAndCheck(cmd, "prefill-qmm t=\(t) n=\(n) k=\(k) g=\(groupSize)")
    guard path == .simdgroupMatrix else {
        fatalError("prefill-qmm t=\(t) n=\(n) k=\(k) g=\(groupSize): "
                   + "ran the \(path.rawValue) path, so the tiled kernel is untested")
    }

    let xRef = xFp16.map { Float($0) }
    var reference = [Float](repeating: 0, count: t * n)
    for row in 0..<t {
        let y = DequantInt4GemvRef.apply(weightRows: rows,
                                         x: Array(xRef[(row * k)..<((row + 1) * k)]),
                                         n: k, groupSize: groupSize)
        reference.replaceSubrange((row * n)..<((row + 1) * n), with: y)
    }
    return result("prefill-qmm t=\(t) n=\(n) k=\(k)", groupSize: groupSize,
           rel: relativeError(actual: Fp16Buffer.read(yBuf, count: t * n),
                              reference: reference),
           tolerance: Double(Tolerance.fp16Reduction),
           detail: "path=\(path.rawValue) "
                   + "tiles=\((t + 63) / 64)x\((n + 63) / 64)x\(k / 32) "
                   + "kTilesPerGroup=\(groupSize / 32)")
}

// MARK: - Vision tower

// The tower's weights are BF16, so none of these kernels read a scale, a bias
// or an affine group: they behave identically at group 32 and group 64 and are
// therefore checked once rather than per group size.
//
// Gemma 4's tower: hidden 1152, 27 layers, 16 heads of 72, MLP 4304, patch 16
// (768 pixels per patch), 3x3 pooling, RoPE theta 100. The cases below use the
// production head dimension and hidden size wherever the reference stays cheap
// enough to compute on the CPU, and shrink only the counts.

let visionHidden = 1152
let visionHeadDim = 72
let visionRopeTheta: Float = 100
let visionEps: Float = 1e-6

func bf16Buffer(_ device: MTLDevice, _ tensor: BF16Tensor) -> MTLBuffer {
    tensor.bits.withUnsafeBufferPointer { pointer in
        guard let buffer = device.makeBuffer(
            bytes: pointer.baseAddress!,
            length: pointer.count * MemoryLayout<UInt16>.size,
            options: .storageModeShared) else {
            fatalError("buffer allocation failed")
        }
        return buffer
    }
}

func randomBF16(count: Int, range: ClosedRange<Float>, rng: inout SplitMix64) -> BF16Tensor {
    BF16Tensor((0..<count).map { _ in rng.uniform(range.lowerBound, range.upperBound) })
}

/// FP16 inputs rounded once on the host, so the reference sees exactly the
/// values the kernel reads.
func randomFP16(count: Int, range: ClosedRange<Float>,
                rng: inout SplitMix64) -> (halves: [Float16], values: [Float]) {
    let halves = (0..<count).map { _ in Float16(rng.uniform(range.lowerBound, range.upperBound)) }
    return (halves, halves.map { Float($0) })
}

func makeFP16Buffer(_ device: MTLDevice, _ halves: [Float16]) -> MTLBuffer {
    guard let buffer = Fp16Buffer.make(device, halves: halves) else {
        fatalError("buffer allocation failed")
    }
    return buffer
}

func makeFP16Buffer(_ device: MTLDevice, count: Int) -> MTLBuffer {
    // A sentinel rather than zeros: a tile that skips its store shows up as the
    // sentinel instead of a plausible value.
    makeFP16Buffer(device, [Float16](repeating: Float16(-7.0), count: count))
}

/// The BF16 QMM at four shapes.
///
/// `k = 4304` (the MLP down projection) is the shape the INT4 kernel could
/// never see: its K was always a multiple of the affine group and therefore of
/// the 32-wide K tile, while 4304 leaves a 16-element tail. `n = 4304` leaves a
/// tile tail on the output side, and the small case puts a tail on all three
/// axes at once with K narrower than a single tile.
func checkVisionQMM(context: MetalContext, t: Int, n: Int, k: Int,
                    seed: UInt64) throws -> CaseResult {
    var rng = SeedTree(seed).key("vision-qmm-t\(t)-n\(n)-k\(k)")
    let weights = randomBF16(count: n * k, range: -0.5...0.5, rng: &rng)
    let x = randomFP16(count: t * k, range: -1...1, rng: &rng)

    let kernel = try VisionBF16QMM(context: context)
    let wBuf = bf16Buffer(context.device, weights)
    let xBuf = makeFP16Buffer(context.device, x.halves)
    let yBuf = makeFP16Buffer(context.device, count: t * n)
    guard let cmd = context.queue.makeCommandBuffer() else {
        fatalError("command buffer allocation failed")
    }
    kernel.encode(commandBuffer: cmd, weights: wBuf, x: xBuf, y: yBuf, t: t, n: n, k: k)
    waitAndCheck(cmd, "vision-qmm t=\(t) n=\(n) k=\(k)")

    let reference = VisionTowerRef.matmul(weights: weights.values, x: x.values,
                                          t: t, n: n, k: k)
    return result("vision-qmm t=\(t) n=\(n) k=\(k)", groupSize: context.affineGroupSize,
                  rel: relativeError(actual: Fp16Buffer.read(yBuf, count: t * n),
                                     reference: reference),
                  tolerance: Double(Tolerance.fp16Reduction),
                  detail: "tiles=\((t + 63) / 64)x\((n + 63) / 64)x\((k + 31) / 32) "
                          + "kTail=\(k % 32) nTail=\(n % 64)")
}

/// Patch embedder: `2(x - 0.5)`, the 768 -> 1152 projection, and the sum of the
/// two position-table rows. Positions are derived from the patch index inside
/// the kernel, so a transposed grid would move every patch off the diagonal —
/// the second case scores the same output against that reference.
func checkVisionPatchEmbed(context: MetalContext, patchesWide: Int, patchesHigh: Int,
                           seed: UInt64) throws -> [CaseResult] {
    var rng = SeedTree(seed).key("vision-patch-embed-\(patchesWide)x\(patchesHigh)")
    let patchCount = patchesWide * patchesHigh
    let patchDim = 768
    let tableLength = 64
    precondition(patchesWide <= tableLength && patchesHigh <= tableLength)

    let projection = randomBF16(count: visionHidden * patchDim, range: -0.05...0.05, rng: &rng)
    let table = randomBF16(count: 2 * tableLength * visionHidden, range: -1...1, rng: &rng)
    let pixels = randomFP16(count: patchCount * patchDim, range: 0...1, rng: &rng)

    let qmm = try VisionBF16QMM(context: context)
    let embed = try VisionPatchEmbed(context: context)
    let pixelBuf = makeFP16Buffer(context.device, pixels.halves)
    let scaledBuf = makeFP16Buffer(context.device, count: patchCount * patchDim)
    let hBuf = makeFP16Buffer(context.device, count: patchCount * visionHidden)
    let projBuf = bf16Buffer(context.device, projection)
    let tableBuf = bf16Buffer(context.device, table)
    guard let cmd = context.queue.makeCommandBuffer() else {
        fatalError("command buffer allocation failed")
    }
    embed.encodePrescale(commandBuffer: cmd, x: pixelBuf, out: scaledBuf,
                         count: patchCount * patchDim)
    qmm.encode(commandBuffer: cmd, weights: projBuf, x: scaledBuf, y: hBuf,
               t: patchCount, n: visionHidden, k: patchDim)
    embed.encodePositionAdd(commandBuffer: cmd, h: hBuf, table: tableBuf,
                            patchCount: patchCount, d: visionHidden,
                            patchesWide: patchesWide, tableLength: tableLength)
    waitAndCheck(cmd, "vision-patch-embed \(patchesWide)x\(patchesHigh)")

    let actual = Fp16Buffer.read(hBuf, count: patchCount * visionHidden)
    let reference = VisionTowerRef.patchEmbed(
        pixels: pixels.values, projection: projection.values,
        positionTable: table.values, patchCount: patchCount, patchDim: patchDim,
        hidden: visionHidden, patchesWide: patchesWide, tableLength: tableLength)
    let transposed = VisionTowerRef.patchEmbed(
        pixels: pixels.values, projection: projection.values,
        positionTable: table.values, patchCount: patchCount, patchDim: patchDim,
        hidden: visionHidden, patchesWide: patchesHigh, tableLength: tableLength)

    return [
        result("vision-patch-embed \(patchesWide)x\(patchesHigh)",
               groupSize: context.affineGroupSize,
               rel: relativeError(actual: actual, reference: reference),
               tolerance: Double(Tolerance.fp16Reduction),
               detail: "P=\(patchCount) tableLen=\(tableLength)"),
        detectionResult("vision-patch-embed/grid-transposed",
                        groupSize: context.affineGroupSize,
                        rel: relativeError(actual: actual, reference: transposed),
                        floor: 0.05),
    ]
}

/// Q/K per-head RMSNorm, scale-less V RMSNorm, and the 2D RoPE.
///
/// The x and y halves of a head use identical frequencies, so exchanging the
/// two positions is invisible on the grid diagonal and nowhere else. The
/// negative control scores the same output against the swapped reference; a
/// non-square patch grid makes sure the mistake cannot cancel.
func checkVisionQKNormRoPE(context: MetalContext, patchesWide: Int, patchesHigh: Int,
                           numHeads: Int, seed: UInt64) throws -> [CaseResult] {
    var rng = SeedTree(seed).key("vision-qk-rope-\(patchesWide)x\(patchesHigh)-h\(numHeads)")
    let patchCount = patchesWide * patchesHigh
    let elements = patchCount * numHeads * visionHeadDim

    let q = randomFP16(count: elements, range: -1...1, rng: &rng)
    let k = randomFP16(count: elements, range: -1...1, rng: &rng)
    let v = randomFP16(count: elements, range: -1...1, rng: &rng)
    let qWeight = randomBF16(count: visionHeadDim, range: 0.5...1.5, rng: &rng)
    let kWeight = randomBF16(count: visionHeadDim, range: 0.5...1.5, rng: &rng)

    let kernel = try VisionQKNormRoPE2D(context: context)
    let qBuf = makeFP16Buffer(context.device, q.halves)
    let kBuf = makeFP16Buffer(context.device, k.halves)
    let vBuf = makeFP16Buffer(context.device, v.halves)
    guard let cmd = context.queue.makeCommandBuffer() else {
        fatalError("command buffer allocation failed")
    }
    kernel.encode(commandBuffer: cmd, q: qBuf, k: kBuf, v: vBuf,
                  qWeight: bf16Buffer(context.device, qWeight),
                  kWeight: bf16Buffer(context.device, kWeight),
                  patchCount: patchCount, headDim: visionHeadDim, numHeads: numHeads,
                  patchesWide: patchesWide, theta: visionRopeTheta, eps: visionEps)
    waitAndCheck(cmd, "vision-qk-rope \(patchesWide)x\(patchesHigh)")

    let actual = Fp16Buffer.read(qBuf, count: elements)
        + Fp16Buffer.read(kBuf, count: elements)
        + Fp16Buffer.read(vBuf, count: elements)

    func reference(_ variant: VisionTowerRef.Variant) -> [Float] {
        let out = VisionTowerRef.qkNormRoPE2D(
            q: q.values, k: k.values, v: v.values,
            qWeight: qWeight.values, kWeight: kWeight.values,
            patchCount: patchCount, headDim: visionHeadDim, numHeads: numHeads,
            patchesWide: patchesWide, theta: visionRopeTheta, eps: visionEps,
            variant: variant)
        return out.q + out.k + out.v
    }

    return [
        result("vision-qk-rope2d \(patchesWide)x\(patchesHigh) heads=\(numHeads)",
               groupSize: context.affineGroupSize,
               rel: relativeError(actual: actual, reference: reference(.upstream)),
               tolerance: Double(Tolerance.fp16Reduction),
               detail: "headDim=\(visionHeadDim) theta=\(visionRopeTheta)"),
        detectionResult("vision-qk-rope2d/axes-swapped",
                        groupSize: context.affineGroupSize,
                        rel: relativeError(actual: actual, reference: reference(.ropeAxesSwapped)),
                        floor: 0.05),
    ]
}

/// Non-causal attention over every patch, at the tower's head dimension of 72.
///
/// 72 is not a multiple of the 32-lane simdgroup, so the kernel masks its
/// per-lane slice instead of dividing it exactly; a patch count that is not a
/// whole number of query blocks (8 per simdgroup, 8 simdgroups per group)
/// exercises the block tail at the same time.
/// Both specialisations are checked at every shape: the fallback is what runs
/// if the segmented kernel is ever disabled, and a fallback that has drifted
/// out of agreement would only show up as degraded output.
func checkVisionAttention(context: MetalContext, patchCount: Int, numHeads: Int,
                          seed: UInt64) throws -> [CaseResult] {
    var rng = SeedTree(seed).key("vision-attention-p\(patchCount)-h\(numHeads)")
    let elements = patchCount * numHeads * visionHeadDim
    let q = randomFP16(count: elements, range: -1...1, rng: &rng)
    let k = randomFP16(count: elements, range: -1...1, rng: &rng)
    let v = randomFP16(count: elements, range: -1...1, rng: &rng)

    let kernel = try VisionAttentionFull(context: context)
    let qBuf = makeFP16Buffer(context.device, q.halves)
    let kBuf = makeFP16Buffer(context.device, k.halves)
    let vBuf = makeFP16Buffer(context.device, v.halves)
    let reference = VisionTowerRef.attentionFull(
        q: q.values, k: k.values, v: v.values,
        patchCount: patchCount, headDim: visionHeadDim, numHeads: numHeads, scale: 1.0)

    var cases: [CaseResult] = []
    for wanted in [VisionAttentionFull.Path.segment8, .qBlock] {
        let outBuf = makeFP16Buffer(context.device, count: elements)
        guard let cmd = context.queue.makeCommandBuffer() else {
            fatalError("command buffer allocation failed")
        }
        let ran = kernel.encode(commandBuffer: cmd, q: qBuf, k: kBuf, v: vBuf,
                                out: outBuf,
                                patchCount: patchCount, headDim: visionHeadDim,
                                numHeads: numHeads, forcePath: wanted)
        waitAndCheck(cmd, "vision-attention P=\(patchCount) \(wanted.rawValue)")
        guard ran == wanted else {
            fatalError("vision-attention: asked for \(wanted.rawValue), ran \(ran.rawValue)")
        }
        cases.append(
            result("vision-attention/\(wanted.rawValue) P=\(patchCount) heads=\(numHeads)",
                   groupSize: context.affineGroupSize,
                   rel: relativeError(actual: Fp16Buffer.read(outBuf, count: elements),
                                      reference: reference),
                   tolerance: Double(Tolerance.fp16ChainedReduction),
                   detail: "scale=1.0 groups="
                           + "\((patchCount + VisionAttentionFull.queriesPerGroup - 1) / VisionAttentionFull.queriesPerGroup) "
                           + "tail=\(patchCount % VisionAttentionFull.queriesPerGroup)"))
    }
    return cases
}

func checkVisionMLPActivation(context: MetalContext, count: Int,
                              seed: UInt64) throws -> CaseResult {
    var rng = SeedTree(seed).key("vision-mlp-act-\(count)")
    let gate = randomFP16(count: count, range: -4...4, rng: &rng)
    let up = randomFP16(count: count, range: -2...2, rng: &rng)

    let kernel = try VisionMLPActivation(context: context)
    let outBuf = makeFP16Buffer(context.device, count: count)
    guard let cmd = context.queue.makeCommandBuffer() else {
        fatalError("command buffer allocation failed")
    }
    kernel.encode(commandBuffer: cmd,
                  gate: makeFP16Buffer(context.device, gate.halves),
                  up: makeFP16Buffer(context.device, up.halves),
                  out: outBuf, count: count)
    waitAndCheck(cmd, "vision-mlp-act \(count)")

    return result("vision-mlp-act n=\(count)", groupSize: context.affineGroupSize,
                  rel: relativeError(
                    actual: Fp16Buffer.read(outBuf, count: count),
                    reference: VisionTowerRef.geluMultiply(gate: gate.values, up: up.values)),
                  tolerance: Double(Tolerance.fp16Reduction),
                  detail: "gelu_pytorch_tanh")
}

/// `hidden += rmsnorm(x) * weight`, the fused residual join used twice a layer.
///
/// The negative control is a reference that overwrites the residual instead of
/// adding to it: the join is the one place in the tower where reading the wrong
/// buffer still produces well-scaled output, so it needs a control that says
/// the residual is really there.
func checkVisionNormResidualAdd(context: MetalContext, t: Int, d: Int,
                                seed: UInt64) throws -> [CaseResult] {
    var rng = SeedTree(seed).key("vision-norm-residual-\(t)x\(d)")
    let hidden = randomFP16(count: t * d, range: -3...3, rng: &rng)
    let x = randomFP16(count: t * d, range: -2...2, rng: &rng)
    let weight = randomBF16(count: d, range: 0.5...1.5, rng: &rng)

    let kernel = try VisionNormResidualAdd(context: context)
    let hiddenBuf = makeFP16Buffer(context.device, hidden.halves)
    guard let cmd = context.queue.makeCommandBuffer() else {
        fatalError("command buffer allocation failed")
    }
    kernel.encode(commandBuffer: cmd,
                  hidden: hiddenBuf,
                  x: makeFP16Buffer(context.device, x.halves),
                  weight: bf16Buffer(context.device, weight),
                  t: t, d: d, eps: visionEps)
    waitAndCheck(cmd, "vision-norm-residual t=\(t) d=\(d)")

    let actual = Fp16Buffer.read(hiddenBuf, count: t * d)
    func reference(_ variant: VisionTowerRef.Variant) -> [Float] {
        VisionTowerRef.normResidualAdd(hidden: hidden.values, x: x.values,
                                       weight: weight.values,
                                       t: t, d: d, eps: visionEps, variant: variant)
    }
    return [
        result("vision-norm-residual-add t=\(t) d=\(d)",
               groupSize: context.affineGroupSize,
               rel: relativeError(actual: actual, reference: reference(.upstream)),
               tolerance: Double(Tolerance.fp16Reduction),
               detail: "in-place residual"),
        detectionResult("vision-norm-residual-add/residual-dropped",
                        groupSize: context.affineGroupSize,
                        rel: relativeError(actual: actual,
                                           reference: reference(.residualDropped)),
                        floor: 0.05),
    ]
}

/// 3x3 average pool, `* sqrt(1152)`, then `(x - std_bias) * std_scale`.
///
/// The pooled grid is emitted row-major; the negative control scores the same
/// output against a column-major reference on a non-square grid, where the two
/// orders disagree everywhere off the diagonal.
func checkVisionPoolStandardize(context: MetalContext, patchesWide: Int, patchesHigh: Int,
                                seed: UInt64) throws -> [CaseResult] {
    var rng = SeedTree(seed).key("vision-pool-\(patchesWide)x\(patchesHigh)")
    let d = 128
    let patchCount = patchesWide * patchesHigh
    let kernelSize = 3
    let cells = (patchesWide / kernelSize) * (patchesHigh / kernelSize)

    let hidden = randomFP16(count: patchCount * d, range: -1...1, rng: &rng)
    let stdScale = randomBF16(count: d, range: 0.5...1.5, rng: &rng)
    let stdBias = randomBF16(count: d, range: -0.5...0.5, rng: &rng)

    let kernel = try VisionPoolStandardize(context: context)
    let outBuf = makeFP16Buffer(context.device, count: cells * d)
    guard let cmd = context.queue.makeCommandBuffer() else {
        fatalError("command buffer allocation failed")
    }
    kernel.encode(commandBuffer: cmd,
                  h: makeFP16Buffer(context.device, hidden.halves),
                  out: outBuf,
                  stdScale: bf16Buffer(context.device, stdScale),
                  stdBias: bf16Buffer(context.device, stdBias),
                  d: d, patchesWide: patchesWide, patchesHigh: patchesHigh,
                  kernelSize: kernelSize, rootHidden: Float(d).squareRoot(),
                  standardize: true)
    waitAndCheck(cmd, "vision-pool \(patchesWide)x\(patchesHigh)")

    let actual = Fp16Buffer.read(outBuf, count: cells * d)
    func reference(_ variant: VisionTowerRef.Variant) -> [Float] {
        VisionTowerRef.poolStandardize(
            hidden: hidden.values, d: d,
            patchesWide: patchesWide, patchesHigh: patchesHigh, kernelSize: kernelSize,
            stdScale: stdScale.values, stdBias: stdBias.values, standardize: true,
            variant: variant)
    }

    return [
        result("vision-pool-std \(patchesWide)x\(patchesHigh)",
               groupSize: context.affineGroupSize,
               rel: relativeError(actual: actual, reference: reference(.upstream)),
               tolerance: Double(Tolerance.fp16Reduction),
               detail: "cells=\(cells) d=\(d)"),
        detectionResult("vision-pool-std/column-major",
                        groupSize: context.affineGroupSize,
                        rel: relativeError(actual: actual, reference: reference(.poolColumnMajor)),
                        floor: 0.05),
    ]
}

// MARK: - Vision throughput

/// `PLAN_VISION.md` §8 makes the BF16 QMM's measured throughput the gate on the
/// tower design: below 1.0 TFLOP/s the 3.54 TFLOP of a 280-soft-token image
/// costs more than 3.5 s and the default soft-token count has to come down.
/// This measures it at the tower's real shapes rather than deriving it.
struct BenchRow {
    let name: String
    let flops: Double
    let seconds: Double
    let count: Int

    var gflopsPerSecond: Double { flops / seconds / 1e9 }
}

/// GPU time for `iterations` back-to-back dispatches in one command buffer,
/// after a warm-up buffer. `gpuEndTime - gpuStartTime` excludes the host-side
/// encode and the queue wait, which at these shapes would otherwise dominate.
func gpuSeconds(context: MetalContext, iterations: Int,
                label: String,
                encode: (MTLCommandBuffer) -> Void) -> Double {
    guard let warmup = context.queue.makeCommandBuffer() else {
        fatalError("command buffer allocation failed")
    }
    encode(warmup)
    waitAndCheck(warmup, "\(label) warmup")

    guard let cmd = context.queue.makeCommandBuffer() else {
        fatalError("command buffer allocation failed")
    }
    for _ in 0..<iterations { encode(cmd) }
    waitAndCheck(cmd, label)
    return (cmd.gpuEndTime - cmd.gpuStartTime) / Double(iterations)
}

func benchVisionQMM(context: MetalContext, name: String, t: Int, n: Int, k: Int,
                    count: Int, iterations: Int) throws -> BenchRow {
    var rng = SeedTree(0x5115).key("bench-qmm-\(t)-\(n)-\(k)")
    let weights = randomBF16(count: n * k, range: -0.5...0.5, rng: &rng)
    let x = randomFP16(count: t * k, range: -1...1, rng: &rng)
    let kernel = try VisionBF16QMM(context: context)
    let wBuf = bf16Buffer(context.device, weights)
    let xBuf = makeFP16Buffer(context.device, x.halves)
    let yBuf = makeFP16Buffer(context.device, count: t * n)

    let seconds = gpuSeconds(context: context, iterations: iterations,
                             label: "bench \(name)") { cmd in
        kernel.encode(commandBuffer: cmd, weights: wBuf, x: xBuf, y: yBuf,
                      t: t, n: n, k: k)
    }
    return BenchRow(name: name, flops: 2.0 * Double(t) * Double(n) * Double(k),
                    seconds: seconds, count: count)
}

func benchVisionAttention(context: MetalContext, patchCount: Int, numHeads: Int,
                          path: VisionAttentionFull.Path,
                          iterations: Int) throws -> BenchRow {
    var rng = SeedTree(0x5116).key("bench-attn-\(patchCount)")
    let elements = patchCount * numHeads * visionHeadDim
    let q = randomFP16(count: elements, range: -1...1, rng: &rng)
    let kernel = try VisionAttentionFull(context: context)
    let qBuf = makeFP16Buffer(context.device, q.halves)
    let outBuf = makeFP16Buffer(context.device, count: elements)

    let seconds = gpuSeconds(context: context, iterations: iterations,
                             label: "bench attention \(path.rawValue)") { cmd in
        kernel.encode(commandBuffer: cmd, q: qBuf, k: qBuf, v: qBuf, out: outBuf,
                      patchCount: patchCount, headDim: visionHeadDim,
                      numHeads: numHeads, forcePath: path)
    }
    // QK^T and the value accumulation, both P x P x headDim per head.
    let flops = 4.0 * Double(patchCount) * Double(patchCount)
        * Double(visionHeadDim) * Double(numHeads)
    return BenchRow(name: "attention/\(path.rawValue)", flops: flops,
                    seconds: seconds, count: 1)
}

/// The two memory-bound passes between the projections. They carry no useful
/// FLOP count, so they are reported in milliseconds and folded into the tower
/// total rather than into any throughput figure.
func benchVisionEpilogues(context: MetalContext, patchCount: Int, numHeads: Int,
                          intermediate: Int,
                          iterations: Int) throws -> [BenchRow] {
    var rng = SeedTree(0x5117).key("bench-epilogue-\(patchCount)")
    let elements = patchCount * numHeads * visionHeadDim
    let data = randomFP16(count: elements, range: -1...1, rng: &rng)
    let weight = randomBF16(count: visionHeadDim, range: 0.5...1.5, rng: &rng)
    let epilogue = try VisionQKNormRoPE2D(context: context)
    let qBuf = makeFP16Buffer(context.device, data.halves)
    let kBuf = makeFP16Buffer(context.device, data.halves)
    let vBuf = makeFP16Buffer(context.device, data.halves)
    let weightBuf = bf16Buffer(context.device, weight)
    let epilogueSeconds = gpuSeconds(context: context, iterations: iterations,
                                     label: "bench qk-norm-rope") { cmd in
        epilogue.encode(commandBuffer: cmd, q: qBuf, k: kBuf, v: vBuf,
                        qWeight: weightBuf, kWeight: weightBuf,
                        patchCount: patchCount, headDim: visionHeadDim,
                        numHeads: numHeads, patchesWide: patchCount / 9,
                        theta: visionRopeTheta, eps: visionEps)
    }

    let actCount = patchCount * intermediate
    let gate = randomFP16(count: actCount, range: -4...4, rng: &rng)
    let activation = try VisionMLPActivation(context: context)
    let gateBuf = makeFP16Buffer(context.device, gate.halves)
    let outBuf = makeFP16Buffer(context.device, count: actCount)
    let actSeconds = gpuSeconds(context: context, iterations: iterations,
                                label: "bench mlp-act") { cmd in
        activation.encode(commandBuffer: cmd, gate: gateBuf, up: gateBuf, out: outBuf,
                          count: actCount)
    }

    return [BenchRow(name: "qk-norm+rope2d", flops: 0, seconds: epilogueSeconds, count: 1),
            BenchRow(name: "mlp act", flops: 0, seconds: actSeconds, count: 1)]
}

func runVisionBench(context: MetalContext, softTokens: Int) throws {
    let patches = softTokens * 9
    let heads = visionHidden / visionHeadDim
    let intermediate = 4304
    let layers = 27

    print("=== vision tower throughput (S=\(softTokens), P=\(patches)) ===")
    var perLayer: [BenchRow] = []
    perLayer.append(try benchVisionQMM(context: context, name: "q/k/v/o proj",
                                       t: patches, n: visionHidden, k: visionHidden,
                                       count: 4, iterations: 20))
    perLayer.append(try benchVisionQMM(context: context, name: "mlp gate/up",
                                       t: patches, n: intermediate, k: visionHidden,
                                       count: 2, iterations: 10))
    perLayer.append(try benchVisionQMM(context: context, name: "mlp down",
                                       t: patches, n: visionHidden, k: intermediate,
                                       count: 1, iterations: 10))
    perLayer.append(try benchVisionAttention(context: context, patchCount: patches,
                                             numHeads: heads, path: .segment8,
                                             iterations: 5))
    perLayer.append(contentsOf: try benchVisionEpilogues(context: context,
                                                         patchCount: patches,
                                                         numHeads: heads,
                                                         intermediate: intermediate,
                                                         iterations: 20))
    // The fallback, measured for the record but not counted into the tower.
    let attentionFallback = try benchVisionAttention(context: context, patchCount: patches,
                                                     numHeads: heads, path: .qBlock,
                                                     iterations: 3)

    let once: [BenchRow] = [
        try benchVisionQMM(context: context, name: "patch embed",
                           t: patches, n: visionHidden, k: 768,
                           count: 1, iterations: 20),
        try benchVisionQMM(context: context, name: "projector",
                           t: softTokens, n: 2816, k: visionHidden,
                           count: 1, iterations: 20),
    ]

    func show(_ row: BenchRow, multiplier: Int) {
        let ms = row.seconds * 1e3
        let rate = row.flops > 0
            ? String(format: "%8.1f GFLOP/s  (%.1f GFLOP)", row.gflopsPerSecond,
                     row.flops / 1e9)
            : "       — GFLOP/s  (memory bound)"
        print(String(format: "  %-18s x%-3d  %8.3f ms  %@",
                     (row.name as NSString).utf8String!, multiplier, ms, rate))
    }

    for row in perLayer { show(row, multiplier: row.count * layers) }
    for row in once { show(row, multiplier: 1) }
    show(attentionFallback, multiplier: 0)

    let projectionRows = perLayer.filter { $0.flops > 0 && !$0.name.hasPrefix("attention") }
    let projectionFlops = projectionRows.reduce(0.0) { $0 + $1.flops * Double($1.count) }
        * Double(layers)
    let projectionSeconds = projectionRows.reduce(0.0) { $0 + $1.seconds * Double($1.count) }
        * Double(layers)
    let towerFlops = perLayer.reduce(0.0) { $0 + $1.flops * Double($1.count) } * Double(layers)
        + once.reduce(0.0) { $0 + $1.flops }
    let towerSeconds = perLayer.reduce(0.0) { $0 + $1.seconds * Double($1.count) } * Double(layers)
        + once.reduce(0.0) { $0 + $1.seconds }

    print("")
    print(String(format: "  BF16 QMM (projections only): %.2f TFLOP / %.2f s = %.2f TFLOP/s",
                 projectionFlops / 1e12, projectionSeconds, projectionFlops / projectionSeconds / 1e12))
    print(String(format: "  whole tower, one image:      %.2f TFLOP / %.2f s = %.2f TFLOP/s",
                 towerFlops / 1e12, towerSeconds, towerFlops / towerSeconds / 1e12))
    let gate = projectionFlops / projectionSeconds / 1e12
    print(String(format: "  PLAN_VISION §8 gate (BF16 QMM >= 1.0 TFLOP/s): %@ (%.2f)",
                 gate >= 1.0 ? "PASS" : "FAIL", gate))
}

// MARK: - Driver

let arguments = CommandLine.arguments
var groupSizes = [64, 32]
if let index = arguments.firstIndex(of: "--group-size"),
   index + 1 < arguments.count,
   let value = Int(arguments[index + 1]) {
    groupSizes = [value]
}
// `--vision-only` skips the INT4 suite; `--bench` adds the tower's throughput
// measurement, which takes tens of seconds and is not a pass/fail check.
let visionOnly = arguments.contains("--vision-only")
let runBench = arguments.contains("--bench")
var benchSoftTokens = 280
if let index = arguments.firstIndex(of: "--bench-soft-tokens"),
   index + 1 < arguments.count,
   let value = Int(arguments[index + 1]) {
    benchSoftTokens = value
}
// `--vision-tower <model.gturbo>` adds the assembled-tower comparison against
// the reference fixtures (PLAN_VISION §6-1 layer B). It needs an installed
// tower and the `scratch/vision-fixtures` dump, so it is opt-in rather than
// part of the default run.
var visionTowerModel: String?
if let index = arguments.firstIndex(of: "--vision-tower"), index + 1 < arguments.count {
    visionTowerModel = arguments[index + 1]
}
var visionFixtureRoot = VisionFixtures.defaultRoot
if let index = arguments.firstIndex(of: "--vision-fixtures"), index + 1 < arguments.count {
    visionFixtureRoot = arguments[index + 1]
}
// `--draft <model.gturbo>` adds the assembled-drafter comparison against the
// M2 reference fixtures. Same opt-in shape as `--vision-tower`.
var draftModel: String?
if let index = arguments.firstIndex(of: "--draft"), index + 1 < arguments.count {
    draftModel = arguments[index + 1]
}
var draftFixtureRoot = "scratch/mtp-fixtures"
if let index = arguments.firstIndex(of: "--draft-fixtures"), index + 1 < arguments.count {
    draftFixtureRoot = arguments[index + 1]
}
// `--verify-block <model.gturbo>` runs the M4 checks: the speculative verify
// pass against scalar decode, and the KV rewind against the world that never
// speculated. It loads the 26B target, so it implies `--model-only` unless the
// synthetic suites were asked for as well.
var verifyBlockModel: String?
if let index = arguments.firstIndex(of: "--verify-block"), index + 1 < arguments.count {
    verifyBlockModel = arguments[index + 1]
}
var verifyBlockTokens = 4
if let index = arguments.firstIndex(of: "--verify-block-size"), index + 1 < arguments.count,
   let value = Int(arguments[index + 1]) {
    verifyBlockTokens = value
}
var verifyRounds = 12
if let index = arguments.firstIndex(of: "--verify-rounds"), index + 1 < arguments.count,
   let value = Int(arguments[index + 1]) {
    verifyRounds = value
}
var verifyPrompt = VerifyBlockPrompt.default
if let index = arguments.firstIndex(of: "--verify-prompt"), index + 1 < arguments.count {
    verifyPrompt = arguments[index + 1]
}
// The goal condition (docs/mtp/10-M0-RESULTS.md §4) is an image plus reasoning,
// not a bare completion, and its cache is not the one a replayed stream leaves
// behind. These three put the cost probe there: `--verify-image` attaches an
// image to the prompt through the chat template, `--verify-thinking` opens the
// thought channel, and `--verify-cold` empties the expert cache before every
// timed phase so decode and the block each pay their own misses.
var verifyImages: [String] = []
for (index, argument) in arguments.enumerated()
where argument == "--verify-image" && index + 1 < arguments.count {
    verifyImages.append(arguments[index + 1])
}
var verifyImageTokens = 280
if let index = arguments.firstIndex(of: "--verify-image-tokens"), index + 1 < arguments.count,
   let value = Int(arguments[index + 1]) {
    verifyImageTokens = value
}
let verifyThinking = arguments.contains("--verify-thinking")
let verifyCold = arguments.contains("--verify-cold")
// Skip the synthetic kernel suites and run only the checks that need an
// installed model. `--verify-block` implies it: those suites take minutes and
// have nothing to say about the pass this run is here to check.
let modelOnly = arguments.contains("--model-only") || verifyBlockModel != nil

// `--gdn` runs the Gated DeltaNet checks (GatedDeltaNetCheck.swift): the
// recurrence that carries 30 of Qwen3.5-MoE's 40 layers, against a CPU
// reference at two precisions. No model and no checkpoint — the inputs are
// synthetic in the shapes the model produces. `docs/qwen35moe/04-PHASES.md`
// Phase 2. `--gdn-tokens` sets the prefill length; the exit condition is
// stated at 2048 because one token cannot show an accumulation bug.
if arguments.contains("--gdn") {
    var gdnTokens = 2048
    if let index = arguments.firstIndex(of: "--gdn-tokens"),
       index + 1 < arguments.count, let value = Int(arguments[index + 1]), value > 1 {
        gdnTokens = value
    }
    if arguments.contains("--gdn-bench") {
        var gdnIterations = 20
        if let index = arguments.firstIndex(of: "--gdn-bench-iterations"),
           index + 1 < arguments.count, let value = Int(arguments[index + 1]), value > 0 {
            gdnIterations = value
        }
        try runGatedDeltaNetBench(tokens: gdnTokens, iterations: gdnIterations)
        exit(0)
    }
    let gdnResults = try runGatedDeltaNetCheck(tokens: gdnTokens)
    printCases(gdnResults)
    let gdnFailures = gdnResults.filter { !$0.passed }
    print("")
    if gdnFailures.isEmpty {
        print("PASS  \(gdnResults.count) cases (gated deltanet)")
        exit(0)
    }
    print("FAIL  \(gdnFailures.count)/\(gdnResults.count) cases")
    for failure in gdnFailures { print("  \(failure.name) — \(failure.detail)") }
    exit(1)
}

// `--qwen-open <model.gturbo>` opens a repacked Qwen3.5-MoE install and reports
// the weight width of every resident tensor (QwenOpenCheck.swift). It runs no
// kernels: the check is that `Model.load` accepts a checkpoint whose attention
// is 4-bit in some layers and 8-bit in others.
// `docs/qwen35moe/04-PHASES.md` 次の一手 #11.
if let index = arguments.firstIndex(of: "--qwen-open"), index + 1 < arguments.count {
    exit(try runQwenOpenCheck(modelPath: arguments[index + 1]) ? 0 : 1)
}

// `--qwen-decode <model.gturbo>` runs the decode path over the real weights and
// compares the greedy tokens with the CPU float32 reference
// (QwenDecodeCheck.swift). This is Phase 3's exit condition: every kernel it
// touches was already scored on synthetic inputs, so what is on trial is the
// wiring. `--qwen-decode-fixture <json>` supplies another
// `{"prompt": [...], "expected": [...]}`, `--qwen-decode-new N` shortens the
// comparison, `--qwen-decode-slots N` changes the expert cache.
if let index = arguments.firstIndex(of: "--qwen-decode"), index + 1 < arguments.count {
    var fixturePath: String?
    if let i = arguments.firstIndex(of: "--qwen-decode-fixture"), i + 1 < arguments.count {
        fixturePath = arguments[i + 1]
    }
    var newTokens: Int?
    if let i = arguments.firstIndex(of: "--qwen-decode-new"), i + 1 < arguments.count,
       let value = Int(arguments[i + 1]), value > 0 {
        newTokens = value
    }
    var slots = 32
    if let i = arguments.firstIndex(of: "--qwen-decode-slots"), i + 1 < arguments.count,
       let value = Int(arguments[i + 1]), value > 0 {
        slots = value
    }
    var faultTokens = defaultFaultTokens
    if let i = arguments.firstIndex(of: "--qwen-decode-fault-tokens"), i + 1 < arguments.count,
       let value = Int(arguments[i + 1]), value > 0 {
        faultTokens = value
    }
    exit(try runQwenDecodeCheck(modelPath: arguments[index + 1],
                                fixturePath: fixturePath,
                                maxNewTokens: newTokens,
                                slotCount: slots,
                                runFaults: !arguments.contains("--qwen-decode-no-faults"),
                                faultTokens: faultTokens)
         ? 0 : 1)
}

// `--qwen` runs the rest of the Qwen3.5-MoE kernels (QwenKernelCheck.swift):
// the causal `conv1d` + l2norm that feeds the recurrence, the decay gates, the
// gated RMSNorm, the partial-RoPE epilogue, and the small elementwise pieces.
// Same shape as `--gdn` — synthetic inputs, no model, no checkpoint, a CPU
// reference at two precisions, and one negative control per kernel.
// `docs/qwen35moe/04-PHASES.md` Phase 2. `--qwen-tokens` sets the prefill
// length; 512 is enough here because none of these kernels accumulate across
// the sequence the way `qwen_delta_rule` does — the one that carries state
// (`conv1d`, window 4) is checked for bit-exact chunk carry instead.
if arguments.contains("--qwen") {
    var qwenTokens = 512
    if let index = arguments.firstIndex(of: "--qwen-tokens"),
       index + 1 < arguments.count, let value = Int(arguments[index + 1]), value > 1 {
        qwenTokens = value
    }
    if arguments.contains("--qwen-bench") {
        var qwenIterations = 20
        if let index = arguments.firstIndex(of: "--qwen-bench-iterations"),
           index + 1 < arguments.count, let value = Int(arguments[index + 1]), value > 0 {
            qwenIterations = value
        }
        try runQwenKernelBench(tokens: qwenTokens, iterations: qwenIterations)
        exit(0)
    }
    let qwenResults = try runQwenKernelCheck(tokens: qwenTokens)
    printCases(qwenResults)
    let qwenFailures = qwenResults.filter { !$0.passed }
    print("")
    if qwenFailures.isEmpty {
        print("PASS  \(qwenResults.count) cases (qwen kernels)")
        exit(0)
    }
    print("FAIL  \(qwenFailures.count)/\(qwenResults.count) cases")
    for failure in qwenFailures { print("  \(failure.name) — \(failure.detail)") }
    exit(1)
}

// `--rows-bench` measures the k-row dense GEMV on its own (RowsBench.swift).
// It needs no model and no expert cache, so the k-scaling that
// `docs/mtp/19-M4.7-RESULTS.md` §5 attributes to arithmetic can be read
// directly. It is a measurement, not a check, so it runs and exits.
if arguments.contains("--rows-bench") {
    var rowsBenchIterations = 20
    if let index = arguments.firstIndex(of: "--rows-bench-iterations"),
       index + 1 < arguments.count,
       let value = Int(arguments[index + 1]) {
        rowsBenchIterations = value
    }
    let verifyCases = try runRowsVerify(groupSize: groupSizes[0])
    printCases(verifyCases)
    print("")
    try runRowsBench(groupSize: groupSizes[0], iterations: rowsBenchIterations)
    exit(verifyCases.allSatisfy(\.passed) ? 0 : 1)
}

// `--moe-rows-bench` does the same for the *routed* MoE rows kernels
// (MoERowsBench.swift), which is where `docs/mtp/28-M8-PROPOSAL.md` §2 puts the
// remaining GPU gap. Separate flag because it reports a slope in r rather than
// a throughput, and because gate/up and down are timed apart.
if arguments.contains("--moe-rows-bench") {
    var moeBenchIterations = 50
    if let index = arguments.firstIndex(of: "--moe-rows-bench-iterations"),
       index + 1 < arguments.count,
       let value = Int(arguments[index + 1]) {
        moeBenchIterations = value
    }
    try runMoERowsBench(groupSize: groupSizes[0], iterations: moeBenchIterations)
    exit(0)
}

// `--moe-rows-shape-sweep` walks the *reduced* dimension of the same two
// kernels (F for `down`, D for `gate/up`) with the output grid held fixed.
// `docs/mtp/34-M9-PROPOSAL.md` §2b-1 / §5 asks whether `moe` is at a bandwidth
// floor or bound by loop trips; a sawtooth in this sweep can only come from the
// second, because a larger shape reading more bytes cannot be faster otherwise.
if arguments.contains("--moe-rows-shape-sweep") {
    var sweepIterations = 50
    if let index = arguments.firstIndex(of: "--moe-rows-bench-iterations"),
       index + 1 < arguments.count,
       let value = Int(arguments[index + 1]) {
        sweepIterations = value
    }
    var rowsPerExpert = 2  // production averages 1.72 rows/expert at bs=4 (27 §6)
    if let index = arguments.firstIndex(of: "--moe-rows-r"),
       index + 1 < arguments.count,
       let value = Int(arguments[index + 1]), (1...8).contains(value) {
        rowsPerExpert = value
    }
    try runMoERowsShapeSweep(groupSize: groupSizes[0], iterations: sweepIterations,
                             rowsPerExpert: rowsPerExpert)
    exit(0)
}

// `--bpw-probe`: the same two kernels with the weight format swept instead of
// the shape. 43 §2 read `gate/up` as sitting on the 135 GB/s floor; if that is
// right, dropping the redundant BF16 bias (44 §1) and shrinking the BF16 scale
// to an 8-bit code against a per-row anchor (44 §2) must show up as time, in
// proportion to the bytes, with the arithmetic held fixed.
// `--mmap-residency-probe`: the same two kernels again, but this time nothing
// about the arithmetic moves -- only where the expert bytes come from.
// `docs/mtp/47-D-MMAP-RESIDENCY-PROPOSAL.md` §5 rests the whole D branch on
// Metal wiring a `bytesNoCopy` buffer at the buffer's granularity rather than
// the mapping's; P-1 measures the wired-page delta of a command buffer that
// names eight experts, against the same command buffer fed from one buffer
// spanning the 420 MB layer file.
if arguments.contains("--mmap-residency-probe") {
    var probeModel = "scratch/gemma4-qat-sym.gturbo"
    if let index = arguments.firstIndex(of: "--mmap-probe-model"), index + 1 < arguments.count {
        probeModel = arguments[index + 1]
    }
    var probeRepeats = 1200
    if let index = arguments.firstIndex(of: "--mmap-probe-repeats"),
       index + 1 < arguments.count, let value = Int(arguments[index + 1]) {
        probeRepeats = value
    }
    var probeTrials = 3
    if let index = arguments.firstIndex(of: "--mmap-probe-trials"),
       index + 1 < arguments.count, let value = Int(arguments[index + 1]) {
        probeTrials = value
    }
    try runMmapResidencyProbe(groupSize: groupSizes[0], modelPath: probeModel,
                              repeats: probeRepeats, trials: probeTrials,
                              gateOnly: arguments.contains("--mmap-probe-gate-only"))
    exit(0)
}

// `--mmap-p5-probe`: 48 §13. The fault cost 48 §11 found inside the command
// buffer (4.00 ms of the 4.25, with every page already in the page cache) is
// what decides whether D is worth anything, so this sweeps the four ways of
// trying to move it out -- `MTLResidencySet` requested off-thread, a CPU
// pre-touch, both, against today's `useResource`.
if arguments.contains("--mmap-p5-probe") {
    var probeModel = "scratch/gemma4-qat-sym.gturbo"
    if let index = arguments.firstIndex(of: "--mmap-probe-model"), index + 1 < arguments.count {
        probeModel = arguments[index + 1]
    }
    var probeRounds = 10
    if let index = arguments.firstIndex(of: "--mmap-p5-rounds"),
       index + 1 < arguments.count, let value = Int(arguments[index + 1]) {
        probeRounds = value
    }
    var probeOrder = MmapP5Order.palindrome
    if let index = arguments.firstIndex(of: "--mmap-p5-order"),
       index + 1 < arguments.count,
       let value = MmapP5Order(rawValue: arguments[index + 1]) {
        probeOrder = value
    }
    var probePollute = 0
    if let index = arguments.firstIndex(of: "--mmap-p5-pollute"),
       index + 1 < arguments.count, let value = Int(arguments[index + 1]) {
        probePollute = value
    }
    var probeExperts = 8
    if let index = arguments.firstIndex(of: "--mmap-p5-experts"),
       index + 1 < arguments.count, let value = Int(arguments[index + 1]) {
        probeExperts = max(1, min(bpwTileExperts, value))
    }
    var probeOnly: MmapP5Arm?
    if let index = arguments.firstIndex(of: "--mmap-p5-only"), index + 1 < arguments.count {
        let wanted = arguments[index + 1]
        probeOnly = MmapP5Arm.allCases.first { $0.rawValue.hasPrefix(wanted + " ") }
    }
    try runMmapP5FaultProbe(groupSize: groupSizes[0], modelPath: probeModel,
                            rounds: probeRounds, order: probeOrder,
                            pollute: probePollute, only: probeOnly,
                            fixtureEarly: arguments.contains("--mmap-p5-fixture-early"),
                            experts: probeExperts)
    exit(0)
}

if arguments.contains("--bpw-probe") {
    var probeIterations = 50
    if let index = arguments.firstIndex(of: "--moe-rows-bench-iterations"),
       index + 1 < arguments.count,
       let value = Int(arguments[index + 1]) {
        probeIterations = value
    }
    try runBpwProbe(groupSize: groupSizes[0], iterations: probeIterations)
    exit(0)
}

var results: [CaseResult] = []

func printCases(_ cases: [CaseResult]) {
    for entry in cases {
        let status = entry.passed ? "PASS" : "FAIL"
        let rel = String(format: "%.3e", entry.relativeError)
        let tol = String(format: "%.1e", entry.tolerance)
        print("  \(status)  \(entry.name)  rel=\(rel) tol=\(tol)  \(entry.detail)")
    }
}

if !visionOnly && !modelOnly {

// Every INT4 case runs once per scheme: the `sym` library has to reproduce the
// `affine` library's answers on weights that satisfy `bias == -8 * scale`. The
// INT8 and BF16 slots do not have a scheme and simply run twice.
let schemes: [Quantization.AffineScheme] =
    arguments.contains("--affine-only") ? [.affine] : [.affine, .sym]

for groupSize in groupSizes {
  for scheme in schemes {
    affineScheme = scheme
    print("=== affine group size \(groupSize), scheme \(scheme.rawValue) ===")
    var pass: [CaseResult] = []
    // Block-only shapes (production hidden dims).
    pass.append(try checkInt4GEMV(m: 128, n: 2816, groupSize: groupSize, seed: 0xA1))
    pass.append(try checkInt4GEMV(m: 64, n: 4096, groupSize: groupSize, seed: 0xA2))
    // Tail shapes: shared-expert intermediate (2112) and MoE intermediate (704).
    pass.append(try checkInt4GEMV(m: 128, n: 2112, groupSize: groupSize, seed: 0xA3))
    pass.append(try checkInt4GEMV(m: 128, n: 704, groupSize: groupSize, seed: 0xA4))
    pass.append(try checkRoutedMoE(d: 2816, f: 704, groupSize: groupSize, seed: 0xB1))
    pass.append(try checkLMHeadGreedy(d: 2816, vocab: 2048, draws: 4,
                                      groupSize: groupSize, seed: 0xC1))
    pass.append(try checkRouter(weightBits: 8, d: 2816, numExperts: 128, draws: 4,
                                groupSize: groupSize, seed: 0xD1))
    pass.append(try checkRouter(weightBits: 16, d: 2816, numExperts: 128, draws: 4,
                                groupSize: groupSize, seed: 0xD2))
    pass.append(try checkInt8GEMV(m: 128, n: 2112, groupSize: groupSize, seed: 0xE1))
    pass.append(try checkSharedExpertInt8(d: 2816, f: 2112, groupSize: groupSize,
                                          seed: 0xE2))
    // Prefill QMM: whole tiles at the shared-MLP gate shape, then a token and
    // row count that both fall mid-tile.
    pass.append(try checkPrefillInt4QMM(t: 64, n: 2112, k: 2816,
                                        groupSize: groupSize, seed: 0xF1))
    pass.append(try checkPrefillInt4QMM(t: 131, n: 100, k: 2816,
                                        groupSize: groupSize, seed: 0xF2))
    pass.append(try checkPrefillInt4QMM(t: 7, n: 64, k: 128,
                                        groupSize: groupSize, seed: 0xF3))

    printCases(pass)
    results.append(contentsOf: pass)
  }
  affineScheme = .affine
}
}

// The tower's weights are BF16 and its kernels never read an affine group, so
// this suite runs once. It is checked at the first requested group size only to
// share one compiled shader library with the run above.
if !modelOnly {
    let context = try makeContext(groupSize: groupSizes[0])
    print("=== vision tower (BF16; independent of the affine group size) ===")
    var pass: [CaseResult] = []
    // Whole tiles, then the three tails the tower actually produces: N = 4304
    // (MLP gate/up), K = 4304 (MLP down, 16 short of a whole K tile), and a
    // small shape whose T, N and K all fall mid-tile with K under one tile.
    pass.append(try checkVisionQMM(context: context, t: 64, n: 1152, k: 1152, seed: 0x71))
    pass.append(try checkVisionQMM(context: context, t: 131, n: 4304, k: 1152, seed: 0x72))
    pass.append(try checkVisionQMM(context: context, t: 37, n: 1152, k: 4304, seed: 0x73))
    pass.append(try checkVisionQMM(context: context, t: 7, n: 100, k: 72, seed: 0x74))
    // Non-square grids throughout: a square one hides an x/y transposition.
    pass.append(contentsOf: try checkVisionPatchEmbed(context: context,
                                                      patchesWide: 9, patchesHigh: 5,
                                                      seed: 0x75))
    pass.append(contentsOf: try checkVisionQKNormRoPE(context: context,
                                                      patchesWide: 9, patchesHigh: 5,
                                                      numHeads: 4, seed: 0x76))
    pass.append(contentsOf: try checkVisionAttention(context: context, patchCount: 48,
                                                     numHeads: 3, seed: 0x77))
    pass.append(contentsOf: try checkVisionAttention(context: context, patchCount: 200,
                                                     numHeads: 2, seed: 0x78))
    pass.append(try checkVisionMLPActivation(context: context, count: 4304 * 7, seed: 0x79))
    // Production hidden size; a row count that is not a whole warp of rows.
    pass.append(contentsOf: try checkVisionNormResidualAdd(context: context,
                                                           t: 37, d: 1152, seed: 0x7B))
    pass.append(contentsOf: try checkVisionPoolStandardize(context: context,
                                                           patchesWide: 9, patchesHigh: 6,
                                                           seed: 0x7A))
    printCases(pass)
    results.append(contentsOf: pass)

    if runBench {
        print("")
        try runVisionBench(context: context, softTokens: benchSoftTokens)
    }
}

// Checks that need an installed `.gturbo`. They share one context — and
// therefore one compiled shader library — but they are not part of the
// synthetic suites above, so `--model-only` keeps them.
if visionTowerModel != nil || draftModel != nil {
    let context = try makeContext(groupSize: groupSizes[0])
    if let modelPath = visionTowerModel {
        print("")
        let towerCases = try runVisionTowerChecks(context: context,
                                                  modelPath: modelPath,
                                                  fixtureRoot: visionFixtureRoot,
                                                  bench: runBench)
        printCases(towerCases)
        results.append(contentsOf: towerCases)
    }

    if let modelPath = draftModel {
        print("")
        let draftCases = try runDraftChecks(context: context,
                                            modelPath: modelPath,
                                            fixtureRoot: draftFixtureRoot)
        printCases(draftCases)
        results.append(contentsOf: draftCases)
    }
}

// M4. This one builds its own context: it loads the 26B target, whose manifest
// picks the affine group size, and `RealForwardRunner` sets it on a context
// that has not compiled a pipeline yet.
if let modelPath = verifyBlockModel {
    print("")
    let verifyCases = try await runVerifyBlockChecks(
        modelPath: modelPath,
        blockTokens: verifyBlockTokens,
        rounds: verifyRounds,
        prompt: verifyPrompt,
        images: verifyImages,
        thinking: verifyThinking,
        imageTokens: verifyImageTokens,
        cold: verifyCold,
        costOnly: arguments.contains("--verify-cost-only"))
    printCases(verifyCases)
    results.append(contentsOf: verifyCases)
}

let failures = results.filter { !$0.passed }
print("")
if failures.isEmpty {
    var scope: [String] = []
    if !visionOnly && !modelOnly { scope.append("group sizes \(groupSizes)") }
    if !modelOnly { scope.append("vision") }
    if visionTowerModel != nil { scope.append("vision tower") }
    if draftModel != nil { scope.append("drafter") }
    if verifyBlockModel != nil { scope.append("verify block") }
    if results.isEmpty {
        print("no cases ran (\(scope.joined(separator: " + ")))")
    } else {
        print("PASS  \(results.count) cases (\(scope.joined(separator: " + ")))")
    }
    exit(0)
}
print("FAIL  \(failures.count)/\(results.count) cases")
for failure in failures {
    print("  group \(failure.groupSize): \(failure.name) — \(failure.detail)")
}
exit(1)
