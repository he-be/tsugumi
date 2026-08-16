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

    var passed: Bool { relativeError <= tolerance }
}

func result(_ name: String, groupSize: Int, rel: Double, tolerance: Double,
            detail: String = "") -> CaseResult {
    CaseResult(name: name, groupSize: groupSize,
               relativeError: rel, tolerance: tolerance, detail: detail)
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

func quantizedRows(count: Int, n: Int, groupSize: Int,
                   rng: inout SplitMix64) -> [Quantization.Int4AffineRow] {
    var rows: [Quantization.Int4AffineRow] = []
    rows.reserveCapacity(count)
    for _ in 0..<count {
        let raw = (0..<n).map { _ in rng.uniform(-0.5, 0.5) }
        rows.append(Quantization.quantizeInt4Affine(raw, groupSize: groupSize))
    }
    return rows
}

func makeContext(groupSize: Int) throws -> MetalContext {
    let context = try MetalContext()
    try context.setAffineGroupSize(groupSize)
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
            rows.map { Quantization.quantizeInt4Affine($0, groupSize: groupSize) }
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

// MARK: - Driver

let arguments = CommandLine.arguments
var groupSizes = [64, 32]
if let index = arguments.firstIndex(of: "--group-size"),
   index + 1 < arguments.count,
   let value = Int(arguments[index + 1]) {
    groupSizes = [value]
}

var results: [CaseResult] = []

for groupSize in groupSizes {
    print("=== affine group size \(groupSize) ===")
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

    for entry in pass {
        let status = entry.passed ? "PASS" : "FAIL"
        let rel = String(format: "%.3e", entry.relativeError)
        let tol = String(format: "%.1e", entry.tolerance)
        print("  \(status)  \(entry.name)  rel=\(rel) tol=\(tol)  \(entry.detail)")
    }
    results.append(contentsOf: pass)
}

let failures = results.filter { !$0.passed }
print("")
if failures.isEmpty {
    print("PASS  \(results.count) cases across group sizes \(groupSizes)")
    exit(0)
}
print("FAIL  \(failures.count)/\(results.count) cases")
for failure in failures {
    print("  group \(failure.groupSize): \(failure.name) — \(failure.detail)")
}
exit(1)
