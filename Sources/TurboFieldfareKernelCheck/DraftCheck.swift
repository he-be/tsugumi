import Foundation
import Metal
import TurboFieldfare
import TurboFieldfareValidationSupport

// M2 (docs/mtp/04-PHASES.md): the assembled MTP drafter forward against
// upstream's own intermediates, on synthetic inputs.
//
//   swift run -c release TurboFieldfareKernelCheck \
//     --draft scratch/gemma4-qat.gturbo
//
// The fixtures come from Scripts/mtp/dump_draft_fixtures.py, which runs
// transformers' `Gemma4AssistantForCausalLM` (float32) on random
// `(target_embed, target_hidden, shared KV)` triples — the 26B target is
// never loaded on either side, so the comparison isolates the drafter
// forward exactly (04-PHASES §5).
//
// The reference multiplies nothing by √2816: the fixture stores the raw
// target embedding and this harness applies the scale when it converts to
// FP16, which is where production applies it too (the fused embed lookup's
// `outScale`). Everything downstream of that rounding is the comparison's
// subject.

private struct DraftCase {
    let name: String
    let kvLen: Int
    let position: Int
    let argmax: Int
    let directory: URL
}

/// Tolerances for the stage comparison, set from measurement
/// (`docs/mtp/12-M2-RESULTS.md` §2) with the same ~2x headroom policy as the
/// vision tower. The large late-stage numbers are not slack: the reference's
/// *own* FP16 floor was measured at max 1.9e-1 / rms 3.2e-2 on the
/// past-window case (`Scripts/mtp/fp16_error_floor.py`), and this pipeline
/// sits 2.5x below it (attention partials accumulate in FP32), so anything
/// systematic still moves these by orders of magnitude — the detection cases
/// below clear them by 2-4x.
struct DraftTolerance {
    let max: Double
    let rms: Double

    static let hPre = DraftTolerance(max: 1e-3, rms: 2e-4)
    static let layer = DraftTolerance(max: 8e-2, rms: 2.5e-2)
    static let hNorm = DraftTolerance(max: 1.5e-1, rms: 3e-2)
    static let lastHidden = DraftTolerance(max: 1.2e-1, rms: 2.5e-2)
    static let logits = DraftTolerance(max: 2e-1, rms: 5e-2)

    /// Detection floors sit between the worst measured positive error
    /// (max 7.2e-2 / rms 1.6e-2) and the smallest fault error
    /// (max 2.2e-1 / rms 5.7e-2).
    static let detectionFloorMax = 1.3e-1
    static let detectionFloorRMS = 3.5e-2
}

private struct StageComparison {
    let name: String
    let actual: [Float]
    let reference: [Float]
    let tolerance: DraftTolerance
}

private struct Profile {
    let maxRelative: Double
    let rmsRelative: Double
}

private func profile(actual: [Float], reference: [Float]) -> Profile {
    precondition(actual.count == reference.count, "length mismatch")
    var maxDiff = 0.0
    var sumSquared = 0.0
    var maxRef = 0.0
    for i in 0..<actual.count {
        let a = Double(actual[i])
        let r = Double(reference[i])
        precondition(r.isFinite, "reference is not finite at \(i) — harness bug")
        maxRef = Swift.max(maxRef, abs(r))
        guard a.isFinite else { return Profile(maxRelative: .infinity, rmsRelative: .infinity) }
        let diff = abs(a - r)
        maxDiff = Swift.max(maxDiff, diff)
        sumSquared += diff * diff
    }
    precondition(maxRef > 1e-4, "reference has no signal — harness bug")
    let rms = (sumSquared / Double(actual.count)).squareRoot()
    return Profile(maxRelative: maxDiff / maxRef, rmsRelative: rms / maxRef)
}

private func loadDraftCases(root: String) throws -> [DraftCase] {
    let rootURL = URL(fileURLWithPath: root)
    let manifestURL = rootURL.appendingPathComponent("manifest.json")
    guard FileManager.default.fileExists(atPath: manifestURL.path) else {
        throw DraftCheckError.fixturesMissing(root)
    }
    let data = try Data(contentsOf: manifestURL)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let cases = object["cases"] as? [[String: Any]] else {
        throw DraftCheckError.fixtureShapeMismatch("manifest has no `cases` array")
    }
    return try cases.map { entry in
        guard let name = entry["name"] as? String,
              let dir = entry["dir"] as? String,
              let kvLen = entry["kv_len"] as? Int,
              let position = entry["position"] as? Int,
              let argmax = entry["argmax"] as? Int else {
            throw DraftCheckError.fixtureShapeMismatch("malformed case entry")
        }
        return DraftCase(name: name, kvLen: kvLen, position: position, argmax: argmax,
                         directory: rootURL.appendingPathComponent(dir))
    }
}

enum DraftCheckError: Error, CustomStringConvertible {
    case fixturesMissing(String)
    case fixtureShapeMismatch(String)

    var description: String {
        switch self {
        case let .fixturesMissing(root):
            return """
                no draft fixtures at \(root); generate them with \
                Scripts/mtp/dump_draft_fixtures.py (docs/mtp/04-PHASES.md §5)
                """
        case let .fixtureShapeMismatch(detail):
            return detail
        }
    }
}

private func fixtureTensor(_ case_: DraftCase, _ stem: String) throws -> VisionFixtures.Tensor {
    try VisionFixtures.readTensor(at: case_.directory.appendingPathComponent("\(stem).bin"))
}

/// `[1, heads, len, dim]` FP32 fixture → `[len, heads, dim]` FP16 device
/// buffer, the layout decode attention reads. The transpose is harness-side:
/// production gets this layout straight from the KV cache views.
private func kvBuffer(_ device: MTLDevice, _ tensor: VisionFixtures.Tensor)
    throws -> MTLBuffer {
    guard tensor.shape.count == 4, tensor.shape[0] == 1 else {
        throw DraftCheckError.fixtureShapeMismatch(
            "KV fixture has shape \(tensor.shape), expected [1, heads, len, dim]")
    }
    let heads = tensor.shape[1]
    let len = tensor.shape[2]
    let dim = tensor.shape[3]
    var halves = [Float16](repeating: 0, count: heads * len * dim)
    for h in 0..<heads {
        for p in 0..<len {
            for d in 0..<dim {
                halves[(p * heads + h) * dim + d] =
                    Float16(tensor.values[(h * len + p) * dim + d])
            }
        }
    }
    guard let buffer = Fp16Buffer.make(device, halves: halves) else {
        fatalError("buffer allocation failed")
    }
    return buffer
}

private func fp16Buffer(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer {
    guard let buffer = Fp16Buffer.make(device, values: values) else {
        fatalError("buffer allocation failed")
    }
    return buffer
}

func runDraftChecks(context: MetalContext,
                    modelPath: String,
                    fixtureRoot: String) throws -> [CaseResult] {
    let directoryURL = URL(fileURLWithPath: modelPath)
    let arch = ArchConfig.gemma4_26B_A4B
    let manifest = try ManifestReader.load(directoryURL: directoryURL, expecting: arch)

    let loadStart = Date()
    let weights = try DraftWeights.load(directoryURL: directoryURL,
                                        manifest: manifest,
                                        arch: arch,
                                        device: context.device,
                                        integrityPolicy: .fullSha256)
    let loadSeconds = Date().timeIntervalSince(loadStart)
    let drafter = try DraftForward(context: context, weights: weights)
    let c = weights.config

    print("=== MTP drafter vs reference fixtures (docs/mtp 04-PHASES M2) ===")
    print("  model    \(directoryURL.lastPathComponent) -> \(weights.relativePath)")
    print(String(format: "  drafter  %d tensors, %d bytes, verified + mapped in %.2f s",
                 manifest.draft?.tensorCount ?? 0,
                 manifest.draft?.payloadBytes ?? 0, loadSeconds))
    print("  source   \(manifest.draft?.sourceRepo ?? "?") "
          + "@ \(String((manifest.draft?.sourceRevision ?? "").prefix(12)))")

    let cases = try loadDraftCases(root: fixtureRoot)
    guard !cases.isEmpty else {
        throw DraftCheckError.fixturesMissing(fixtureRoot)
    }
    let device = context.device
    let backbone = c.backboneHiddenSize
    let embedScale = Float(backbone).squareRoot()

    guard let outLastHidden = Fp16Buffer.make(device, count: backbone),
          let outToken = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                           options: .storageModeShared),
          let outLogits = Fp16Buffer.make(device, count: c.vocabSize) else {
        fatalError("buffer allocation failed")
    }

    var results: [CaseResult] = []

    for case_ in cases {
        let targetEmbed = try fixtureTensor(case_, "target_embed_in")
        let lastHidden = try fixtureTensor(case_, "last_hidden_in")
        let kSwa = try kvBuffer(device, try fixtureTensor(case_, "k_swa"))
        let vSwa = try kvBuffer(device, try fixtureTensor(case_, "v_swa"))
        let kFull = try kvBuffer(device, try fixtureTensor(case_, "k_full"))
        let vFull = try kvBuffer(device, try fixtureTensor(case_, "v_full"))

        // The √2816 embedding scale is applied at the FP16 conversion, which
        // is where production applies it (fused embed lookup `outScale`).
        let embedBuffer = fp16Buffer(device, targetEmbed.values.map { $0 * embedScale })
        let hiddenBuffer = fp16Buffer(device, lastHidden.values)

        let probes = try DraftProbes(device: device, hiddenSize: c.hiddenSize,
                                     numLayers: c.numLayers)
        guard let cmd = context.queue.makeCommandBuffer() else {
            fatalError("command buffer allocation failed")
        }
        try drafter.encode(commandBuffer: cmd,
                           targetEmbed: embedBuffer,
                           lastHidden: hiddenBuffer,
                           slidingK: kSwa, slidingV: vSwa,
                           fullK: kFull, fullV: vFull,
                           position: UInt32(case_.position),
                           outLastHidden: outLastHidden,
                           outToken: outToken,
                           outLogits: outLogits,
                           probes: probes)
        waitAndCheck(cmd, "draft \(case_.name)")

        var comparisons: [StageComparison] = [
            StageComparison(name: "h-pre",
                            actual: Fp16Buffer.read(probes.buffer(.hPre)!, count: c.hiddenSize),
                            reference: try fixtureTensor(case_, "h_pre").values,
                            tolerance: .hPre),
        ]
        for layer in 0..<c.numLayers {
            let stage = DraftProbes.Stage(rawValue: layer + 1)!
            comparisons.append(StageComparison(
                name: "layer\(layer)",
                actual: Fp16Buffer.read(probes.buffer(stage)!, count: c.hiddenSize),
                reference: try fixtureTensor(case_, "layer\(layer)").values,
                tolerance: .layer))
        }
        comparisons.append(StageComparison(
            name: "h-norm",
            actual: Fp16Buffer.read(probes.buffer(.hNorm)!, count: c.hiddenSize),
            reference: try fixtureTensor(case_, "h_norm").values,
            tolerance: .hNorm))
        comparisons.append(StageComparison(
            name: "last-hidden",
            actual: Fp16Buffer.read(outLastHidden, count: backbone),
            reference: try fixtureTensor(case_, "last_hidden_out").values,
            tolerance: .lastHidden))
        comparisons.append(StageComparison(
            name: "logits",
            actual: Fp16Buffer.read(outLogits, count: c.vocabSize),
            reference: try fixtureTensor(case_, "logits").values,
            tolerance: .logits))

        for comparison in comparisons {
            let p = profile(actual: comparison.actual, reference: comparison.reference)
            results.append(result("draft/\(case_.name)/\(comparison.name)/max",
                                  groupSize: context.affineGroupSize,
                                  rel: p.maxRelative,
                                  tolerance: comparison.tolerance.max,
                                  detail: "kv=\(case_.kvLen) rms=%.1e"
                                      .replacingOccurrences(of: "%.1e",
                                                            with: String(format: "%.1e", p.rmsRelative))))
            results.append(result("draft/\(case_.name)/\(comparison.name)/rms",
                                  groupSize: context.affineGroupSize,
                                  rel: p.rmsRelative,
                                  tolerance: comparison.tolerance.rms,
                                  detail: "kv=\(case_.kvLen)"))
        }

        let gpuToken = Int(outToken.contents().load(as: UInt32.self))
        results.append(result("draft/\(case_.name)/argmax",
                              groupSize: context.affineGroupSize,
                              rel: gpuToken == case_.argmax ? 0 : 1,
                              tolerance: 0,
                              detail: "gpu=\(gpuToken) reference=\(case_.argmax)"))
    }

    // Detection power (04-PHASES M2): three deliberately broken drafters must
    // clear the tolerances the positive case just met, or the comparison is
    // not evidence. Runs on the longest fixture, where a RoPE position error
    // and a window misread both have the most room to diverge.
    guard let worst = cases.max(by: { $0.kvLen < $1.kvLen }) else {
        fatalError("no cases")
    }
    let referenceNorm = try fixtureTensor(worst, "h_norm").values
    for fault in DraftForward.Fault.allCases where fault != .none {
        let targetEmbed = try fixtureTensor(worst, "target_embed_in")
        let lastHidden = try fixtureTensor(worst, "last_hidden_in")
        let embedBuffer = fp16Buffer(device, targetEmbed.values.map { $0 * embedScale })
        let hiddenBuffer = fp16Buffer(device, lastHidden.values)
        let probes = try DraftProbes(device: device, hiddenSize: c.hiddenSize,
                                     numLayers: c.numLayers)
        guard let cmd = context.queue.makeCommandBuffer() else {
            fatalError("command buffer allocation failed")
        }
        try drafter.encode(commandBuffer: cmd,
                           targetEmbed: embedBuffer,
                           lastHidden: hiddenBuffer,
                           slidingK: try kvBuffer(device, try fixtureTensor(worst, "k_swa")),
                           slidingV: try kvBuffer(device, try fixtureTensor(worst, "v_swa")),
                           fullK: try kvBuffer(device, try fixtureTensor(worst, "k_full")),
                           fullV: try kvBuffer(device, try fixtureTensor(worst, "v_full")),
                           position: UInt32(worst.position),
                           fault: fault,
                           outLastHidden: outLastHidden,
                           outToken: nil,
                           probes: probes)
        waitAndCheck(cmd, "draft detect \(fault.rawValue)")
        let p = profile(actual: Fp16Buffer.read(probes.buffer(.hNorm)!, count: c.hiddenSize),
                        reference: referenceNorm)
        results.append(detectionResult("draft/detect/\(fault.rawValue)/max",
                                       groupSize: context.affineGroupSize,
                                       rel: p.maxRelative,
                                       floor: DraftTolerance.detectionFloorMax,
                                       detail: "kv=\(worst.kvLen)"))
        results.append(detectionResult("draft/detect/\(fault.rawValue)/rms",
                                       groupSize: context.affineGroupSize,
                                       rel: p.rmsRelative,
                                       floor: DraftTolerance.detectionFloorRMS,
                                       detail: "kv=\(worst.kvLen)"))
    }

    return results
}
