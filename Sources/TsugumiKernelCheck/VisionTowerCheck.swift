import Foundation
import Metal
import Tsugumi
import TsugumiValidationSupport

// PLAN_VISION.md §6-1 layer B: the assembled tower against upstream's own
// intermediates.
//
// The fixtures carry the reference `pixel_values`, so the resize never enters
// this comparison. That is the whole point of layer B — our CoreGraphics
// downscale differs from torchvision's antialiased bicubic by construction
// (§4-3), and if that difference were mixed in here there would be no way to
// tell a misread of the algorithm from a different resampler. Layer C measures
// the resize separately, against these same numbers.
//
//   swift run -c release TsugumiKernelCheck --vision-tower <model.moepack>
//
// The tower's weights come from the installed model, not from `scratch/`, so
// this exercises the real loading path: manifest -> vision/vision_weights.bin
// -> schema check -> mmap -> per-layer accessors.

/// Stages compared against the reference dump, in the order the tower produces
/// them. Reporting them all is what turns "the soft tokens are wrong" into "the
/// divergence starts at layer N".
private struct StageComparison {
    let name: String
    let actual: [Float]
    let reference: [Float]
    let tolerance: VisionTowerTolerance
}

/// Both the worst element and the bulk, because they answer different
/// questions.
///
/// The pass/fail metric is the worst element, as everywhere else in this
/// harness. But a maximum taken over four times as many elements samples four
/// times as much of the error distribution's tail, so a max that grows with the
/// patch count is not by itself evidence that anything is worse — the RMS is
/// what says whether the *distribution* moved. Reporting the reference's own
/// magnitude alongside them is what makes the FP16 quantum comparable: at
/// magnitude 512 an FP16 step is 0.5, and a residual stream carried in FP16
/// through 54 joins cannot be closer to a float32 reference than that.
private struct ErrorProfile {
    let maxRelative: Double
    let rmsRelative: Double
    let maxReference: Double
    /// Size of one FP16 step at the reference's largest magnitude, relative to
    /// that magnitude — the floor no FP16 pipeline can beat.
    var fp16Quantum: Double {
        guard maxReference > 0 else { return 0 }
        let exponent = (Double(maxReference).exponent)
        return exp2(Double(exponent - 10)) / maxReference
    }

    var detail: String {
        String(format: "rms=%.1e maxRef=%.3g fp16ulp=%.1e",
               rmsRelative, maxReference, fp16Quantum)
    }
}

private func errorProfile(actual: [Float], reference: [Float]) -> ErrorProfile {
    precondition(actual.count == reference.count, "length mismatch")
    var maxDiff = 0.0
    var sumSquaredDiff = 0.0
    var maxRef = 0.0
    for i in 0..<actual.count {
        let a = Double(actual[i])
        let r = Double(reference[i])
        precondition(r.isFinite, "reference is not finite at \(i) — harness bug")
        maxRef = Swift.max(maxRef, abs(r))
        guard a.isFinite else { return ErrorProfile(maxRelative: .infinity,
                                                    rmsRelative: .infinity,
                                                    maxReference: maxRef) }
        let diff = abs(a - r)
        maxDiff = Swift.max(maxDiff, diff)
        sumSquaredDiff += diff * diff
    }
    precondition(maxRef > 1e-4, "reference has no signal — harness bug")
    let rms = (sumSquaredDiff / Double(actual.count)).squareRoot()
    return ErrorProfile(maxRelative: maxDiff / maxRef,
                        rmsRelative: rms / maxRef,
                        maxReference: maxRef)
}

/// The §6-1 layer B gate, set from measurement (`PLAN_VISION.md` §6-1 left the
/// number to V4: "V4 で実測してから確定").
///
/// Two thresholds per stage, because the worst element and the bulk answer
/// different questions and neither alone is a gate:
///
///  * `max` guards against a single blown element — the shape a real indexing
///    or geometry bug takes. It has to sit above the FP16 floor, which
///    `Scripts/vision/fp16_error_floor.py` measured by running the *upstream*
///    implementation at our precision: 5.7e-2 on the worst case
///    (tall-480x1200 at 280 soft tokens), against our 6.2e-2. A tighter number
///    would be rejecting FP16 itself, not our arithmetic.
///  * `rms` is the sensitive instrument: it is flat in the patch count where
///    the max is not, it sits within 1.4x of that same floor on all six
///    fixtures, and at 2e-3 it is an order of magnitude *tighter* than the
///    2e-2 the plan pencilled in. Anything systematic moves it.
///
/// Both are roughly 2x the largest measured value, which is enough headroom
/// that a driver or compiler change does not flake the check and little enough
/// that a regression cannot hide under it. The detection cases below run at the
/// same thresholds and clear them by 300x.
private struct VisionTowerTolerance {
    let max: Double
    let rms: Double

    static let patchEmbed = VisionTowerTolerance(max: 2e-3, rms: 5e-5)
    static let earlyLayer = VisionTowerTolerance(max: 2e-3, rms: 5e-5)
    static let midLayer = VisionTowerTolerance(max: 8e-3, rms: 6e-5)
    static let lateLayer = VisionTowerTolerance(max: 8e-2, rms: 8e-4)
    static let pooled = VisionTowerTolerance(max: 8e-2, rms: 3e-3)
    /// The exit condition of V4.
    static let softTokens = VisionTowerTolerance(max: 8e-2, rms: 2e-3)
}

func runVisionTowerChecks(context: MetalContext,
                          modelPath: String,
                          fixtureRoot: String,
                          bench: Bool) throws -> [CaseResult] {
    let directoryURL = URL(fileURLWithPath: modelPath)
    let arch = ArchConfig.gemma4_26B_A4B
    let manifest = try ManifestReader.load(directoryURL: directoryURL, expecting: arch)
    guard let declared = manifest.vision else {
        throw VisionError.towerNotInstalled(path: modelPath)
    }

    let loadStart = Date()
    let weights = try VisionWeights.load(directoryURL: directoryURL,
                                         manifest: manifest,
                                         config: arch,
                                         device: context.device,
                                         integrityPolicy: .fullSha256)
    let loadSeconds = Date().timeIntervalSince(loadStart)
    let tower = try VisionTower(context: context, weights: weights)

    print("=== vision tower vs reference fixtures (PLAN_VISION §6-1 layer B) ===")
    print("  model    \(directoryURL.lastPathComponent) -> \(declared.weightsPath)")
    print(String(format: "  tower    %d tensors, %llu bytes, verified + mapped in %.2f s",
                 declared.tensorCount, declared.payloadBytes, loadSeconds))
    print(String(format: "  scratch  %.1f MB for %d patches",
                 Double(tower.scratchBytes) / 1e6, tower.maxPatchCount))
    print("  source   \(declared.sourceRepo) @ \(String(declared.sourceRevision.prefix(12)))")

    guard let cases = try VisionFixtures.load(root: fixtureRoot) else {
        throw VisionCheckError.fixturesMissing(fixtureRoot)
    }
    guard !cases.isEmpty else {
        throw VisionCheckError.fixturesMissing(fixtureRoot)
    }

    let probeLayers = [0, 13, 26].filter { $0 < weights.config.numLayers }
    var results: [CaseResult] = []

    for fixture in cases.sorted(by: { $0.name < $1.name }) {
        let run = try runTower(context: context, tower: tower, fixture: fixture,
                               probeLayers: probeLayers, fault: .none)

        var comparisons: [StageComparison] = [
            StageComparison(name: "patch-embed",
                            actual: run.stage(.patchEmbed),
                            reference: try fixture.tensor("patch_embed").values,
                            tolerance: VisionTowerTolerance.patchEmbed),
        ]
        for (position, layer) in probeLayers.enumerated() {
            let tolerance: VisionTowerTolerance
            switch position {
            case 0: tolerance = .earlyLayer
            case probeLayers.count - 1: tolerance = .lateLayer
            default: tolerance = .midLayer
            }
            comparisons.append(StageComparison(
                name: "layer\(layer)",
                actual: run.stage(.layer(layer)),
                reference: try fixture.tensor("layer\(layer)").values,
                tolerance: tolerance))
        }
        comparisons.append(StageComparison(name: "pooled",
                                           actual: run.stage(.pooled),
                                           reference: try fixture.tensor("pooled").values,
                                           tolerance: VisionTowerTolerance.pooled))
        comparisons.append(StageComparison(name: "soft-tokens",
                                           actual: run.softTokens,
                                           reference: try fixture.tensor("soft_tokens").values,
                                           tolerance: VisionTowerTolerance.softTokens))

        let shape = "P=\(fixture.patchCount) "
            + "\(fixture.patchesWide)x\(fixture.patchesHigh) "
            + "S=\(fixture.softTokenCount)"
        for comparison in comparisons {
            let profile = errorProfile(actual: comparison.actual,
                                       reference: comparison.reference)
            results.append(result("vision-tower/\(fixture.name)/\(comparison.name)/max",
                                  groupSize: context.affineGroupSize,
                                  rel: profile.maxRelative,
                                  tolerance: comparison.tolerance.max,
                                  detail: shape + " " + profile.detail))
            results.append(result("vision-tower/\(fixture.name)/\(comparison.name)/rms",
                                  groupSize: context.affineGroupSize,
                                  rel: profile.rmsRelative,
                                  tolerance: comparison.tolerance.rms,
                                  detail: shape))
        }
    }

    results.append(contentsOf: try runDetectionChecks(context: context,
                                                      tower: tower,
                                                      cases: cases,
                                                      probeLayers: probeLayers))

    if bench {
        print("")
        try benchVisionTower(context: context, tower: tower, cases: cases)
    }

    return results
}

// MARK: - Detection power (§6-3)

/// The same comparison, run against a tower that has been broken on purpose.
///
/// A grid transposition and a dropped standardization would both leave output
/// that looks like soft tokens; pinning every layer to layer 0's weights is the
/// failure an accessor bug would produce. All three must clear the tolerance
/// the positive case just met by a wide margin, or the comparison above is not
/// evidence of anything.
private func runDetectionChecks(context: MetalContext,
                                tower: VisionTower,
                                cases: [VisionFixtures.Case],
                                probeLayers: [Int]) throws -> [CaseResult] {
    // A square grid hides an x/y transposition on the diagonal, so the
    // detection cases run on the most rectangular fixture available.
    guard let fixture = cases.max(by: { lhs, rhs in
        abs(lhs.patchesWide - lhs.patchesHigh) < abs(rhs.patchesWide - rhs.patchesHigh)
    }), fixture.patchesWide != fixture.patchesHigh else {
        throw VisionCheckError.noRectangularFixture
    }

    let reference = try fixture.tensor("soft_tokens").values
    var results: [CaseResult] = []
    for fault in VisionTower.Fault.allCases where fault != .none {
        let run = try runTower(context: context, tower: tower, fixture: fixture,
                               probeLayers: probeLayers, fault: fault)
        let profile = errorProfile(actual: run.softTokens, reference: reference)
        let shape = "\(fixture.name) \(fixture.patchesWide)x\(fixture.patchesHigh)"
        // The floors are 6x the tolerances the same comparison passes at, so a
        // fault cannot be mistaken for FP16 noise in either metric.
        results.append(detectionResult(
            "vision-tower/detect/\(fault.rawValue)/max",
            groupSize: context.affineGroupSize,
            rel: profile.maxRelative,
            floor: VisionTowerTolerance.softTokens.max * 6,
            detail: shape))
        results.append(detectionResult(
            "vision-tower/detect/\(fault.rawValue)/rms",
            groupSize: context.affineGroupSize,
            rel: profile.rmsRelative,
            floor: VisionTowerTolerance.softTokens.rms * 6,
            detail: shape))
    }
    return results
}

// MARK: - Running one image

private struct TowerRun {
    let softTokens: [Float]
    private let stages: [VisionTowerProbes.Stage: [Float]]

    init(softTokens: [Float], stages: [VisionTowerProbes.Stage: [Float]]) {
        self.softTokens = softTokens
        self.stages = stages
    }

    func stage(_ stage: VisionTowerProbes.Stage) -> [Float] {
        guard let values = stages[stage] else {
            fatalError("probe \(stage) was not captured — harness bug")
        }
        return values
    }
}

private func runTower(context: MetalContext,
                      tower: VisionTower,
                      fixture: VisionFixtures.Case,
                      probeLayers: [Int],
                      fault: VisionTower.Fault) throws -> TowerRun {
    let config = tower.config
    let hidden = config.hiddenSize
    let geometry = try resolveGeometry(for: fixture)
    let pixels = try fixture.tensor("pixel_values")
    let patchDim = config.patchSize * config.patchSize * 3
    guard pixels.values.count == geometry.patchCount * patchDim else {
        throw VisionCheckError.fixtureShapeMismatch(
            "\(fixture.name) pixel_values has \(pixels.values.count) values, "
            + "expected \(geometry.patchCount * patchDim)")
    }

    // One FP32 -> FP16 rounding the reference does not have. It is the same
    // rounding the preprocessor applies in production, and it is inside the
    // tolerance rather than excluded from it.
    guard let pixelBuffer = Fp16Buffer.make(context.device, values: pixels.values) else {
        fatalError("buffer allocation failed")
    }
    let probes = try VisionTowerProbes(device: context.device,
                                       hiddenSize: hidden,
                                       patchCount: geometry.patchCount,
                                       softTokenCount: geometry.softTokenCount,
                                       layers: probeLayers)
    guard let cmd = context.queue.makeCommandBuffer() else {
        fatalError("command buffer allocation failed")
    }
    let output = try tower.encode(commandBuffer: cmd,
                                  pixels: pixelBuffer,
                                  geometry: geometry,
                                  fault: fault,
                                  probes: probes)
    waitAndCheck(cmd, "vision-tower \(fixture.name) fault=\(fault.rawValue)")

    var stages: [VisionTowerProbes.Stage: [Float]] = [:]
    func read(_ stage: VisionTowerProbes.Stage, count: Int) {
        guard let buffer = probes.buffer(stage) else { return }
        stages[stage] = Fp16Buffer.read(buffer, count: count)
    }
    read(.patchEmbed, count: geometry.patchCount * hidden)
    for layer in probeLayers { read(.layer(layer), count: geometry.patchCount * hidden) }
    read(.pooled, count: geometry.softTokenCount * hidden)

    return TowerRun(softTokens: Fp16Buffer.read(output.softTokens,
                                                count: output.softTokenCount * output.hiddenSize),
                    stages: stages)
}

/// The geometry the runtime derives for this fixture's image, cross-checked
/// against the grid the reference recorded. V1 pins that equality on its own;
/// re-asserting it here keeps a geometry regression from being reported as a
/// tower error.
private func resolveGeometry(for fixture: VisionFixtures.Case) throws -> VisionImageGeometry {
    let preprocessor = try VisionPreprocessorConfig(maxSoftTokens: fixture.maxSoftTokens)
    let geometry = try preprocessor.geometry(imageWidth: fixture.imageWidth,
                                             imageHeight: fixture.imageHeight)
    guard geometry.patchesWide == fixture.patchesWide,
          geometry.patchesHigh == fixture.patchesHigh,
          geometry.softTokenCount == fixture.softTokenCount else {
        throw VisionCheckError.fixtureShapeMismatch("""
            \(fixture.name): runtime geometry \(geometry.patchesWide)x\(geometry.patchesHigh) \
            (\(geometry.softTokenCount) soft tokens) != reference \
            \(fixture.patchesWide)x\(fixture.patchesHigh) (\(fixture.softTokenCount))
            """)
    }
    return geometry
}

// MARK: - Throughput

/// End-to-end tower time, which §0-F-4 left open: the V3 numbers were a sum of
/// individually benchmarked kernels and excluded the four RMSNorms per layer
/// (reused kernels the per-kernel bench could not reach) and the residual
/// joins. This measures the whole assembled pass on the largest fixture.
private func benchVisionTower(context: MetalContext,
                              tower: VisionTower,
                              cases: [VisionFixtures.Case]) throws {
    print("=== vision tower, end to end ===")
    print("  \("shape".padding(toLength: 30, withPad: " ", startingAt: 0))"
          + "     GPU s     wall s      TFLOP    TFLOP/s")

    if let fixture = cases.max(by: { $0.patchCount < $1.patchCount }) {
        let geometry = try resolveGeometry(for: fixture)
        let pixels = try fixture.tensor("pixel_values")
        guard let buffer = Fp16Buffer.make(context.device, values: pixels.values) else {
            fatalError("buffer allocation failed")
        }
        benchOne(context: context, tower: tower, geometry: geometry,
                 label: fixture.name, pixels: buffer)
    }

    // The largest fixture is not the largest image: no test image happens to
    // land on the full 280-soft-token budget (§0-B-2), and V3's 1.37 s estimate
    // was quoted at P = 2520. A 1000x700 image resizes to exactly 960x672,
    // which is 60x42 patches — the worst case the tower has to handle. Its
    // pixels are synthetic because only the shape matters to the clock.
    let preprocessor = try VisionPreprocessorConfig()
    let worstCase = try preprocessor.geometry(imageWidth: 1000, imageHeight: 700)
    precondition(worstCase.patchCount == tower.maxPatchCount,
                 "expected the 280-soft-token maximum, got \(worstCase.patchCount)")
    var rng = SeedTree(0x5118).key("bench-tower-worst-case")
    let patchDim = tower.config.patchSize * tower.config.patchSize * 3
    let synthetic = (0..<(worstCase.patchCount * patchDim)).map { _ in rng.uniform(0, 1) }
    guard let syntheticBuffer = Fp16Buffer.make(context.device, values: synthetic) else {
        fatalError("buffer allocation failed")
    }
    benchOne(context: context, tower: tower, geometry: worstCase,
             label: "worst case (1000x700)", pixels: syntheticBuffer)
}

private func benchOne(context: MetalContext,
                      tower: VisionTower,
                      geometry: VisionImageGeometry,
                      label: String,
                      pixels: MTLBuffer) {
    let iterations = 3
    let gpu = gpuSeconds(context: context, iterations: iterations,
                         label: "bench vision tower \(label)") { cmd in
        _ = try? tower.encode(commandBuffer: cmd, pixels: pixels, geometry: geometry)
    }

    // Wall clock covers what the GPU timestamps leave out and what §0-F-4 asked
    // for: encoding ~380 dispatches, allocating the per-image output buffer,
    // and the queue round trip. That difference is the honest TTFT contribution,
    // not the GPU figure.
    let wallStart = Date()
    for _ in 0..<iterations {
        guard let cmd = context.queue.makeCommandBuffer() else {
            fatalError("command buffer allocation failed")
        }
        _ = try? tower.encode(commandBuffer: cmd, pixels: pixels, geometry: geometry)
        waitAndCheck(cmd, "bench vision tower wall \(label)")
    }
    let wall = Date().timeIntervalSince(wallStart) / Double(iterations)

    // Same accounting as `runVisionBench`: Q/K/V/O and the MLP as 2*T*N*K, the
    // attention as QK^T plus the value accumulation, and the two one-off
    // projections at the ends.
    let config = tower.config
    let patches = Double(geometry.patchCount)
    let hidden = Double(config.hiddenSize)
    let intermediate = Double(config.intermediateSize)
    let layers = Double(config.numLayers)
    let projections = 2 * patches * hidden * hidden * 4 + 2 * patches * intermediate * hidden * 3
    let attention = 4 * patches * patches * Double(config.headDim) * Double(config.numHeads)
    let once = 2 * patches * hidden * Double(config.patchSize * config.patchSize * 3)
        + 2 * Double(geometry.softTokenCount) * Double(tower.textHiddenSize) * hidden
    let flops = (projections + attention) * layers + once

    let shape = "\(label) P=\(geometry.patchCount) S=\(geometry.softTokenCount)"
    print(String(format: "  %@  %9.3f  %9.3f  %9.2f  %9.2f",
                 shape.padding(toLength: 30, withPad: " ", startingAt: 0),
                 gpu, wall, flops / 1e12, flops / gpu / 1e12))
}

// MARK: - Errors

enum VisionCheckError: Error, CustomStringConvertible {
    case fixturesMissing(String)
    case fixtureShapeMismatch(String)
    case noRectangularFixture

    var description: String {
        switch self {
        case let .fixturesMissing(root):
            return """
                no vision fixtures at \(root); generate them with \
                Scripts/vision/dump_vision_fixtures.py (PLAN_VISION §5-V0)
                """
        case let .fixtureShapeMismatch(detail):
            return detail
        case .noRectangularFixture:
            return """
                every fixture has a square patch grid, so a transposed grid \
                cannot be detected (PLAN_VISION §0-F-5)
                """
        }
    }
}
