import Testing
import Foundation
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// The runtime's own view of what a vision tower must contain.
///
/// `VisionWeights` derives the expected tensor table from `VisionConfig` rather
/// than believing the file's index (`PLAN_VISION.md` §0-D-2). These tests cover
/// that derivation and the checks built on it; the assembled tower is checked
/// against the reference fixtures by `TurboFieldfareKernelCheck --vision-tower`,
/// which needs a GPU and the installed 1.15 GB tower and therefore cannot live
/// here.
@Suite struct VisionWeightsSchemaTests {

    static let config = VisionConfig.gemma4Vision
    static let textHidden = ArchConfig.gemma4_26B_A4B.hiddenSize

    static func inventory() -> [(name: String, shape: [Int])] {
        VisionWeights.expectedInventory(config: config, textHiddenSize: textHidden)
    }

    /// A valid index built from the runtime's own expectations, laid out back to
    /// back after a nominal index region — the shape `VisionRepackPlanner`
    /// writes.
    static func makeIndex(
        mutate: ([(name: String, shape: [Int])]) -> [(name: String, shape: [Int])] = { $0 }
    ) -> (index: ResidentIndex, payloadBytes: UInt64, tensorCount: Int) {
        let indexSize: UInt64 = 49_152
        var cursor = indexSize
        var entries: [String: ResidentIndexEntry] = [:]
        let tensors = mutate(inventory())
        for (name, shape) in tensors {
            var padded = shape.map { UInt32($0) }
            while padded.count < 4 { padded.append(0) }
            let bytes = shape.reduce(UInt64(1)) { $0 * UInt64($1) }
                * UInt64(MemoryLayout<UInt16>.size)
            entries[name] = ResidentIndexEntry(
                name: name,
                dtype: GTurboFormatV1.DType.bf16.rawValue,
                fileOffset: cursor,
                sizeBytes: bytes,
                shape: (padded[0], padded[1], padded[2], padded[3]),
                scaleOffset: 0, scaleSize: 0,
                biasOffset: 0, biasSize: 0)
            cursor += bytes
        }
        let payload = cursor - indexSize
        return (ResidentIndex(header: ResidentIndexHeader(indexSize: indexSize,
                                                          residentSize: payload,
                                                          entryCount: UInt64(entries.count)),
                              entries: entries),
                payload,
                tensors.count)
    }

    static func manifestVision(tensorCount: Int, payloadBytes: UInt64) -> ManifestVision {
        ManifestVision(hiddenSize: config.hiddenSize,
                       numLayers: config.numLayers,
                       numHeads: config.numHeads,
                       numKVHeads: config.numKVHeads,
                       headDim: config.headDim,
                       intermediateSize: config.intermediateSize,
                       patchSize: config.patchSize,
                       poolingKernelSize: config.poolingKernelSize,
                       positionEmbeddingSize: config.positionEmbeddingSize,
                       ropeTheta: config.ropeTheta,
                       rmsNormEps: config.rmsNormEps,
                       hiddenActivation: config.hiddenActivation,
                       standardize: config.standardize,
                       maxSoftTokens: config.maxSoftTokens,
                       weightDType: "bf16",
                       imageTokenID: config.imageTokenID,
                       boiTokenID: config.boiTokenID,
                       eoiTokenID: config.eoiTokenID,
                       weightsPath: "vision/vision_weights.bin",
                       tensorCount: tensorCount,
                       payloadBytes: payloadBytes,
                       sourceRepo: "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
                       sourceRevision: "f1e06dc520982d9b9edd76859fdb7ab209449949")
    }

    static func validate(
        mutate: ([(name: String, shape: [Int])]) -> [(name: String, shape: [Int])] = { $0 },
        declaredTensorCount: Int? = nil,
        declaredPayloadBytes: UInt64? = nil,
        adjust: (inout ResidentIndex) -> Void = { _ in }
    ) throws {
        var (index, payload, count) = makeIndex(mutate: mutate)
        adjust(&index)
        try VisionWeights.validateSchema(
            residentIndex: index,
            config: config,
            textHiddenSize: textHidden,
            declared: manifestVision(tensorCount: declaredTensorCount ?? count,
                                     payloadBytes: declaredPayloadBytes ?? payload))
    }

    // MARK: - The derivation itself

    /// The whole point of deriving rather than listing: the count and the byte
    /// total fall out of the config, and they are the numbers pinned against the
    /// real upstream repository in `PLAN_VISION.md` §1-1. If the runtime's
    /// derivation ever drifts, this is where it shows.
    @Test func inventoryMatchesThePinnedTowerShape() {
        let tensors = Self.inventory()
        #expect(tensors.count == 356)
        #expect(tensors.count == 2 + 13 * Self.config.numLayers + 2 + 1)

        let bytes = tensors.reduce(UInt64(0)) { total, entry in
            total + entry.shape.reduce(UInt64(1)) { $0 * UInt64($1) }
                * UInt64(MemoryLayout<UInt16>.size)
        }
        #expect(bytes == 1_145_588_832)
    }

    /// Two independent derivations of the same table — the installer's, written
    /// against the source repository, and the runtime's, written against
    /// `VisionConfig`. They exist separately so that a mistake in one is caught
    /// by the other; this asserts they currently agree, name for name and shape
    /// for shape.
    @Test func runtimeAndInstallerDeriveTheSameInventory() {
        let runtime = Self.inventory()
        let installer = VisionRepackPlanner.expectedInventory(
            config: VisionModelSource.pin.config, textHiddenSize: Self.textHidden)

        #expect(runtime.count == installer.count)
        let runtimeShapes = Dictionary(uniqueKeysWithValues:
            runtime.map { ($0.name, $0.shape.map(UInt64.init)) })
        let installerShapes = Dictionary(uniqueKeysWithValues:
            installer.map { ($0.name, $0.shape) })
        #expect(runtimeShapes == installerShapes)
    }

    @Test func acceptsAWellFormedTower() throws {
        try Self.validate()
    }

    // MARK: - Rejections

    @Test func rejectsAMissingTensor() {
        #expect(throws: ModelError.self) {
            try Self.validate(mutate: { $0.filter { $0.name != "vision_tower.std_bias" } })
        }
    }

    /// An extra tensor is as much a schema break as a missing one: the tower in
    /// the file is then not the tower this runtime knows how to run.
    @Test func rejectsAnExtraTensor() {
        #expect(throws: ModelError.self) {
            try Self.validate(mutate: { $0 + [(name: "vision_tower.surprise", shape: [1152])] })
        }
    }

    @Test func rejectsAReshapedTensor() {
        #expect(throws: ModelError.self) {
            try Self.validate(mutate: { tensors in
                tensors.map { entry in
                    entry.name == "vision_tower.encoder.layers.13.mlp.down_proj.linear.weight"
                        ? (name: entry.name, shape: [entry.shape[1], entry.shape[0]])
                        : entry
                }
            })
        }
    }

    /// The tower is BF16 throughout and carries no quantization companions.
    /// Both halves of that are load-bearing: the kernels read `bfloat` directly
    /// and never look for a scale.
    @Test func rejectsANonBF16Tensor() {
        #expect(throws: ModelError.self) {
            try Self.validate(adjust: { index in
                Self.replace(&index, "vision_tower.std_scale") { entry in
                    ResidentIndexEntry(name: entry.name,
                                       dtype: GTurboFormatV1.DType.u32.rawValue,
                                       fileOffset: entry.fileOffset,
                                       sizeBytes: entry.sizeBytes,
                                       shape: entry.shape,
                                       scaleOffset: 0, scaleSize: 0,
                                       biasOffset: 0, biasSize: 0)
                }
            })
        }
    }

    @Test func rejectsAQuantizationCompanion() {
        #expect(throws: ModelError.self) {
            try Self.validate(adjust: { index in
                Self.replace(&index, "vision_tower.std_scale") { entry in
                    ResidentIndexEntry(name: entry.name,
                                       dtype: entry.dtype,
                                       fileOffset: entry.fileOffset,
                                       sizeBytes: entry.sizeBytes,
                                       shape: entry.shape,
                                       scaleOffset: entry.fileOffset, scaleSize: 64,
                                       biasOffset: 0, biasSize: 0)
                }
            })
        }
    }

    /// The manifest's own claims are cross-checked against the schema, so a
    /// tower whose file is internally consistent but disagrees with what the
    /// installer recorded is still rejected.
    @Test func rejectsAManifestTensorCountDisagreement() {
        #expect(throws: ModelError.self) {
            try Self.validate(declaredTensorCount: 355)
        }
    }

    @Test func rejectsAManifestPayloadDisagreement() {
        #expect(throws: ModelError.self) {
            try Self.validate(declaredPayloadBytes: 1_145_588_830)
        }
    }

    /// A resident region that does not cover exactly the tensors the schema
    /// needs would leave the last tensor mapped past the end of the buffer.
    @Test func rejectsAShortResidentRegion() {
        #expect(throws: ModelError.self) {
            try Self.validate(adjust: { index in
                index = ResidentIndex(
                    header: ResidentIndexHeader(indexSize: index.header.indexSize,
                                                residentSize: index.header.residentSize - 2,
                                                entryCount: index.header.entryCount),
                    entries: index.entries)
            })
        }
    }

    private static func replace(_ index: inout ResidentIndex,
                                _ name: String,
                                _ transform: (ResidentIndexEntry) -> ResidentIndexEntry) {
        guard let entry = index.entries[name] else { return }
        var entries = index.entries
        entries[name] = transform(entry)
        index = ResidentIndex(header: index.header, entries: entries)
    }
}
