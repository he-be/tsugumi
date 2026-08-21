import Foundation
import Testing
@testable import TurboFieldfareFormat

/// A manifest for a family whose layers are not all attention layers. The
/// fixture mirrors Qwen3.5-MoE's shape at 1/10th scale: 4 layers, of which 3
/// hold a recurrent state and 1 attends (`docs/qwen35moe/01-MODEL.md` §1).
private enum LinearFixture {
    static let zeroSHA = String(repeating: "0", count: 64)

    static let linear = GTurboManifestLinearAttentionV1(
        numKeyHeads: 2, numValueHeads: 4,
        keyHeadDim: 16, valueHeadDim: 16,
        convKernelDim: 4, layerCount: 3)

    static let kinds = ["linear_attention", "linear_attention",
                        "linear_attention", "full_attention"]

    static func arch(slidingWindow: Int = 0,
                     layerKinds: [String]? = kinds,
                     mask: [Int] = [0, 0, 0, 1],
                     linearAttention: GTurboManifestLinearAttentionV1? = linear,
                     family: String? = "qwen3_5_moe") -> GTurboManifestArchV1 {
        GTurboManifestArchV1(
            hiddenSize: 128, ffnIntermediate: 96, moeIntermediateSize: 64,
            numHeads: 2, numKVHeads: 2, numFullKVHeads: 2,
            headDim: 32, fullHeadDim: 32, vocabSize: 512,
            slidingWindow: slidingWindow, finalLogitSoftcap: 0,
            ropeTheta: 10_000_000, fullRopeTheta: 10_000_000,
            partialRotaryFactor: 0.25, numLayers: 4, numExperts: 2,
            topKExperts: 2, tieWordEmbeddings: false, attentionKEqV: false,
            hiddenActivation: "silu", fullAttentionLayerMask: mask,
            family: family, layerKinds: layerKinds, linearAttention: linearAttention)
    }

    static func manifest(arch: GTurboManifestArchV1 = arch(),
                         flagged: Bool = true,
                         minor: Int = GTurboFormatV1.versionMinorLinearAttention)
        -> GTurboManifestV1 {
        var flags = [
            "streamingPresent": true,
            "turboQuantKV": false,
            "aneSharedExpert": false,
        ]
        if flagged { flags["linearAttention"] = true }
        return GTurboManifestV1(
            versionMinor: minor,
            flags: flags,
            modelID: "fixture/qwen",
            sourceSnapshotHash: "snapshot",
            arch: arch,
            quant: nil,
            files: ["model_weights.bin": GTurboManifestFileV1(size: 16_384, sha256: zeroSHA)],
            expertsPerLayer: 2,
            numLayers: 4,
            expertStride: GTurboFormatV1.alignmentBytes,
            bitWidthOverridesHonored: 0)
    }
}

@Suite
struct GTurboLinearAttentionManifestTests {

    @Test func linearAttentionManifestRoundTrips() throws {
        let data = try GTurboManifestCodec.encode(LinearFixture.manifest())
        let decoded = try GTurboManifestCodec.decode(data)
        #expect(decoded.arch.family == "qwen3_5_moe")
        #expect(decoded.arch.layerKinds == LinearFixture.kinds)
        #expect(decoded.arch.linearAttention == LinearFixture.linear)
        #expect(decoded.flags["linearAttention"] == true)
        #expect(decoded.versionMinor >= GTurboFormatV1.versionMinorLinearAttention)
    }

    /// The three keys are written only by a family that needs them. A Gemma
    /// manifest has to stay the bytes it has always been.
    @Test func gemmaManifestOmitsTheNewArchKeys() throws {
        let arch = GTurboManifestArchV1(
            hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
            numHeads: 4, numKVHeads: 2, numFullKVHeads: 1,
            headDim: 16, fullHeadDim: 32, vocabSize: 1024,
            slidingWindow: 128, finalLogitSoftcap: 30,
            ropeTheta: 10_000, fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25, numLayers: 1, numExperts: 2,
            topKExperts: 1, tieWordEmbeddings: true, attentionKEqV: true,
            hiddenActivation: "gelu_pytorch_tanh", fullAttentionLayerMask: [0])
        let manifest = GTurboManifestV1(
            versionMinor: 0,
            flags: ["streamingPresent": true],
            modelID: "fixture/gemma",
            sourceSnapshotHash: "snapshot",
            arch: arch,
            quant: nil,
            files: ["model_weights.bin": GTurboManifestFileV1(
                size: 16_384, sha256: LinearFixture.zeroSHA)],
            expertsPerLayer: 2,
            numLayers: 1,
            expertStride: GTurboFormatV1.alignmentBytes,
            bitWidthOverridesHonored: 0)
        let json = String(decoding: try GTurboManifestCodec.encode(manifest), as: UTF8.self)
        #expect(!json.contains("\"family\""))
        #expect(!json.contains("\"layerKinds\""))
        #expect(!json.contains("\"linearAttention\""))
        let decoded = try GTurboManifestCodec.decode(Data(json.utf8))
        #expect(decoded.arch.family == nil)
        #expect(decoded.arch.layerKinds == nil)
        #expect(decoded.arch.linearAttention == nil)
    }

    /// A window of zero is only meaningful for a family with no sliding layer.
    @Test func attendingFamilyStillHasToNameAWindow() throws {
        let arch = LinearFixture.arch(layerKinds: nil, linearAttention: nil, family: nil)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.encode(LinearFixture.manifest(arch: arch, flagged: false))
        }
    }

    @Test func sectionAndFlagAreOneFactWrittenTwice() throws {
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.encode(LinearFixture.manifest(flagged: false))
        }
        let noSection = LinearFixture.arch(slidingWindow: 128, linearAttention: nil)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.encode(LinearFixture.manifest(arch: noSection))
        }
    }

    @Test func layerKindsMustAgreeWithTheCompatibilityMask() throws {
        let disagreeing = LinearFixture.arch(mask: [0, 0, 1, 1])
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.encode(LinearFixture.manifest(arch: disagreeing))
        }
    }

    @Test func layerCountMustMatchTheLayerKinds() throws {
        let miscounted = LinearFixture.arch(
            linearAttention: GTurboManifestLinearAttentionV1(
                numKeyHeads: 2, numValueHeads: 4, keyHeadDim: 16, valueHeadDim: 16,
                convKernelDim: 4, layerCount: 2))
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.encode(LinearFixture.manifest(arch: miscounted))
        }
    }

    @Test func unknownLayerKindIsRejected() throws {
        let strange = LinearFixture.arch(
            layerKinds: ["linear_attention", "linear_attention", "linear_attention", "quantum"],
            mask: [0, 0, 0, 0])
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.encode(LinearFixture.manifest(arch: strange))
        }
    }

    @Test func linearAttentionRequiresItsMinorVersion() throws {
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.encode(LinearFixture.manifest(minor: 2))
        }
    }

    /// A runtime that predates the section sees an unknown flag and refuses the
    /// model rather than reading its zeros as sliding windows.
    @Test func theFlagIsWhatAnOlderRuntimeTripsOn() {
        #expect(!GTurboFormatV1.knownFlags.isSubset(of: [
            "streamingPresent", "turboQuantKV", "aneSharedExpert", "visionTower", "mtpDraft",
        ]))
        #expect(GTurboFormatV1.knownFlags.contains("linearAttention"))
    }
}
