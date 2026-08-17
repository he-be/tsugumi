import Foundation

/// Everything the installer needs to know about a pinned MTP drafter source.
/// The drafter is a separate checkpoint from the text weights — a different
/// repository, a different quantization, its own revision — so it carries its
/// own identity, inventory and provenance evidence (`docs/mtp/01-CHECKPOINT.md`).
public struct DraftSourcePin: Sendable, Equatable {
    /// A tensor the drafter's MLX conversion left unquantized, whose digest was
    /// measured in Google's BF16 assistant release. Matching it proves the
    /// checkpoint being installed is that release converted, not a lookalike.
    public struct ProvenanceTensor: Sendable, Equatable {
        /// Name in the drafter repository (upstream prefix included).
        public let repoName: String
        /// SHA-256 both the BF16 release and this conversion produce.
        public let sha256: String

        public init(repoName: String, sha256: String) {
            self.repoName = repoName
            self.sha256 = sha256
        }
    }

    public let repoID: String
    public let revision: String
    public let indexSha256Hex: String
    public let displayName: String
    /// Stripped from every tensor name before it is written to the index, so the
    /// installed drafter uses `layers.0.…` rather than the upstream `model.`
    /// prefix. Names that lack it (`pre_projection.weight`) are kept as they are.
    public let strippedNamePrefix: String
    /// Entries in `draft/draft_weights.bin`, i.e. tensors after each quantized
    /// weight has absorbed its `.scales`/`.biases` companions.
    public let expectedTensorCount: Int
    public let expectedPayloadBytes: UInt64
    public let provenanceTensors: [ProvenanceTensor]
    public let config: DraftSourceConfig

    public init(repoID: String, revision: String, indexSha256Hex: String,
                displayName: String, strippedNamePrefix: String,
                expectedTensorCount: Int, expectedPayloadBytes: UInt64,
                provenanceTensors: [ProvenanceTensor],
                config: DraftSourceConfig) {
        self.repoID = repoID
        self.revision = revision
        self.indexSha256Hex = indexSha256Hex
        self.displayName = displayName
        self.strippedNamePrefix = strippedNamePrefix
        self.expectedTensorCount = expectedTensorCount
        self.expectedPayloadBytes = expectedPayloadBytes
        self.provenanceTensors = provenanceTensors
        self.config = config
    }
}

/// The drafter geometry the installer expects to read back out of the source
/// repository's `config.json`. Pinning it means a silently reshaped or
/// re-quantized upstream upload fails the install rather than producing weights
/// the runtime would misread.
public struct DraftSourceConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let intermediateSize: Int
    public let backboneHiddenSize: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let rmsNormEps: Double
    public let hiddenActivation: String
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    /// One entry per drafter layer: 1 = full attention, 0 = sliding.
    public let fullAttentionLayerMask: [Int]
    public let quantBits: Int
    public let quantGroupSize: Int
    public let quantMode: String

    public init(hiddenSize: Int, numLayers: Int, numHeads: Int, numKVHeads: Int,
                numFullKVHeads: Int, headDim: Int, fullHeadDim: Int,
                intermediateSize: Int, backboneHiddenSize: Int, vocabSize: Int,
                slidingWindow: Int, ropeTheta: Double, fullRopeTheta: Double,
                partialRotaryFactor: Double, rmsNormEps: Double,
                hiddenActivation: String, tieWordEmbeddings: Bool,
                attentionKEqV: Bool, fullAttentionLayerMask: [Int],
                quantBits: Int, quantGroupSize: Int, quantMode: String) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.intermediateSize = intermediateSize
        self.backboneHiddenSize = backboneHiddenSize
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.rmsNormEps = rmsNormEps
        self.hiddenActivation = hiddenActivation
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.quantBits = quantBits
        self.quantGroupSize = quantGroupSize
        self.quantMode = quantMode
    }

    /// Head dimension of one drafter layer. The last layer is the full-attention
    /// one and uses the wider head, matching the target's full layers.
    public func headDim(forLayer layer: Int) -> Int {
        fullAttentionLayerMask[layer] == 1 ? fullHeadDim : headDim
    }
}

/// The MLX conversion of Google's QAT assistant. Its unquantized BF16 tensors
/// are byte-identical to the ones in `google/…-qat-q4_0-unquantized-assistant`
/// (`docs/mtp/01-CHECKPOINT.md` §1), which is what makes it legitimate to pair
/// with the QAT text checkpoint this project installs — Google's model card
/// requires the assistant to be a QAT checkpoint of the same precision.
public enum DraftModelSource {
    public static let pin = DraftSourcePin(
        repoID: "mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit",
        revision: "bb94eae1b70a80dac16cbf959bb4b7d56bd1fb8c",
        indexSha256Hex:
            "ab54b0e481714d358d800ad10366f585841e678f982be3274ea6660e9bedd3eb",
        displayName: "Gemma 4 26B-A4B IT QAT assistant (MTP drafter, 4-bit)",
        strippedNamePrefix: "model.",
        expectedTensorCount: 48,
        expectedPayloadBytes: 236_114_440,
        provenanceTensors: [
            DraftSourcePin.ProvenanceTensor(
                repoName: "model.norm.weight",
                sha256: "3bf68317e6d4e33e29a3d019eb744d52d5fb3ebf5dca52513e878e1f845f9047"),
            DraftSourcePin.ProvenanceTensor(
                repoName: "model.layers.0.input_layernorm.weight",
                sha256: "fbd6be5ad58d336c6bd398bd161db25bcc13bf2e07e50db8c6b55d8f400959eb"),
            DraftSourcePin.ProvenanceTensor(
                repoName: "model.layers.3.self_attn.q_norm.weight",
                sha256: "07101aaa0ac6e6df4b28cc3b209f04edaa9c277fc5037b50525478e3e60df15b"),
        ],
        config: DraftSourceConfig(
            hiddenSize: 1024,
            numLayers: 4,
            numHeads: 16,
            numKVHeads: 8,
            numFullKVHeads: 2,
            headDim: 256,
            fullHeadDim: 512,
            intermediateSize: 8192,
            backboneHiddenSize: 2816,
            vocabSize: 262144,
            slidingWindow: 1024,
            ropeTheta: 10000.0,
            fullRopeTheta: 1000000.0,
            partialRotaryFactor: 0.25,
            rmsNormEps: 1e-6,
            hiddenActivation: "gelu_pytorch_tanh",
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 0, 0, 1],
            quantBits: 4,
            quantGroupSize: 64,
            quantMode: "affine"))

    /// Text checkpoint the drafter may be paired with. Google trained it against
    /// the QAT target, and naming the requirement here makes the refusal
    /// specific instead of leaving it to a bad acceptance rate to discover.
    public static let requiredTextRepoID = QATAlignedModelSource.repoID
}
