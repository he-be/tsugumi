import Foundation

/// Everything the installer needs to know about a pinned vision tower source.
/// The tower does not ship with the text checkpoint we install, so it is
/// fetched from its own repository and has its own identity, its own expected
/// inventory, and its own parity evidence (`PLAN_VISION.md` §1-1, §1-2).
public struct VisionSourcePin: Sendable, Equatable {
    /// One tensor that exists in both the text checkpoint and the vision
    /// repository, used to prove the two derive from the same weights.
    public struct ParityTensor: Sendable, Equatable {
        /// Name in the installed text checkpoint (MLX renames the prefixes).
        public let textName: String
        /// Name of the same tensor in the vision repository.
        public let visionRepoName: String
        /// SHA-256 both sides must produce.
        public let sha256: String

        public init(textName: String, visionRepoName: String, sha256: String) {
            self.textName = textName
            self.visionRepoName = visionRepoName
            self.sha256 = sha256
        }
    }

    public let repoID: String
    public let revision: String
    public let indexSha256Hex: String
    public let displayName: String
    /// Prefixes selecting the tower tensors inside the source shards.
    public let tensorPrefixes: [String]
    /// Stripped from every selected name before it is written to the index, so
    /// the installed model uses the project's `vision_tower.` / `embed_vision.`
    /// vocabulary rather than the upstream `model.` prefix.
    public let strippedNamePrefix: String
    public let expectedTensorCount: Int
    public let expectedPayloadBytes: UInt64
    public let parityTensors: [ParityTensor]
    public let config: VisionSourceConfig

    public init(repoID: String, revision: String, indexSha256Hex: String,
                displayName: String, tensorPrefixes: [String],
                strippedNamePrefix: String, expectedTensorCount: Int,
                expectedPayloadBytes: UInt64,
                parityTensors: [ParityTensor],
                config: VisionSourceConfig) {
        self.repoID = repoID
        self.revision = revision
        self.indexSha256Hex = indexSha256Hex
        self.displayName = displayName
        self.tensorPrefixes = tensorPrefixes
        self.strippedNamePrefix = strippedNamePrefix
        self.expectedTensorCount = expectedTensorCount
        self.expectedPayloadBytes = expectedPayloadBytes
        self.parityTensors = parityTensors
        self.config = config
    }
}

/// The tower geometry the installer expects to read back out of the source
/// repository's `config.json`. Pinning it means a silently reshaped upstream
/// upload fails the install instead of producing a model the runtime cannot
/// interpret.
public struct VisionSourceConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let positionEmbeddingSize: Int
    public let ropeTheta: Double
    public let rmsNormEps: Double
    public let hiddenActivation: String
    public let standardize: Bool
    public let maxSoftTokens: Int
    public let imageTokenID: Int
    public let boiTokenID: Int
    public let eoiTokenID: Int

    public init(hiddenSize: Int, numLayers: Int, numHeads: Int, numKVHeads: Int,
                headDim: Int, intermediateSize: Int, patchSize: Int,
                poolingKernelSize: Int, positionEmbeddingSize: Int,
                ropeTheta: Double, rmsNormEps: Double, hiddenActivation: String,
                standardize: Bool, maxSoftTokens: Int,
                imageTokenID: Int, boiTokenID: Int, eoiTokenID: Int) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.positionEmbeddingSize = positionEmbeddingSize
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.hiddenActivation = hiddenActivation
        self.standardize = standardize
        self.maxSoftTokens = maxSoftTokens
        self.imageTokenID = imageTokenID
        self.boiTokenID = boiTokenID
        self.eoiTokenID = eoiTokenID
    }
}

/// Google's unquantized QAT release. Its BF16 tensors are byte-identical to the
/// ones in the QAT-aligned text checkpoint this project installs, which is what
/// makes it legitimate to pair its tower with those text weights
/// (`PLAN_VISION.md` §1-2, verified again at install time by `parityTensors`).
public enum VisionModelSource {
    public static let pin = VisionSourcePin(
        repoID: "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
        revision: "f1e06dc520982d9b9edd76859fdb7ab209449949",
        indexSha256Hex:
            "907826a6e46ff454272bd6db1fee629d5531a2303be22986d825a0871d7dc7a7",
        displayName: "Gemma 4 26B-A4B IT QAT q4_0 (unquantized, vision tower)",
        tensorPrefixes: ["model.vision_tower.", "model.embed_vision."],
        strippedNamePrefix: "model.",
        expectedTensorCount: 356,
        expectedPayloadBytes: 1_145_588_832,
        parityTensors: [
            VisionSourcePin.ParityTensor(
                textName: "language_model.model.norm.weight",
                visionRepoName: "model.language_model.norm.weight",
                sha256: "134bc0ecd2a53f9871eb3a62e0fab0221ee8324a276462073766793013e95f43"),
            VisionSourcePin.ParityTensor(
                textName: "language_model.model.layers.0.input_layernorm.weight",
                visionRepoName: "model.language_model.layers.0.input_layernorm.weight",
                sha256: "978017393fdf7a414eccc7d91844b1312101411b0205c5073379c3f5eeaefa77"),
            VisionSourcePin.ParityTensor(
                textName: "language_model.model.layers.7.post_feedforward_layernorm.weight",
                visionRepoName: "model.language_model.layers.7.post_feedforward_layernorm.weight",
                sha256: "5889b58a3573b37a190d156a94f67cbdff1dc3ddd27d792a5b9e9d9671455a10"),
        ],
        config: VisionSourceConfig(
            hiddenSize: 1152,
            numLayers: 27,
            numHeads: 16,
            numKVHeads: 16,
            headDim: 72,
            intermediateSize: 4304,
            patchSize: 16,
            poolingKernelSize: 3,
            positionEmbeddingSize: 10240,
            ropeTheta: 100.0,
            rmsNormEps: 1e-6,
            hiddenActivation: "gelu_pytorch_tanh",
            standardize: true,
            maxSoftTokens: 280,
            imageTokenID: 258880,
            boiTokenID: 255999,
            eoiTokenID: 258882))

    /// Text checkpoint the tower may be paired with. Any other checkpoint would
    /// fail the parity check anyway; naming it here makes the refusal specific.
    public static let requiredTextRepoID = QATAlignedModelSource.repoID
}
