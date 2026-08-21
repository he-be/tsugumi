import Foundation
import Metal

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
    }

    /// Canonical Gemma 4 26B-A4B baseline, checked against the installed
    /// model manifest.
    /// `intermediateSize = 2112` is the shared-expert FFN width (3 × moe).
    public static let gemma4_26B_A4B = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4LayerMask(),
        hiddenActivation: "gelu_pytorch_tanh"
    )

    private static func gemma4LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 30)
        for i in stride(from: 5, to: 30, by: 6) { mask[i] = 1 }
        return mask
    }

    /// Ornith-1.5-35B-A3B (`qwen3_5_moe`), the second architecture this runtime
    /// carries (`docs/qwen35moe/01-MODEL.md`). Three fields read oddly against
    /// the Gemma baseline and are correct:
    ///
    /// - `slidingWindow = 0` — there is no sliding layer at all. The 30 zeros in
    ///   `fullAttentionLayerMask` are recurrent layers, not windowed ones, which
    ///   is why the manifest also carries `layerKinds`.
    /// - `numKVHeads` / `headDim` restate the full-attention values. The
    ///   recurrent layers have no K/V of that kind; their geometry lives in
    ///   `manifest.arch.linearAttention`.
    /// - `intermediateSize = moeIntermediateSize = 512` — upstream gives the
    ///   shared expert and one routed expert the same width.
    public static let ornith1_5_35B_A3B = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,
        moeIntermediateSize: 512,
        numHeads: 16,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 40,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.ornithLayerMask(),
        hiddenActivation: "silu"
    )

    /// Every fourth layer attends; the other three keep a recurrent state.
    private static func ornithLayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { mask[i] = 1 }
        return mask
    }
}

/// Compile-time vision tower baseline, checked field-by-field against
/// `manifest.json -> vision` when the installed model carries a tower. Unlike
/// `ArchConfig` there is no per-checkpoint variation to accommodate: the tower
/// comes from one pinned upstream repository (`PLAN_VISION.md` §1-1).
public struct VisionConfig: Sendable, Equatable {
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

    /// Gemma 4 vision tower. `maxSoftTokens` is an upper bound: the token count
    /// of one image follows its aspect ratio (`PLAN_VISION.md` §2-1).
    public static let gemma4Vision = VisionConfig(
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
        eoiTokenID: 258882
    )
}

/// Failure modes for the validation gates in `Model.load`.
enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAGTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.gturbo directory at \(p) is missing manifest.json"
        case .notAGTurboDirectory:
            return "manifest.json magic does not equal \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.gturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
