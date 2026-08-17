import Foundation

/// Errors raised while turning an image into soft tokens.
public enum VisionError: Error, Equatable, CustomStringConvertible {
    case unsupportedSoftTokenBudget(Int, supported: [Int])
    case degenerateImage(width: Int, height: Int)
    case patchBudgetExceeded(width: Int, height: Int, patches: Int, budget: Int)
    case imageDecodeFailed(path: String, detail: String)
    case unattachedMediaToken(String)
    case unsupportedMedia(String)
    case towerNotInstalled(path: String)

    public var description: String {
        switch self {
        case let .unsupportedSoftTokenBudget(value, supported):
            return "unsupported image token budget \(value); supported: \(supported)"
        case let .degenerateImage(width, height):
            return "image \(width)x\(height) resizes to a zero-area grid"
        case let .patchBudgetExceeded(width, height, patches, budget):
            return "image \(width)x\(height) produced \(patches) patches, over the \(budget) budget"
        case let .imageDecodeFailed(path, detail):
            return "could not decode image at \(path): \(detail)"
        case let .unattachedMediaToken(token):
            return """
                prompt contains the special token \(token) but no matching media was attached; \
                pass the image with --image (CLI) or an image_url content part (server) \
                instead of writing the token into the text
                """
        case let .unsupportedMedia(kind):
            return "\(kind) input is not supported by this runtime"
        case let .towerNotInstalled(path):
            return """
                the model at \(path) was installed without a vision tower; \
                add one with `TurboFieldfareRepack --add-vision --input-gturbo \(path)`
                """
        }
    }
}

/// How an image is laid out once it has been resized onto the patch grid.
///
/// Everything downstream is derived from this: the resize target, the patch
/// count, the position of each patch, and — the number that leaks furthest —
/// how many soft tokens the image occupies in the prompt.
///
/// That last number is **not** `maxSoftTokens`. Upstream rounds each side down
/// to a multiple of `poolingKernelSize * patchSize`, so a budget of 280 yields
/// 256 soft tokens for a square image and 266 for a 4:3 one. Writing 280 into
/// the token stream would desynchronise the prompt from the tower's output
/// (PLAN_VISION §2-1, measured in §0-B-2).
public struct VisionImageGeometry: Sendable, Equatable {
    /// Resize target, in pixels. Both sides are multiples of `sideMultiple`.
    public let targetWidth: Int
    public let targetHeight: Int
    /// Patch grid. `patchesWide * patchesHigh == patchCount`.
    public let patchesWide: Int
    public let patchesHigh: Int
    /// Soft tokens this image contributes to the prompt: `patchCount / k^2`.
    public let softTokenCount: Int

    public var patchCount: Int { patchesWide * patchesHigh }

    /// Patch `i` sits at `(i % patchesWide, i / patchesWide)`.
    ///
    /// Upstream keeps an explicit `[num_patches, 2]` position table because it
    /// pads batches to a common length with `(-1, -1)` rows. We process one
    /// image at a time (PLAN_VISION §4-4), so there is no padding and the
    /// positions are recoverable by division — no table is stored or uploaded.
    public func position(ofPatch index: Int) -> (x: Int, y: Int) {
        precondition(index >= 0 && index < patchCount, "patch index out of range")
        return (index % patchesWide, index / patchesWide)
    }

    /// Index of the pooled soft token that patch `index` averages into.
    ///
    /// Upstream: `kernel_idxs = x/k + (max_x/k) * (y/k)` where `max_x` is the
    /// grid width. Row-major over the *pooled* grid, not the patch grid.
    public func pooledIndex(ofPatch index: Int, poolingKernelSize k: Int) -> Int {
        let (x, y) = position(ofPatch: index)
        return (x / k) + (patchesWide / k) * (y / k)
    }
}

/// The fixed side of the vision preprocessor: patch size, pooling, and the
/// soft-token budgets the checkpoint's processor config allows.
public struct VisionPreprocessorConfig: Sendable, Equatable {
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let maxSoftTokens: Int

    /// Upstream `_SUPPORTED_SOFT_TOKENS`. We expose the first three
    /// (PLAN_VISION §4-6); 560 and 1120 are listed so the error message tells
    /// the truth about what the upstream processor accepts.
    public static let upstreamSupportedSoftTokens = [70, 140, 280, 560, 1120]
    /// Budgets this runtime accepts on the command line.
    public static let supportedSoftTokens = [70, 140, 280]

    public init(patchSize: Int = 16,
                poolingKernelSize: Int = 3,
                maxSoftTokens: Int = 280) throws {
        guard Self.upstreamSupportedSoftTokens.contains(maxSoftTokens) else {
            throw VisionError.unsupportedSoftTokenBudget(
                maxSoftTokens, supported: Self.upstreamSupportedSoftTokens)
        }
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.maxSoftTokens = maxSoftTokens
    }

    /// `max_patches` in upstream terms: the patch budget the resize must fit.
    public var maxPatches: Int { maxSoftTokens * poolingKernelSize * poolingKernelSize }

    /// Both target sides are rounded down to a multiple of this.
    public var sideMultiple: Int { poolingKernelSize * patchSize }

    /// Port of `get_aspect_ratio_preserving_size` + the patchify step that
    /// follows it (`image_processing_pil_gemma4.py`), including its two
    /// degenerate-side rescues.
    ///
    /// The float arithmetic is done in `Double` to match Python's `math.sqrt` /
    /// `math.floor` on the same inputs. `Float` would put the ideal side within
    /// a rounding step of a `sideMultiple` boundary for some aspect ratios and
    /// silently pick a different grid than the reference.
    public func geometry(imageWidth width: Int, imageHeight height: Int) throws -> VisionImageGeometry {
        guard width > 0, height > 0 else {
            throw VisionError.degenerateImage(width: width, height: height)
        }

        let targetPixels = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPixels / Double(height * width)).squareRoot()
        let side = Double(sideMultiple)

        var targetHeight = Int((factor * Double(height) / side).rounded(.down)) * sideMultiple
        var targetWidth = Int((factor * Double(width) / side).rounded(.down)) * sideMultiple

        if targetHeight == 0 && targetWidth == 0 {
            throw VisionError.degenerateImage(width: width, height: height)
        }

        // Extreme aspect ratios round one side to zero. Upstream pins that side
        // to one grid cell and derives the other from the *integer* side ratio
        // — not from `factor`. Mirrored exactly, oddity included, because the
        // fixtures are generated by that code.
        let maxSideLength = (maxPatches / (poolingKernelSize * poolingKernelSize)) * sideMultiple
        if targetHeight == 0 {
            targetHeight = sideMultiple
            targetWidth = min((width / height) * sideMultiple, maxSideLength)
        } else if targetWidth == 0 {
            targetWidth = sideMultiple
            targetHeight = min((height / width) * sideMultiple, maxSideLength)
        }

        let patchesWide = targetWidth / patchSize
        let patchesHigh = targetHeight / patchSize
        let patchCount = patchesWide * patchesHigh
        guard patchCount > 0, patchCount <= maxPatches else {
            throw VisionError.patchBudgetExceeded(width: width, height: height,
                                                  patches: patchCount, budget: maxPatches)
        }

        return VisionImageGeometry(
            targetWidth: targetWidth,
            targetHeight: targetHeight,
            patchesWide: patchesWide,
            patchesHigh: patchesHigh,
            softTokenCount: patchCount / (poolingKernelSize * poolingKernelSize))
    }
}
