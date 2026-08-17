import Foundation
import Tokenizers

/// The multimodal marker tokens, resolved against the installed tokenizer.
///
/// Resolved rather than hard-coded so a tokenizer that does not carry them
/// fails at load with `missingSpecialToken` instead of writing plausible-looking
/// integers into the prompt.
public struct VisionMediaTokenIDs: Sendable, Equatable {
    /// `<|image>` 255999 — opens an image span.
    public let beginImage: Int32
    /// `<|image|>` 258880 — one soft token. Repeated `softTokenCount` times.
    public let image: Int32
    /// `<image|>` 258882 — closes an image span.
    public let endImage: Int32
    /// `<|audio|>` 258881 and `<|video|>` 258884. Resolved only so they can be
    /// rejected by name (PLAN_VISION §9); nothing consumes them.
    public let audio: Int32
    public let video: Int32

    public static let beginImageToken = "<|image>"
    public static let imageToken = "<|image|>"
    public static let endImageToken = "<image|>"
    public static let audioToken = "<|audio|>"
    public static let videoToken = "<|video|>"

    /// Every marker a user could type that this runtime must not accept as text.
    ///
    /// Ordered longest-first so scanning cannot match `<|image>` inside
    /// `<|image|>` and misreport which token was found.
    public static let mediaMarkerTokens = [
        imageToken, audioToken, videoToken, beginImageToken, endImageToken,
        "<|audio>", "<audio|>", "<|video>", "<video|>",
    ].sorted { $0.count > $1.count }

    public init(beginImage: Int32, image: Int32, endImage: Int32, audio: Int32, video: Int32) {
        self.beginImage = beginImage
        self.image = image
        self.endImage = endImage
        self.audio = audio
        self.video = video
    }

    public init(tokenizer: any Tokenizer) throws {
        self.beginImage = try GFTokenizer.requireTokenID(tokenizer, Self.beginImageToken)
        self.image = try GFTokenizer.requireTokenID(tokenizer, Self.imageToken)
        self.endImage = try GFTokenizer.requireTokenID(tokenizer, Self.endImageToken)
        self.audio = try GFTokenizer.requireTokenID(tokenizer, Self.audioToken)
        self.video = try GFTokenizer.requireTokenID(tokenizer, Self.videoToken)
    }

    public init(tokenizer: GFTokenizer) throws {
        try self.init(tokenizer: tokenizer.tokenizer)
    }
}

/// Where one image's soft tokens sit in the prompt.
///
/// `tokenOffset` points at the first `<|image|>`, *after* the `<|image>` opener.
/// That is the range the tower's output is scattered into (§4-5-c) and the range
/// that attends bidirectionally (§4-5-b) — upstream builds its
/// `mm_token_type_ids` from `image_ids = [image_token_id]` alone, so the
/// openers and closers are ordinary causal text on both counts.
public struct VisionImageSpan: Sendable, Equatable {
    public let imageIndex: Int
    public let tokenOffset: Int
    public let tokenCount: Int

    public var tokenEnd: Int { tokenOffset + tokenCount }

    public init(imageIndex: Int, tokenOffset: Int, tokenCount: Int) {
        self.imageIndex = imageIndex
        self.tokenOffset = tokenOffset
        self.tokenCount = tokenCount
    }
}

/// The image side of one prefill request: what to run the tower on, and where
/// the result goes.
///
/// `spans` are offsets into the token slice handed to `prefillChunked`, not
/// into the original prompt. The two coincide unless a prefix was served from a
/// cache — which is why a cached prefix and an image are refused together
/// (`PLAN_VISION.md` §4-6): the offsets would silently point at the wrong rows.
public struct VisionPrefillInput: Sendable {
    public let spans: [VisionImageSpan]
    public let images: [VisionPreprocessedImage]

    public init(spans: [VisionImageSpan], images: [VisionPreprocessedImage]) throws {
        guard spans.count == images.count else {
            throw VisionError.spanImageMismatch(
                "\(spans.count) image spans for \(images.count) images")
        }
        var previousEnd = 0
        for (index, span) in spans.enumerated() {
            let expected = images[index].geometry.softTokenCount
            guard span.tokenCount == expected else {
                throw VisionError.spanImageMismatch(
                    "image \(index) occupies \(span.tokenCount) prompt tokens but its "
                    + "geometry yields \(expected) soft tokens")
            }
            guard span.tokenOffset >= previousEnd else {
                throw VisionError.spanImageMismatch(
                    "image spans are out of order or overlap at image \(index)")
            }
            previousEnd = span.tokenEnd
        }
        self.spans = spans
        self.images = images
    }

    public var isEmpty: Bool { spans.isEmpty }
}

/// A prompt with its image spans resolved.
public struct VisionPrompt: Sendable {
    public let tokens: [Int32]
    public let spans: [VisionImageSpan]
    public let images: [VisionPreprocessedImage]

    public init(tokens: [Int32], spans: [VisionImageSpan], images: [VisionPreprocessedImage]) {
        self.tokens = tokens
        self.spans = spans
        self.images = images
    }
}

public enum VisionPromptAssembler {

    /// Reject media marker tokens written as literal text.
    ///
    /// Without this the ids flow through `prefill_embed_lookup_int4_block` as
    /// ordinary embeddings: `<|image|>` becomes whatever row 258880 of the
    /// embedding table happens to hold, no tower runs, and the model answers as
    /// though it saw something. The failure is invisible — the output is fluent,
    /// just wrong — which is why it has to be an error and not a warning
    /// (PLAN_VISION §4-5-d).
    ///
    /// Applied to user-supplied text only. The assembler inserts its own
    /// placeholders after this check has run.
    public static func rejectMediaMarkers(in text: String) throws {
        for marker in VisionMediaTokenIDs.mediaMarkerTokens where text.contains(marker) {
            throw VisionError.unattachedMediaToken(marker)
        }
    }

    /// Replace each `<|image|>` placeholder with `<|image>` + n soft tokens +
    /// `<image|>`, and report where the soft tokens landed.
    ///
    /// `softTokenCounts` is per image, in prompt order, and comes from
    /// `VisionImageGeometry.softTokenCount` — never from `maxSoftTokens`. The
    /// counts differ per image (§0-B-2), and a mismatch here desynchronises the
    /// scatter in §4-5-c from the tower's actual output length.
    public static func expandImagePlaceholders(
        tokens: [Int32],
        softTokenCounts: [Int],
        ids: VisionMediaTokenIDs
    ) throws -> (tokens: [Int32], spans: [VisionImageSpan]) {
        let placeholders = tokens.reduce(into: 0) { $0 += ($1 == ids.image ? 1 : 0) }
        guard placeholders == softTokenCounts.count else {
            throw GFTokenizerError.invalidChatTemplate(
                "prompt has \(placeholders) image placeholders but \(softTokenCounts.count) images")
        }

        let expandedTotal = softTokenCounts.reduce(0) { $0 + $1 + 2 }
        var out: [Int32] = []
        out.reserveCapacity(tokens.count - placeholders + expandedTotal)
        var spans: [VisionImageSpan] = []
        spans.reserveCapacity(softTokenCounts.count)

        var imageIndex = 0
        for token in tokens {
            guard token == ids.image else {
                out.append(token)
                continue
            }
            let count = softTokenCounts[imageIndex]
            guard count > 0 else {
                throw GFTokenizerError.invalidChatTemplate(
                    "image \(imageIndex) resolved to zero soft tokens")
            }
            out.append(ids.beginImage)
            spans.append(VisionImageSpan(imageIndex: imageIndex,
                                         tokenOffset: out.count,
                                         tokenCount: count))
            out.append(contentsOf: repeatElement(ids.image, count: count))
            out.append(ids.endImage)
            imageIndex += 1
        }
        return (out, spans)
    }

    /// Expand the placeholders for `images` and pair the result with the images
    /// themselves, which is everything prefill needs.
    ///
    /// The soft-token counts come from each image's own geometry, so the number
    /// of `<|image|>` tokens in the prompt and the number of rows the tower
    /// produces are the same number by construction rather than by agreement.
    public static func makePrefillPrompt(
        tokens: [Int32],
        images: [VisionPreprocessedImage],
        ids: VisionMediaTokenIDs
    ) throws -> (tokens: [Int32], vision: VisionPrefillInput) {
        let expanded = try expandImagePlaceholders(
            tokens: tokens,
            softTokenCounts: images.map(\.geometry.softTokenCount),
            ids: ids)
        return (expanded.tokens,
                try VisionPrefillInput(spans: expanded.spans, images: images))
    }
}
