import CryptoKit
import Foundation
import TurboFieldfare

/// The lineage a cached KV belongs to.
///
/// Not a fact about the conversation — CACHE-1 forbids the cache from
/// consulting those — but about what the token ids *mean*: a different model,
/// runtime, or template makes the same numbers a different prefix, and the KV
/// rows behind them describe something else.
struct ServerPromptCacheDomain: Sendable, Equatable {
    let modelID: String
    let sourceSnapshotHash: String?
    let runtimeProfileHash: String
    let maximumContext: Int
    let kvStorage: String
    let fp16RingEnabled: Bool
    let templateSHA256: String
}

/// One picture inside a token sequence: where its soft tokens sit, how many
/// there are, and which photograph they came from.
///
/// CACHE-4: this is exactly the pair the reference compares inside the walk —
/// `mtmd_input_chunk_get_id` and `mtmd_input_chunk_get_n_tokens`
/// (`server-common.cpp:678`) — because the ids in the sequence say nothing
/// about which picture widened into them.
struct ServerPromptMediaChunk: Sendable, Equatable {
    /// The first soft token, after the `<|image>` opener — `VisionImageSpan`'s
    /// offset, in the same coordinates as the token array it describes.
    let tokenOffset: Int
    let tokenCount: Int
    let digest: String

    var tokenEnd: Int { tokenOffset + tokenCount }
}

/// What one served completion left behind.
struct ServerPromptCacheEntry: Sendable, Equatable {
    let domain: ServerPromptCacheDomain
    /// The tokens the KV holds, in order — prompt and generation both. This is
    /// the entire state the reuse rule reads.
    let tokenIDs: [Int32]
    /// One digest per picture inside `tokenIDs`, in prompt order.
    ///
    /// Two photographs widen into the *same* soft-token ids, so the token walk
    /// cannot tell them apart. Until the walk compares media chunks the way the
    /// reference does (CACHE-4), this is what stops a request from resuming
    /// from a prefix that holds a different picture's rows and answering about
    /// it.
    let imageDigests: [String]

    var kvPosition: Int { tokenIDs.count }
}

enum ServerPromptCacheMatch: Sendable, Equatable {
    case miss
    /// `vision` is the image side of the tokens still to be prefilled — the
    /// pictures the served prefix does not already hold, with offsets into the
    /// remaining slice rather than into the whole prompt. nil when every
    /// picture is already in the KV, which includes every text-only session.
    case hit(effectivePromptIDs: [Int32],
             cachedPromptTokens: Int,
             vision: VisionPrefillInput? = nil)
}

/// CACHE-1: how much of two token sequences is the same from the start.
///
/// This is the whole reuse rule. The reference implementation spends one line
/// on it too (`server-context.cpp:3125`, `get_common_prefix`), and the reason
/// it can is that a prefix of identical tokens is exactly what a KV cache is
/// valid for — no fact about roles, tools, thinking, or pictures adds anything
/// to that, and every such fact that was consulted here before could only make
/// the answer smaller than the truth.
func commonPrefixLength(_ lhs: [Int32], _ rhs: [Int32]) -> Int {
    var index = 0
    let limit = min(lhs.count, rhs.count)
    while index < limit, lhs[index] == rhs[index] { index += 1 }
    return index
}

/// CACHE-4: the same walk, with the pictures compared as chunks inside it.
///
/// **Not implemented yet (P1-D4).** The media arguments are accepted so the
/// SPEC line can be stated as a test, but the walk still sees tokens only —
/// which is why two photographs behind the same words still look identical to
/// it. Until this compares chunks, `ServerPromptCache` keeps the separate
/// digest check that CACHE-4 calls the interim arrangement.
func commonPrefixLength(_ lhs: [Int32], _ rhs: [Int32],
                        lhsMedia: [ServerPromptMediaChunk],
                        rhsMedia: [ServerPromptMediaChunk]) -> Int {
    commonPrefixLength(lhs, rhs)
}

struct ServerPromptCache: Sendable {
    private(set) var entry: ServerPromptCacheEntry?

    mutating func invalidate() {
        entry = nil
    }

    /// The per-image digests of a request, in prompt order.
    static func imageDigests(_ request: ValidatedChatRequest) -> [String] {
        (request.vision?.images ?? []).map { attachment in
            SHA256.hash(data: attachment.data)
                .map { String(format: "%02x", $0) }
                .joined()
        }
    }

    /// Record what the KV now holds.
    ///
    /// There is nothing to be careful about beyond knowing the tokens exactly:
    /// under CACHE-1 a KV that holds text the client will never send back is
    /// not dangerous, it is merely a short common prefix next time. The stop
    /// reason, the stop-string filtering, and the shape of the turn — all of
    /// which the old design screened on here — cannot make the answer wrong.
    /// - Parameter vision: the image side of the prompt this KV was built
    ///   from, with offsets into the whole prompt — which is a prefix of
    ///   `result.kvBackedTokenIDs`, so the offsets describe the KV too.
    ///   Unused until the walk compares chunks (P1-D4, CACHE-4).
    mutating func publish(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        result: RawDecodeResult,
        vision: VisionPrefillInput? = nil
    ) {
        guard result.kvPosition == result.kvBackedTokenIDs.count,
              result.kvPosition > 0 else {
            entry = nil
            return
        }
        entry = ServerPromptCacheEntry(
            domain: domain,
            tokenIDs: result.kvBackedTokenIDs,
            imageDigests: Self.imageDigests(request))
    }

    /// How much of `renderedPromptIDs` this KV can serve.
    ///
    /// - Parameter maximumRewind: how far the KV cursor may be moved back and
    ///   still describe rows that are all still present (CACHE-2). Under FP16
    ///   ring storage that is finite; the runner computes it.
    /// - Parameter vision: the image side of the freshly rendered prompt, whose
    ///   spans are offsets into the whole prompt.
    func match(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        vision: VisionPrefillInput? = nil,
        maximumRewind: Int = 0
    ) -> ServerPromptCacheMatch {
        guard let entry, entry.domain == domain, entry.kvPosition > 0 else {
            return .miss
        }

        var reusable = commonPrefixLength(entry.tokenIDs, renderedPromptIDs)

        // CACHE-3: with nothing left to prefill there is no token to draw the
        // next logits from, so the last one is decoded again (`n_past--`).
        if reusable == renderedPromptIDs.count { reusable -= 1 }

        // A picture cannot be half served: its soft tokens are scattered in one
        // go. Cutting back to where it starts keeps everything in front of it.
        if let straddling = (vision?.spans ?? []).first(where: {
            $0.tokenOffset < reusable && $0.tokenEnd > reusable
        }) {
            reusable = straddling.tokenOffset
        }

        guard reusable > 0 else { return .miss }
        // CACHE-2 is bounded by how far back the cursor may go (SPEC §12
        // DEV-13). Serving the whole entry never moves it and is always fine.
        guard entry.kvPosition - reusable <= maximumRewind else { return .miss }

        switch servedVision(request: request,
                            vision: vision,
                            reusable: reusable,
                            entry: entry) {
        case .unusable:
            return .miss
        case .allInTheKV:
            return .hit(effectivePromptIDs: renderedPromptIDs,
                        cachedPromptTokens: reusable)
        case .remaining(let input):
            return .hit(effectivePromptIDs: renderedPromptIDs,
                        cachedPromptTokens: reusable,
                        vision: input)
        }
    }

    private enum ServedVision {
        /// The prefix cannot be served: it holds a picture this request does
        /// not have, or the remainder cannot be described.
        case unusable
        /// Every picture is already in the KV — the text-only case too.
        case allInTheKV
        /// The pictures the remaining slice adds, offsets already rebased.
        case remaining(VisionPrefillInput)
    }

    /// The image side of a hit: what is already in the KV has to be the same
    /// pictures, and what is not has to be handed on with its offsets rebased
    /// onto the slice that will actually be prefilled.
    ///
    private func servedVision(
        request: ValidatedChatRequest,
        vision: VisionPrefillInput?,
        reusable: Int,
        entry: ServerPromptCacheEntry
    ) -> ServedVision {
        guard let vision, !vision.spans.isEmpty else { return .allInTheKV }
        let digests = Self.imageDigests(request)

        // The tokens agree on the served prefix, so both sides hold the same
        // number of pictures inside it — which makes them the first `n` of each
        // list, and comparable one to one.
        let inTheKV = vision.spans.filter { $0.tokenEnd <= reusable }.count
        guard digests.count >= inTheKV,
              entry.imageDigests.count >= inTheKV,
              digests.prefix(inTheKV)
                .elementsEqual(entry.imageDigests.prefix(inTheKV)) else {
            return .unusable
        }

        let remaining = zip(vision.spans, vision.images).filter { $0.0.tokenOffset >= reusable }
        guard !remaining.isEmpty else { return .allInTheKV }
        let rebased = remaining.map { span, _ in
            VisionImageSpan(imageIndex: span.imageIndex,
                            tokenOffset: span.tokenOffset - reusable,
                            tokenCount: span.tokenCount)
        }
        guard let input = try? VisionPrefillInput(spans: rebased,
                                                  images: remaining.map(\.1)) else {
            return .unusable
        }
        return .remaining(input)
    }
}
