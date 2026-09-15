import Foundation
import Testing
@testable import Tsugumi
@testable import TsugumiServerCore

/// S-2 (`docs/qwen38/26` §5): a tool call cut by the end of the context is
/// reported as a context overflow, not as a malformed call.
@Suite("S-2 generation reached the context")
struct GenerationReachedContextTests {
    /// The two records of `docs/qwen38/21` §9 (12K, MTP on): prompt + generated
    /// + the verify row came to exactly the context.
    @Test func the12KRecordsAreOverflows() throws {
        for (prompt, generated) in [(12_199, 88), (12_286, 1)] {
            let error = try #require(ServerRequestError.generationReachedContext(
                stop: .maxTokens, promptTokens: prompt, generatedTokens: generated,
                reserved: 1, maxContext: 12_288))
            #expect(error.type == .exceedContextSize)
            #expect(error.code == "context_length_exceeded")
            #expect(error.message.contains("\(prompt)"))
        }
    }

    /// The request's own `max_tokens`, an end of turn, or a context with room
    /// left keep the decoder's failure.
    @Test func otherStopsStayTheDecodersFailure() {
        #expect(ServerRequestError.generationReachedContext(
            stop: .maxTokens, promptTokens: 12_199, generatedTokens: 87,
            reserved: 1, maxContext: 12_288) == nil)
        #expect(ServerRequestError.generationReachedContext(
            stop: .endOfTurn, promptTokens: 12_199, generatedTokens: 88,
            reserved: 1, maxContext: 12_288) == nil)
        #expect(ServerRequestError.generationReachedContext(
            stop: .maxTokens, promptTokens: 1_000, generatedTokens: 100,
            maxContext: 12_288) == nil)
        // Gemma reserves nothing: the sum alone reaches it.
        #expect(ServerRequestError.generationReachedContext(
            stop: .maxTokens, promptTokens: 12_200, generatedTokens: 88,
            maxContext: 12_288) != nil)
    }
}
