import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct ReasoningLivePresentationTests {
    @Test func shortTextPassesThroughUntouched() {
        #expect(ReasoningLivePresentation.liveTail(of: "brief thought", cap: 20)
            == "brief thought")
    }

    @Test func textAtTheCapIsNotElided() {
        let text = String(repeating: "x", count: 10)
        #expect(ReasoningLivePresentation.liveTail(of: text, cap: 10) == text)
    }

    @Test func longTextKeepsOnlyTheNewestSlice() {
        let text = "old head that must go " + String(repeating: "y", count: 30)
        let tail = ReasoningLivePresentation.liveTail(of: text, cap: 30)
        #expect(tail == "…" + String(repeating: "y", count: 30))
    }

    @Test func multibyteTextIsCutOnCharacterBoundaries() {
        let text = String(repeating: "思", count: 40)
        let tail = ReasoningLivePresentation.liveTail(of: text, cap: 8)
        #expect(tail == "…" + String(repeating: "思", count: 8))
    }
}
