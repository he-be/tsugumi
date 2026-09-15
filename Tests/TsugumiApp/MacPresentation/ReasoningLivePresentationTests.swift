import Testing
@testable import TsugumiMacPresentation

@Suite struct ReasoningLivePresentationTests {
    @Test func sameTextIsUnchanged() {
        #expect(ReasoningLivePresentation.edit(from: "thought", to: "thought") == .unchanged)
        #expect(ReasoningLivePresentation.edit(from: "", to: "") == .unchanged)
    }

    @Test func aGrowingTextAppendsOnlyWhatIsNew() {
        #expect(ReasoningLivePresentation.edit(from: "", to: "The user") == .append("The user"))
        #expect(ReasoningLivePresentation.edit(from: "The user", to: "The user wants")
            == .append(" wants"))
    }

    /// A delta can end in the middle of a grapheme (a combining mark or a
    /// joiner arrives in the next token); the split is by scalars so the
    /// appended piece is exactly what was added.
    @Test func multibyteAndSplitGraphemesAppendByScalars() {
        #expect(ReasoningLivePresentation.edit(from: "寿司ネタ", to: "寿司ネタの名前")
            == .append("の名前"))
        #expect(ReasoningLivePresentation.edit(from: "e", to: "e\u{301}x") == .append("\u{301}x"))
    }

    @Test func anythingElseReplaces() {
        #expect(ReasoningLivePresentation.edit(from: "old turn", to: "new") == .replace("new"))
        #expect(ReasoningLivePresentation.edit(from: "thought", to: "") == .replace(""))
    }
}
