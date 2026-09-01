import Foundation
import Testing
@testable import TsugumiMacPresentation

@Suite struct ModelRevealPolicyTests {
    @Test func installedModelDirectoryIsSelected() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/checkout/scratch/gemma4.moepack",
            fileExists: { $0 == "/Users/dev/checkout/scratch/gemma4.moepack" })
        #expect(target == .selectItem(
            URL(fileURLWithPath: "/Users/dev/checkout/scratch/gemma4.moepack",
                isDirectory: true)))
    }

    @Test func missingModelFallsBackToItsContainer() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/checkout/scratch/gemma4.moepack",
            fileExists: { $0 == "/Users/dev/checkout/scratch" })
        #expect(target == .openContainer(
            URL(fileURLWithPath: "/Users/dev/checkout/scratch", isDirectory: true)))
    }

    @Test func missingContainerIsUnavailable() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/checkout/scratch/gemma4.moepack",
            fileExists: { _ in false })
        #expect(target == .unavailable)
    }

    @Test(arguments: ["", "   "])
    func blankPathIsUnavailable(path: String) {
        #expect(ModelRevealPolicy.target(forModelPath: path,
                                         fileExists: { _ in true }) == .unavailable)
    }

    @Test func pathIsStandardizedBeforeReveal() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/checkout/./scratch/../scratch/gemma4.moepack",
            fileExists: { $0 == "/Users/dev/checkout/scratch/gemma4.moepack" })
        #expect(target == .selectItem(
            URL(fileURLWithPath: "/Users/dev/checkout/scratch/gemma4.moepack",
                isDirectory: true)))
    }

    @Test func rootPathDoesNotLoopIntoItself() {
        #expect(ModelRevealPolicy.target(forModelPath: "/",
                                         fileExists: { _ in false }) == .unavailable)
    }
}
