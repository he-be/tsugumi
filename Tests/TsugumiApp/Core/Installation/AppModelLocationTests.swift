import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct AppModelLocationTests {
    @Test func explicitURLWins() {
        let result = AppModelLocation.resolve(
            explicitURL: URL(fileURLWithPath: "/models/explicit.moepack"),
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/models/explicit.moepack")
    }

    @Test func executableAncestorFindsPackageRootOutsideCWD() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/TsugumiApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/TsugumiMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4-qat-sym.moepack")
    }

    @Test func currentDirectoryCanBePackageRoot() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/TsugumiApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4-qat-sym.moepack")
    }

    @Test func aPreRenameInstallIsFoundUnderItsOldDirectoryName() {
        let files: Set<String> = [
            "/repo/Package.swift", "/repo/Sources/TsugumiApp/Mac",
            "/repo/scratch/gemma4-qat-sym.gturbo/manifest.json",
        ]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4-qat-sym.gturbo")
    }

    @Test func theCurrentNameWinsWhenBothDirectoriesAreInstalled() {
        let files: Set<String> = [
            "/repo/Package.swift", "/repo/Sources/TsugumiApp/Mac",
            "/repo/scratch/gemma4-qat-sym.moepack/manifest.json",
            "/repo/scratch/gemma4-qat-sym.gturbo/manifest.json",
        ]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4-qat-sym.moepack")
    }

    /// An empty leftover directory is not an install: answering with it would
    /// hide the working one sitting next to it under the other name.
    @Test func anEmptyCurrentDirectoryDoesNotHideALegacyInstall() {
        let files: Set<String> = [
            "/repo/Package.swift", "/repo/Sources/TsugumiApp/Mac",
            "/repo/scratch/gemma4-qat-sym.moepack",
            "/repo/scratch/gemma4-qat-sym.gturbo/manifest.json",
        ]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4-qat-sym.gturbo")
    }

    @Test func standaloneAppFallsBackToApplicationSupport() {
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/TsugumiMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/support/Tsugumi/gemma4-qat-sym.moepack")
    }
}
