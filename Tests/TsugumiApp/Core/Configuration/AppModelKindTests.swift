import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct AppModelKindTests {
    @Test func capabilitiesMatchTheShippedCheckpoints() {
        #expect(AppModelKind.gemmaQATSym.supportsVision)
        #expect(!AppModelKind.ornith.supportsVision)
        // MTP exists on both; the block width is each family's own.
        #expect(AppModelKind.gemmaQATSym.draftBlockSize == 4)
        #expect(AppModelKind.ornith.draftBlockSize == 2)
        // Thinking defaults: Gemma off, Ornith on.
        #expect(!AppModelKind.gemmaQATSym.thinkingDefault)
        #expect(AppModelKind.ornith.thinkingDefault)
        // S1 pins Ornith's sampler to the official values.
        #expect(!AppModelKind.gemmaQATSym.samplingIsLocked)
        #expect(AppModelKind.ornith.samplingIsLocked)
        #expect(AppModelKind.ornith.officialTemperature == 0.6)
        #expect(AppModelKind.ornith.officialTopK == 20)
        #expect(AppModelKind.ornith.officialTopP == 0.95)
        #expect(AppModelKind.gemmaQATSym.officialTemperature == 1.0)
        #expect(AppModelKind.gemmaQATSym.officialTopK == 64)
        // Gemma and Ornith reach 128K; 12K is Qwen3.8's alone.
        for kind in [AppModelKind.gemmaQATSym, .ornith] {
            #expect(kind.contextOptions.contains(.oneTwentyEightK))
            #expect(!kind.contextOptions.contains(.twelveK))
        }
        // Qwen3.8: MTP n_max 1, thinking off, the non-thinking sampler pinned, 12K at most.
        #expect(!AppModelKind.qwen38.supportsVision)
        #expect(AppModelKind.qwen38.draftBlockSize == 2)
        #expect(!AppModelKind.qwen38.thinkingDefault)
        #expect(AppModelKind.qwen38.samplingIsLocked)
        #expect(AppModelKind.qwen38.officialTemperature == 0.7)
        #expect(AppModelKind.qwen38.officialTopK == 20)
        #expect(AppModelKind.qwen38.officialTopP == 0.8)
        #expect(AppModelKind.qwen38.contextOptions == [.fourK, .eightK, .twelveK])
        #expect(AppModelKind.qwen38.archConfig == nil)
    }

    @Test func settingsDefaultsFollowTheKind() {
        let gemma = MacAppSettings.defaults(for: .gemmaQATSym)
        #expect(gemma.temperature == 1.0)
        #expect(gemma.topK == 64)
        #expect(!gemma.thinkingEnabled)
        #expect(gemma.mtpEnabled)
        #expect(gemma.contextTokens == 32_768)

        let ornith = MacAppSettings.defaults(for: .ornith)
        #expect(ornith.temperature == 0.6)
        #expect(ornith.topK == 20)
        #expect(ornith.thinkingEnabled)
        #expect(ornith.mtpEnabled)
        #expect(ornith.contextTokens == 32_768)

        let qwen38 = MacAppSettings.defaults(for: .qwen38)
        #expect(qwen38.temperature == 0.7)
        #expect(qwen38.topP == 0.8)
        #expect(!qwen38.thinkingEnabled)
        #expect(qwen38.contextTokens == 12_288)
    }

    @Test func probeReadsTheManifestFamily() throws {
        func makeDirectory(manifest: String?) throws -> URL {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("kind-probe-\(UUID().uuidString).moepack")
            try FileManager.default.createDirectory(at: directory,
                                                    withIntermediateDirectories: true)
            if let manifest {
                try Data(manifest.utf8).write(
                    to: directory.appendingPathComponent("manifest.json"))
            }
            return directory
        }

        let gemma = try makeDirectory(manifest: "{\"arch\": {}}")
        defer { try? FileManager.default.removeItem(at: gemma) }
        #expect(AppModelKind.probe(modelDirectory: gemma) == .gemmaQATSym)

        let ornith = try makeDirectory(
            manifest: "{\"arch\": {\"family\": \"qwen3_5_moe\"}}")
        defer { try? FileManager.default.removeItem(at: ornith) }
        #expect(AppModelKind.probe(modelDirectory: ornith) == .ornith)

        let qwen38 = try makeDirectory(
            manifest: "{\"arch\": {\"family\": \"qwen4exp\"}}")
        defer { try? FileManager.default.removeItem(at: qwen38) }
        #expect(AppModelKind.probe(modelDirectory: qwen38) == .qwen38)

        let unknown = try makeDirectory(
            manifest: "{\"arch\": {\"family\": \"someone-else\"}}")
        defer { try? FileManager.default.removeItem(at: unknown) }
        #expect(AppModelKind.probe(modelDirectory: unknown) == nil)

        let missing = try makeDirectory(manifest: nil)
        defer { try? FileManager.default.removeItem(at: missing) }
        #expect(AppModelKind.probe(modelDirectory: missing) == nil)
    }

    @Test func perKindLocationsAndSources() {
        #expect(AppModelKind.gemmaQATSym.directoryName == "gemma4-qat-sym.moepack")
        #expect(AppModelKind.ornith.directoryName == "ornith-oq4e-g64.moepack")
        let root = URL(fileURLWithPath: "/repo")
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/TsugumiApp/Mac"]
        let resolved = AppModelLocation.resolve(
            kind: .ornith,
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: root,
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(resolved.path == "/repo/scratch/ornith-oq4e-g64.moepack")
        #expect(PrebuiltModelSource.source(for: .ornith).kind == .ornith)
        #expect(PrebuiltModelSource.source(for: .gemmaQATSym).kind == .gemmaQATSym)
    }
}
