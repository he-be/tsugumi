import Foundation
import Testing
@testable import TurboFieldfareAppCore

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
        // Both reach 128K.
        for kind in AppModelKind.allCases {
            #expect(kind.contextOptions.contains(.oneTwentyEightK))
        }
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
    }

    @Test func probeReadsTheManifestFamily() throws {
        func makeDirectory(manifest: String?) throws -> URL {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("kind-probe-\(UUID().uuidString).gturbo")
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

        let unknown = try makeDirectory(
            manifest: "{\"arch\": {\"family\": \"someone-else\"}}")
        defer { try? FileManager.default.removeItem(at: unknown) }
        #expect(AppModelKind.probe(modelDirectory: unknown) == nil)

        let missing = try makeDirectory(manifest: nil)
        defer { try? FileManager.default.removeItem(at: missing) }
        #expect(AppModelKind.probe(modelDirectory: missing) == nil)
    }

    @Test func perKindLocationsAndSources() {
        #expect(AppModelKind.gemmaQATSym.directoryName == "gemma4-qat-sym.gturbo")
        #expect(AppModelKind.ornith.directoryName == "ornith-oq4e-g64.gturbo")
        let root = URL(fileURLWithPath: "/repo")
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/TurboFieldfareApp/Mac"]
        let resolved = AppModelLocation.resolve(
            kind: .ornith,
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: root,
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(resolved.path == "/repo/scratch/ornith-oq4e-g64.gturbo")
        #expect(PrebuiltModelSource.source(for: .ornith).kind == .ornith)
        #expect(PrebuiltModelSource.source(for: .gemmaQATSym).kind == .gemmaQATSym)
    }
}
