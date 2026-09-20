import Foundation
import Testing
@testable import TsugumiAppCore

@Suite struct AppModelInstallationProbeTests {
    @Test func missingDirectoryIsMissing() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tsugumi-missing-\(UUID().uuidString).moepack")
        #expect(AppModelInstallationProbe.status(at: url) == .missing)
    }

    @Test func manifestWithoutFinalMetadataIsPartial() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tsugumi-partial-\(UUID().uuidString).moepack")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("{}".utf8).write(to: url.appendingPathComponent("manifest.json"))
        guard case .partial = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected partial status")
            return
        }
    }

    /// A llama-server model is installed when its manifest's server and GGUF are both on this machine
    /// (`LlamaServerModelDirectory`); there is no receipt or snapshot hash to bind.
    @Test func aLlamaServerDirectoryIsCompleteWhenItsServerAndGGUFExist() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tsugumi-llama-server-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("""
            { "arch": { "family": "qwen3_8_dense_llamacpp" }, "llama_server": "/bin/ls", "gguf": "model.gguf" }
            """.utf8).write(to: url.appendingPathComponent("manifest.json"))
        let descriptor = AppModelInstallDescriptor.descriptor(for: .bonsai27b)
        guard case .partial = AppModelInstallationProbe.status(at: url, descriptor: descriptor) else {
            Issue.record("expected partial status while the GGUF is missing")
            return
        }
        try Data().write(to: url.appendingPathComponent("model.gguf"))
        #expect(AppModelInstallationProbe.status(at: url, descriptor: descriptor) == .complete)
    }

    @Test func validBoundedMetadataIsComplete() throws {
        let url = try makeCompleteModelInstall("probe")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(AppModelInstallationProbe.status(at: url) == .complete)
    }

    @Test func receiptBoundToDifferentPathIsPartial() throws {
        let url = try makeCompleteModelInstall("wrong-path")
        defer { try? FileManager.default.removeItem(at: url) }
        let receiptURL = url.appendingPathComponent("verified-install.json")
        var receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as! [String: Any]
        receipt["modelDirectoryPath"] = "/different/model.moepack"
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys]).write(to: receiptURL)
        guard case .partial = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected partial status")
            return
        }
    }

    @Test func differentCheckpointIsPartial() throws {
        let url = try makeCompleteModelInstall("wrong-checkpoint")
        defer { try? FileManager.default.removeItem(at: url) }
        let descriptor = AppModelInstallDescriptor(
            kind: .gemmaQATSym,
            displayName: "different",
            repoID: "example/different",
            revision: "revision",
            sourceIndexSHA256: String(repeating: "f", count: 64),
            approximateDownloadBytes: 1,
            installedBytes: 1,
            rangeStagingBytes: 1,
            reserveBytes: 1)
        guard case .partial = AppModelInstallationProbe.status(at: url, descriptor: descriptor) else {
            Issue.record("expected checkpoint mismatch to be partial")
            return
        }
    }
}
