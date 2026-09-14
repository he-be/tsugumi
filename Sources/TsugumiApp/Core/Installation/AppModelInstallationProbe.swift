import Foundation
import Tsugumi

public enum AppModelInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    case complete
}

public enum AppModelInstallationProbe {
    public static func status(
        at directory: URL,
        descriptor: AppModelInstallDescriptor = .default
    ) -> AppModelInstallationStatus {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }

        guard let archConfig = descriptor.kind.archConfig else {
            // Qwen3.8: the manifest names the GGUF files, and the tokenizer sidecar has to be beside them.
            do {
                _ = try Qwen38ModelDirectory(directory: directory)
                guard GFTokenizer.tokenizerFolder(forModelDirectory: directory) != nil else {
                    return .partial("tokenizer/ is missing")
                }
                return .complete
            } catch {
                return .partial("\(error)")
            }
        }
        do {
            let manifest = try ManifestReader.load(directoryURL: directory,
                                                   expecting: archConfig)
            let expectedSource = "sha256:" + descriptor.sourceIndexSHA256
            guard manifest.sourceSnapshotHash == expectedSource else {
                return .partial("installed checkpoint does not match \(descriptor.displayName)")
            }
            let layout = directory.appendingPathComponent("packed_experts/layout.json")
            guard FileManager.default.fileExists(atPath: layout.path) else {
                return .partial("packed_experts/layout.json is missing")
            }
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestHash)
            return .complete
        } catch {
            return .partial("\(error)")
        }
    }
}
