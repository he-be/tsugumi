import Foundation

/// `expert-mirror.json` in a model directory: a copy of `packed_experts/` on a
/// second SSD that the runtime reads at the same time as the original
/// (docs/m6-prefill/44). `Scripts/make-expert-mirror.sh` writes it after the
/// copy compares equal to the original, byte for byte.
///
/// With the file present the mirror is on by default. `TF_EXPERT_MIRROR=0`
/// turns it off (the A/B arm).
public struct ExpertMirrorReceipt: Codable, Equatable, Sendable {
    public static let fileName = "expert-mirror.json"
    public static let maxBytes: UInt64 = 64 * 1024

    /// Absolute path of the directory holding the `layer_*.bin` copies.
    public let directory: String
    /// Fraction of each layer's experts read from the copy, in (0, 1].
    public let share: Double
    /// SHA-256 of the `manifest.json` the copy was made from. A reinstalled
    /// model no longer matches, and the mirror is left unused.
    public let manifestSha256: String

    public init(directory: String, share: Double, manifestSha256: String) {
        self.directory = directory
        self.share = share
        self.manifestSha256 = manifestSha256
    }

    public static var disabledByEnvironment: Bool {
        ProcessInfo.processInfo.environment["TF_EXPERT_MIRROR"] == "0"
    }

    /// The receipt a load should use, or nil for none. A missing file is
    /// silence; a mirror that cannot be used (the volume is not mounted, the
    /// model has changed since the copy was made) is a warning on stderr and
    /// a load from the model directory alone. A receipt that does not parse
    /// is an error — it was written by a tool and should not be ignored.
    static func resolve(modelDirectory: MoEPackModelDirectory,
                        manifestSha256: String,
                        warn: (String) -> Void = ExpertMirrorReceipt.warnOnStderr)
        throws -> ExpertMirrorReceipt? {
        guard !disabledByEnvironment else { return nil }
        let data: Data
        do {
            data = try modelDirectory.readMetadata(fileName, maxBytes: maxBytes)
        } catch ModelError.missingFile {
            return nil
        }
        let receipt: ExpertMirrorReceipt
        do {
            receipt = try JSONDecoder().decode(ExpertMirrorReceipt.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "\(fileName): \(error)")
        }
        guard receipt.share > 0, receipt.share <= 1, receipt.directory.hasPrefix("/") else {
            throw ModelError.indexCorrupt(
                detail: "\(fileName): share must be in (0, 1] and directory absolute")
        }
        guard receipt.manifestSha256.lowercased() == manifestSha256.lowercased() else {
            warn("\(fileName) was made for another manifest.json; reading experts from the model directory only")
            return nil
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: receipt.directory, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            warn("\(receipt.directory) is not there (volume not mounted?); reading experts from the model directory only")
            return nil
        }
        return receipt
    }

    static func warnOnStderr(_ message: String) {
        FileHandle.standardError.write(Data("[expert mirror] \(message)\n".utf8))
    }
}
