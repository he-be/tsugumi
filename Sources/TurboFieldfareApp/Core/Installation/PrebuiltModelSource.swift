import Foundation

/// One file of a prebuilt install: where it goes relative to the model
/// directory, how many bytes it is, and the SHA-256 the download must match.
public struct PrebuiltFileEntry: Equatable, Sendable {
    public let path: String
    public let bytes: UInt64
    public let sha256: String

    public init(path: String, bytes: UInt64, sha256: String) {
        self.path = path
        self.bytes = bytes
        self.sha256 = sha256
    }
}

/// A finished `.gturbo` install published as plain files on Hugging Face.
///
/// The app used to repack upstream checkpoints itself, but neither adopted
/// weight set can come out of a streaming repack: `sym` needs the staged
/// snapshot's bias ranges (`docs/mtp/44-W1-WEIGHT-DIET.md` §7), and Ornith
/// additionally needs the q_norm bake and the grafted MTP head, which only
/// the Python pipeline produces (`Scripts/qwen35/`). So the finished
/// artifacts are uploaded once — byte for byte the directories this
/// repository built and measured — and the installer downloads files and
/// verifies hashes, nothing more.
///
/// Every file is pinned by SHA-256 below, so `revision` is a convenience for
/// the resolve URL, not the integrity anchor. The repo layout mirrors the
/// install layout exactly, including Ornith's `mtp-head/` sidecar
/// (`docs/qwen35moe/30-MTP-HEAD-GRAFT.md` §6: the head is a sidecar, not a
/// `.gturbo` section).
public struct PrebuiltModelSource: Equatable, Sendable {
    public let kind: AppModelKind
    public let repoID: String
    public let revision: String
    /// `manifest.json`'s `sourceSnapshotHash` without the `sha256:` prefix —
    /// what the installation probe matches the installed manifest against.
    public let sourceIndexSHA256: String
    public let files: [PrebuiltFileEntry]

    public var totalBytes: UInt64 {
        files.reduce(0) { $0 + $1.bytes }
    }

    public func file(at path: String) -> PrebuiltFileEntry? {
        files.first { $0.path == path }
    }

    public static func source(for kind: AppModelKind) -> PrebuiltModelSource {
        switch kind {
        case .gemmaQATSym: gemmaQATSym
        case .ornith: ornith
        }
    }

    public static let gemmaQATSym = PrebuiltModelSource(
        kind: .gemmaQATSym,
        repoID: "mh73772/turbofieldfare-gemma4-qat-sym",
        revision: "main",
        sourceIndexSHA256:
            "7dbbeef0345505798abcf0ac54434116a48c2f1e7aad828071c17a7a871adfe7",
        files: PrebuiltFileTables.gemmaQATSymFiles)

    public static let ornith = PrebuiltModelSource(
        kind: .ornith,
        repoID: "mh73772/turbofieldfare-ornith-oq4e-g64",
        revision: "main",
        sourceIndexSHA256:
            "4280eb9999b17eeb94f45f8ac6ba60510afbf4e1ea5adf32aa83754e68d33bf3",
        files: PrebuiltFileTables.ornithFiles)

    /// The `resolve` URL a file downloads from.
    public func downloadURL(for entry: PrebuiltFileEntry) -> URL {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repoID)/resolve/\(revision)/\(entry.path)"
        return components.url!
    }
}
