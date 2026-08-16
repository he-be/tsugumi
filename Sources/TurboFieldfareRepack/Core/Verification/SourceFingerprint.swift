import Foundation

/// Snapshot fingerprints pinned by the project. Adding a new entry means the
/// importer has been validated against a fresh upload of the source.
public enum SourceFingerprint {
    /// A pinned upstream checkpoint. The revision is part of the pin because a
    /// locally staged snapshot carries no commit of its own: the index digest
    /// is what identifies it, and the revision is what we record for it.
    public struct KnownSource: Sendable, Equatable {
        public let repoID: String
        public let revision: String
        public let indexSha256Hex: String
        public let displayName: String
    }

    public static let knownSources: [KnownSource] = [
        KnownSource(repoID: SupportedModelSource.repoID,
                    revision: SupportedModelSource.revision,
                    indexSha256Hex: SupportedModelSource.sourceIndexSHA256,
                    displayName: SupportedModelSource.displayName),
        KnownSource(repoID: QATAlignedModelSource.repoID,
                    revision: QATAlignedModelSource.revision,
                    indexSha256Hex: QATAlignedModelSource.sourceIndexSHA256,
                    displayName: QATAlignedModelSource.displayName),
    ]

    public static let knownFingerprints: [String: String] = Dictionary(
        uniqueKeysWithValues: knownSources.map { ($0.repoID, $0.indexSha256Hex) })

    /// Returns the pinned source for a given index.json SHA-256, or nil.
    public static func source(forIndexSha256 sha256Hex: String) -> KnownSource? {
        knownSources.first { $0.indexSha256Hex == sha256Hex }
    }

    /// Returns the recognised model ID for a given index.json SHA-256, or nil.
    public static func modelID(forIndexSha256 sha256Hex: String) -> String? {
        source(forIndexSha256: sha256Hex)?.repoID
    }
}
