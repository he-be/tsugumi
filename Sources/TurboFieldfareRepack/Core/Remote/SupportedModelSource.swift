import Foundation

public enum SupportedModelSource {
    public static let displayName = "Gemma 4 26B-A4B IT 4-bit"
    public static let repoID = "mlx-community/gemma-4-26b-a4b-it-4bit"
    public static let revision = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
    public static let sourceIndexSHA256 =
        "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13"
    public static let approximateDownloadBytes: UInt64 = 14_620_479_420
    public static let installedBytes: UInt64 = 14_291_921_884
    public static let reserveBytes: UInt64 = 1_073_741_824

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false,
                                      includeVision: Bool = false,
                                      includeDraft: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume,
            includeVision: includeVision,
            includeDraft: includeDraft)
    }
}

/// Lattice-aligned QAT checkpoint. Its shards are staged on disk first and
/// repacked with `--source-snapshot`; there is no streaming path for it.
public enum QATAlignedModelSource {
    public static let displayName = "Gemma 4 26B-A4B IT QAT q4_0 (lattice-aligned)"
    public static let repoID = "mlx-community/gemma-4-26B-A4B-it-qat-q4_0-mlx-aligned"
    public static let revision = "745a97a754ed4b7713163c7d0e9c11da41809e0c"
    public static let sourceIndexSHA256 =
        "7dbbeef0345505798abcf0ac54434116a48c2f1e7aad828071c17a7a871adfe7"
}

/// Qwen3.5-MoE (Ornith-1.5-35B-A3B), staged on disk and repacked with
/// `--source-snapshot` like the QAT checkpoint. The pinned digest is the one
/// `Scripts/qwen35/bake_snapshot.py` writes: the published oQ4e-g64 conversion
/// with `head_dim ** -0.5` folded into `q_norm`
/// (`docs/qwen35moe/12-OQ4E-G64-AUDIT.md` §5). The bake is deterministic, so
/// re-running it reproduces this digest.
public enum OrnithModelSource {
    public static let displayName = "Ornith-1.5-35B-A3B oQ4e-g64 (q_norm baked)"
    public static let repoID = "scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp"
    /// A staged snapshot has no commit of its own, and the install checkpoint
    /// records a 40-character revision, so the index digest stands in for one —
    /// the same substitution `LocalSnapshotLoader` makes for an unpinned
    /// snapshot.
    public static let revision = "4280eb9999b17eeb94f45f8ac6ba60510afbf4e1"
    public static let sourceIndexSHA256 =
        "4280eb9999b17eeb94f45f8ac6ba60510afbf4e1ea5adf32aa83754e68d33bf3"
}
