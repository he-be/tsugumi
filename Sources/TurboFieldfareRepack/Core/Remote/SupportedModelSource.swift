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
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
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
