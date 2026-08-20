import CryptoKit
import Foundation

/// SPEC §9 **RSP-5** (`system_fingerprint`) and §3 **EP-4** (`build_info`).
///
/// EP-4 says the two are the same value, so the value is defined once, here,
/// and both read it. It has to be stable for one binary and it must not need
/// the model — `/props` answers before the load has landed (LIF-1/LIF-2) and a
/// fingerprint that changed with the checkpoint would be a model id, not a
/// build id.
///
/// Nothing in this checkout stamps a build number, so the only thing that
/// actually distinguishes one build of this server from another is the bytes
/// of the executable that is running. That is what this hashes.
public enum ServerBuildIdentity {
    /// The name every fingerprint carries, so a client that reads one can see
    /// what produced it before it looks at the digest.
    public static let name = "TurboFieldfareServer"

    /// How much of the digest goes into the fingerprint. Twelve hex digits is
    /// the width the reference implementation's `build_info` commit half has,
    /// and it is far past the point where two builds collide by accident.
    static let digestWidth = 12

    /// What a build with no readable executable is called. Still a stable
    /// string: a client may key a cache on it, so it may not be a random one.
    static let unknownDigest = "unknown"

    /// EP-4's `build_info` and RSP-5's `system_fingerprint`.
    public static let fingerprint = "\(name)-\(unknownDigest)"

    /// The fingerprint of a named executable, which is what makes the value
    /// above checkable without reasoning about whatever binary the tests
    /// themselves happen to be running as.
    static func fingerprint(forExecutableAt url: URL?) -> String {
        "\(name)-\(unknownDigest)"
    }
}
