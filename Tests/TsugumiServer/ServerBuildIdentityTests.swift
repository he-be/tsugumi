import Foundation
import Testing
@testable import TsugumiServerCore

/// SPEC §9 **RSP-5** (`system_fingerprint`), which §3 **EP-4** makes the same
/// value as `/props`'s `build_info`.
@Suite("RSP-5 build identity")
struct ServerBuildIdentityTests {
    private static func temporaryFile(_ contents: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tsugumi-build-\(UUID().uuidString)")
        try Data(contents.utf8).write(to: url)
        return url
    }

    /// RSP-5 calls it a hash of the build, so the value is the build's name and
    /// a digest — not the bare name, which says nothing about *which* build.
    @Test func RSP_5_system_fingerprint_names_the_build_and_a_digest() {
        let fingerprint = ServerBuildIdentity.fingerprint
        let prefix = ServerBuildIdentity.name + "-"

        #expect(fingerprint.hasPrefix(prefix))
        let digest = fingerprint.dropFirst(prefix.count)
        #expect(digest.count == ServerBuildIdentity.digestWidth)
        #expect(digest.allSatisfy { $0.isHexDigit && !$0.isUppercase })
    }

    /// One binary, one value: a client that keys anything on the fingerprint
    /// must not see it move underneath a running server.
    @Test func RSP_5_system_fingerprint_is_stable_for_one_binary() throws {
        #expect(ServerBuildIdentity.fingerprint == ServerBuildIdentity.fingerprint)

        let url = try Self.temporaryFile("one build")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(ServerBuildIdentity.fingerprint(forExecutableAt: url)
            == ServerBuildIdentity.fingerprint(forExecutableAt: url))
    }

    /// It follows the bytes of the executable and nothing else, which is what
    /// makes it a build identity: same bytes, same value; different bytes,
    /// different value.
    @Test func RSP_5_system_fingerprint_follows_the_executable_bytes() throws {
        let first = try Self.temporaryFile("one build")
        let copy = try Self.temporaryFile("one build")
        let other = try Self.temporaryFile("another build")
        defer {
            for url in [first, copy, other] { try? FileManager.default.removeItem(at: url) }
        }

        #expect(ServerBuildIdentity.fingerprint(forExecutableAt: first)
            == ServerBuildIdentity.fingerprint(forExecutableAt: copy))
        #expect(ServerBuildIdentity.fingerprint(forExecutableAt: first)
            != ServerBuildIdentity.fingerprint(forExecutableAt: other))
    }

    /// A build whose executable cannot be read still answers with a stable
    /// string. RSP-5 has no room for a missing field and a value that changed
    /// per process would be worse than one that admits it knows nothing.
    @Test func RSP_5_system_fingerprint_falls_back_when_the_binary_is_unreadable() {
        let missing = URL(fileURLWithPath: "/nonexistent/tsugumi-server")

        #expect(ServerBuildIdentity.fingerprint(forExecutableAt: missing)
            == "\(ServerBuildIdentity.name)-\(ServerBuildIdentity.unknownDigest)")
        #expect(ServerBuildIdentity.fingerprint(forExecutableAt: nil)
            == "\(ServerBuildIdentity.name)-\(ServerBuildIdentity.unknownDigest)")
    }
}
