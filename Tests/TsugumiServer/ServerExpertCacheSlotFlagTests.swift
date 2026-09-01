import Testing
@testable import Tsugumi
@testable import TsugumiServerCore

/// SPEC §11 FLAG-2 の残り半分。`--expert-cache-slots` も `-c/--ctx-size` と
/// 同じく自由な整数を受け、この機体で確保できる値へ下に丸める (§12 DEV-2)。
@Suite("Server expert cache slot flag")
struct ServerExpertCacheSlotFlagTests {
    /// FLAG-2 / DEV-2. The enumeration is what this machine's Metal working-set
    /// arithmetic was measured at, not a vocabulary the operator has to learn:
    /// a count between the steps — or past the operating point — is rounded
    /// **down** to one this device can hold, never refused for missing the set.
    @Test func FLAG_2_expert_cache_slots_round_down_to_a_supported_count() throws {
        #expect(try ServerArguments.parse(["--model", "m.moepack"]).expertCacheSlots == 32,
                "既定は運用点のまま")
        let expected: [(Int, Int)] = [
            (8, 8),
            (9, 8),
            (15, 8),
            (16, 16),
            (23, 16),
            (24, 24),
            (31, 24),
            (32, 32),
            (33, 32),
            // 32 is the operating point and the ceiling; asking for more is
            // answered with the ceiling, the way an over-long context is.
            (48, 32),
            (96, 32),
            // Below the smallest measured count there is nothing to round down
            // to, so the smallest is the floor rather than an error.
            (1, 8),
            (7, 8),
        ]
        for (requested, effective) in expected {
            let arguments = try ServerArguments.parse([
                "--model", "m.moepack", "--expert-cache-slots", String(requested),
            ])
            #expect(arguments.expertCacheSlots == effective,
                    "--expert-cache-slots \(requested) は \(effective) へ下に丸まるべき")
        }
        // A slot count is a positive number of slots; zero and below are typos,
        // not sizes to round.
        for value in ["0", "-1", "-32", "", "eight"] {
            #expect(throws: ServerArgumentError.self) {
                try ServerArguments.parse([
                    "--model", "m.moepack", "--expert-cache-slots", value,
                ])
            }
        }
    }

    /// Rounding does not loosen the prefill coupling: chunked prefill needs 16
    /// slots, and a count that rounds *below* 16 still fails the combination
    /// check — the guard reads the effective value, not the number the operator
    /// typed. `main.swift` resolves the configuration right after parsing, so
    /// this is still `exit 2` + usage rather than a failure mid-load.
    @Test func FLAG_2_the_prefill_guard_reads_the_rounded_count() throws {
        let tooFew = try ServerArguments.parse([
            "--model", "m.moepack", "--expert-cache-slots", "12", "--prefill", "on",
        ])
        #expect(tooFew.expertCacheSlots == 8)
        #expect(throws: ServerArgumentError.self) {
            try tooFew.resolvedRuntimeConfiguration()
        }

        let fits = try ServerArguments.parse([
            "--model", "m.moepack", "--expert-cache-slots", "20", "--prefill", "on",
        ])
        #expect(fits.expertCacheSlots == 16)
        #expect(throws: Never.self) { try fits.resolvedRuntimeConfiguration() }
    }
}
