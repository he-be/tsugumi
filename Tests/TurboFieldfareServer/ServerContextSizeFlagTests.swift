import Testing
@testable import TurboFieldfareServerCore

/// SPEC §11 FLAG-1 / FLAG-2 の起動フラグ面。文脈長のフラグは参照実装の綴り
/// (`-c` / `--ctx-size`) を取り、旧名 `--max-context` は別名として残さず退役する。
@Suite("Server context size flag")
struct ServerContextSizeFlagTests {
    /// FLAG-1. The reference implementation spells this concept `-c` /
    /// `--ctx-size`, so this server does too — both spellings, because the
    /// reference takes both and an operator's command line may carry either.
    @Test func FLAG_1_context_size_takes_the_reference_spelling() throws {
        #expect(try ServerArguments.parse(["--model", "m.gturbo"]).maxContext == 16_384,
                "既定は変わらない")
        for spelling in ["-c", "--ctx-size"] {
            for tokens in [4_096, 8_192, 16_384, 32_768, 65_536, 131_072] {
                let arguments = try ServerArguments.parse([
                    "--model", "m.gturbo", spelling, String(tokens),
                ])
                #expect(arguments.maxContext == tokens, "\(spelling) \(tokens)")
            }
        }
        // A token count is a number. Nothing else is one.
        for value in ["", "16k", "sixteen"] {
            #expect(throws: ServerArgumentError.self) {
                try ServerArguments.parse(["--model", "m.gturbo", "--ctx-size", value])
            }
        }
    }

    /// FLAG-1. Retired means it stops being accepted — `--max-context` is a
    /// usage error like any other unknown flag (exit 2 + usage, from
    /// `main.swift`) — but it is refused *by name* so the message can name the
    /// spelling that replaced it. This is the `--thinking` pattern of FLAG-4:
    /// "unknown flag" would leave an operator holding a command line that used
    /// to work with no idea which half to change. Checked before the "requires
    /// a value" guard, so the bare flag is answered with the retirement too.
    @Test func FLAG_1_max_context_is_retired_and_the_error_names_ctx_size() throws {
        for arguments in [["--model", "m.gturbo", "--max-context", "16384"],
                          ["--model", "m.gturbo", "--max-context", "100000"],
                          ["--model", "m.gturbo", "--max-context"]] {
            let error = #expect(throws: ServerArgumentError.self) {
                try ServerArguments.parse(arguments)
            }
            #expect(error?.description.contains("--ctx-size") == true,
                    "\(arguments) の誤りが新しい綴りを示していない")
        }
        // It is refused, not silently accepted as an alias.
        #expect(throws: ServerArgumentError.self) {
            try ServerArguments.parse(["--model", "m.gturbo", "--max-context", "4096"])
        }
    }

    /// FLAG-2 / DEV-2. The flag takes a free integer and rounds it **down** to
    /// a size this machine's KV arithmetic was measured at. A value outside the
    /// set is never a 400-shaped refusal: the reference takes any `-c` and this
    /// server's only real constraint is what it can keep resident, so the
    /// answer to "too big" is "here is the biggest that fits", not "no". The
    /// value that survived is what `/props` reports as `n_ctx` (EP-4).
    @Test func FLAG_2_ctx_size_rounds_down_to_a_supported_size() throws {
        let expected: [(Int, Int)] = [
            (4_096, 4_096),
            (4_097, 4_096),
            (8_191, 4_096),
            (12_000, 8_192),
            (100_000, 65_536),
            (131_071, 65_536),
            (131_072, 131_072),
            (262_144, 131_072),
            (1_000_000, 131_072),
            // Below the smallest measured size there is nothing to round down
            // to, so the smallest is the floor rather than an error.
            (1, 4_096),
            (2_048, 4_096),
        ]
        for (requested, effective) in expected {
            let arguments = try ServerArguments.parse([
                "--model", "m.gturbo", "--ctx-size", String(requested),
            ])
            #expect(arguments.maxContext == effective,
                    "--ctx-size \(requested) は \(effective) へ下に丸まるべき")
        }
        // A context is a positive number of tokens; zero and below are not
        // sizes to round, they are typos.
        for value in ["0", "-1", "-4096"] {
            #expect(throws: ServerArgumentError.self) {
                try ServerArguments.parse(["--model", "m.gturbo", "-c", value])
            }
        }
    }

    /// FLAG-1. The usage text is where an operator looks after the refusal, so
    /// it carries the new spelling and not the old one.
    @Test func FLAG_1_usage_documents_the_new_spelling_only() {
        #expect(ServerArguments.usage.contains("--ctx-size"))
        #expect(!ServerArguments.usage.contains("--max-context"))
    }
}
