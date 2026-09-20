import Foundation
import Tsugumi

/// The checkpoints the Mac app knows, and everything about them the
/// UI or the engine has to branch on. One value, so a new model is one new
/// case and the compiler lists every decision it has to make.
///
/// The defaults encode the adopted operating points, not neutral values:
/// MTP is on for both (`docs/SERVER_RUNBOOK.md` — block 4 for Gemma, the
/// fixed width 2 for Ornith), thinking follows each family's recommended use
/// (Ornith reasons by default, Gemma answers directly), and Ornith's sampler
/// is pinned to the official values (S1, `docs/qwen35moe/42-SAMPLING.md`).
public enum AppModelKind: String, CaseIterable, Codable, Sendable, Identifiable {
    case gemmaQATSym = "gemma4-qat-sym"
    case ornith = "ornith-oq4e-g64"
    /// Qwen3.8-Flash-Next off the DS4-IQ2 GGUF (`docs/qwen38`). Not a `.moepack` and not downloadable: a local
    /// directory whose manifest names the GGUF (`Qwen38ModelDirectory`). Operating point 12K, thinking off, MTP n_max 1
    /// (memory of `docs/qwen38/15`).
    case qwen38 = "qwen38-flash-next-iq2"
    /// Ternary Bonsai 2 27B (PQ2_0 + MTP), a dense Qwen3.8 this app has no runner for: a local directory whose
    /// manifest names a GGUF and the `llama-server` that runs it (`LlamaServerModelDirectory`,
    /// `docs/qwen38-27b/10`). Thinking off, MTP n_max 1, 32K (`docs/qwen38-27b/09`).
    case bonsai27b = "bonsai2-27b-pq2"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .gemmaQATSym: "Gemma 4 26B-A4B QAT (Vision + MTP)"
        case .ornith: "Ornith-1.5 35B-A3B (MTP)"
        case .qwen38: "Qwen3.8-Flash-Next IQ2 (MTP)"
        case .bonsai27b: "Bonsai 2 27B PQ2_0 (MTP)"
        }
    }

    public var shortName: String {
        switch self {
        case .gemmaQATSym: "Gemma 4"
        case .ornith: "Ornith-1.5"
        case .qwen38: "Qwen3.8"
        case .bonsai27b: "Bonsai 2"
        }
    }

    /// Whether the model runs in a `llama-server` the app starts and stops (`LlamaServerInferenceClient`) instead of
    /// this project's engines behind the decode service.
    public var runsOnLlamaServer: Bool { self == .bonsai27b }

    /// Only Gemma carries the vision tower; Ornith's Phase 9 never happened. Bonsai has an mmproj, which the app does
    /// not load: one image swapped 297 MB (`docs/qwen38-27b/05` §7).
    public var supportsVision: Bool { self == .gemmaQATSym }

    /// Whether the app declares its tools (web search, the local Wikipedia) to this model. Gemma's and Qwen3.8's
    /// tool loops are the ones the app runs (`docs/WEB_SEARCH.md`, `docs/qwen38/20`); Ornith declares none.
    public var supportsTools: Bool { self != .ornith }

    /// Both templates can render the thought channel; what differs is the
    /// default the toggle starts at.
    public var thinkingDefault: Bool { self == .ornith }

    /// The speculative block each family runs when MTP is on. Gemma's drafter
    /// proposes a block of 4; Ornith's MTP head drafts exactly one token a
    /// pass, so 2 is the only width its kernels have.
    public var draftBlockSize: Int {
        switch self {
        case .gemmaQATSym: 4
        // Bonsai: `--spec-draft-n-max 1`, the server's own; the width only reads the same way here.
        case .ornith, .qwen38, .bonsai27b: 2
        }
    }

    /// S1: Ornith may only run the official recommended sampler, whatever the
    /// UI asked for, so its controls are shown pinned rather than editable.
    public var samplingIsLocked: Bool { self != .gemmaQATSym }

    /// The official recommended sampler for each checkpoint. Gemma's are the
    /// editable defaults; Ornith's are the pinned values the session enforces.
    public var officialTemperature: Double {
        switch self {
        case .gemmaQATSym: 1.0
        case .ornith: 0.6
        case .qwen38, .bonsai27b: 0.7
        }
    }

    public var officialTopK: Int {
        switch self {
        case .gemmaQATSym: 64
        case .ornith, .qwen38, .bonsai27b: 20
        }
    }

    public var officialTopP: Double {
        switch self {
        case .gemmaQATSym, .ornith: 0.95
        case .qwen38, .bonsai27b: 0.8
        }
    }

    /// The whole official sampler, by whether the thought channel is open. `minP` and `presencePenalty` are the two
    /// values a llama-server would otherwise fill with its own defaults (`AppGenerationRequest.minP`); nil where the
    /// model runs only on this Mac's engines, which carry their own.
    public func officialSampling(thinking: Bool) -> AppOfficialSampling {
        switch self {
        case .gemmaQATSym, .ornith:
            AppOfficialSampling(temperature: officialTemperature, topK: officialTopK, topP: officialTopP,
                                minP: nil, presencePenalty: nil)
        // Thinking off is the only operating point; these are `Qwen38Sampler`'s values.
        case .qwen38:
            AppOfficialSampling(temperature: officialTemperature, topK: officialTopK, topP: officialTopP,
                                minP: 0.0, presencePenalty: 1.5)
        // `Qwen/Qwen3.8-27B`'s card, both rows (`docs/qwen38-27b/01` §4). The GGUF's `general.sampling.*` holds only
        // the thinking row, so it is not read.
        case .bonsai27b:
            thinking
                ? AppOfficialSampling(temperature: 1.0, topK: 20, topP: 0.95, minP: 0.0, presencePenalty: 0.0)
                : AppOfficialSampling(temperature: 0.7, topK: 20, topP: 0.8, minP: 0.0, presencePenalty: 1.5)
        }
    }

    /// Directory name of the installed checkpoint, shared by the package-root
    /// `scratch/` layout and the Application Support fallback.
    public var directoryName: String {
        switch self {
        case .gemmaQATSym: "gemma4-qat-sym.moepack"
        case .ornith: "ornith-oq4e-g64.moepack"
        case .qwen38: "Qwen3.8-Flash-Next-DS4-IQ2"
        case .bonsai27b: "Ternary-Bonsai-2-27B-MTP"
        }
    }

    /// The directory name an install made before the rename carries. The
    /// resolver prefers `directoryName` and falls back to this when only the
    /// old directory is on disk, so a working install does not have to be
    /// moved (or re-downloaded) to survive the rename.
    public var legacyDirectoryName: String {
        switch self {
        case .gemmaQATSym: "gemma4-qat-sym.gturbo"
        case .ornith: "ornith-oq4e-g64.gturbo"
        case .qwen38: "Qwen3.8-Flash-Next-DS4-IQ2"
        case .bonsai27b: "Ternary-Bonsai-2-27B-MTP"
        }
    }

    /// The name of the MTP-head sidecar directory *inside* the model
    /// directory. Ornith's head is a 503 MB sidecar, not a `.moepack` section
    /// (`docs/qwen35moe/30-MTP-HEAD-GRAFT.md` §6). The engine falls back to
    /// `QwenMTPSidecar.defaultDirectory` when this is absent, which is where
    /// the development machine keeps it.
    public static let mtpSidecarDirectoryName = "mtp-head"

    /// The architecture the manifest must validate against for this kind; nil for Qwen3.8 and Bonsai, which have no
    /// `.moepack` manifest to validate.
    public var archConfig: ArchConfig? {
        switch self {
        case .gemmaQATSym: .gemma4_26B_A4B
        case .ornith: .ornith1_5_35B_A3B
        case .qwen38, .bonsai27b: nil
        }
    }

    /// Contexts this kind may be loaded at. Both reach 128K; the note about
    /// Ornith's decode cliff at 128K lives in the UI, not here.
    public var contextOptions: [AppContextLengthOption] {
        switch self {
        case .gemmaQATSym, .ornith: AppContextLengthOption.allCases.filter { $0 != .twelveK }
        // 32K is the operating point: with the KV cache in Q8_0 and the indexer keys in F16 it holds 0.59 GB, less than
        // the float32 cache at 12K that `docs/qwen38/09` measured (0.75 GB, `docs/qwen38/22`, `23`).
        case .qwen38: [.fourK, .eightK, .twelveK, .sixteenK, .thirtyTwoK]
        // 32K is what `docs/qwen38-27b/09` ran; the server is restarted with `-c` for another.
        case .bonsai27b: [.fourK, .eightK, .sixteenK, .thirtyTwoK]
        }
    }

    /// The context a fresh settings file starts at.
    public var defaultContextTokens: Int {
        AppContextLengthOption.thirtyTwoK.tokens
    }

    public static let defaultKind = AppModelKind.gemmaQATSym

    /// Which kind an installed directory holds, read from the manifest's
    /// `arch.family` key: absent means Gemma 4, `qwen3_5_moe` means Ornith
    /// (`ManifestArch.family`'s rule). `nil` when there is no readable
    /// manifest or the family is one this app does not ship.
    public static func probe(modelDirectory: URL) -> AppModelKind? {
        struct ArchPeek: Decodable {
            struct Arch: Decodable { let family: String? }
            let arch: Arch
        }
        let manifestURL = modelDirectory.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL),
              let peek = try? JSONDecoder().decode(ArchPeek.self, from: data) else {
            return nil
        }
        switch peek.arch.family {
        case nil: return .gemmaQATSym
        case "qwen3_5_moe": return .ornith
        case "qwen4exp": return .qwen38
        case "qwen3_8_dense_llamacpp": return .bonsai27b
        default: return nil
        }
    }
}

public struct AppOfficialSampling: Equatable, Sendable {
    public var temperature: Double
    public var topK: Int
    public var topP: Double
    public var minP: Double?
    public var presencePenalty: Double?
}
