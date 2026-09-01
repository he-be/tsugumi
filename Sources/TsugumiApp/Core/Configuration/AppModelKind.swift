import Foundation
import Tsugumi

/// The two checkpoints the Mac app ships with, and everything about them the
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

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .gemmaQATSym: "Gemma 4 26B-A4B QAT (Vision + MTP)"
        case .ornith: "Ornith-1.5 35B-A3B (MTP)"
        }
    }

    public var shortName: String {
        switch self {
        case .gemmaQATSym: "Gemma 4"
        case .ornith: "Ornith-1.5"
        }
    }

    /// Only Gemma carries the vision tower; Ornith's Phase 9 never happened.
    public var supportsVision: Bool { self == .gemmaQATSym }

    /// Both templates can render the thought channel; what differs is the
    /// default the toggle starts at.
    public var thinkingDefault: Bool { self == .ornith }

    /// The speculative block each family runs when MTP is on. Gemma's drafter
    /// proposes a block of 4; Ornith's MTP head drafts exactly one token a
    /// pass, so 2 is the only width its kernels have.
    public var draftBlockSize: Int {
        switch self {
        case .gemmaQATSym: 4
        case .ornith: 2
        }
    }

    /// S1: Ornith may only run the official recommended sampler, whatever the
    /// UI asked for, so its controls are shown pinned rather than editable.
    public var samplingIsLocked: Bool { self == .ornith }

    /// The official recommended sampler for each checkpoint. Gemma's are the
    /// editable defaults; Ornith's are the pinned values the session enforces.
    public var officialTemperature: Double {
        switch self {
        case .gemmaQATSym: 1.0
        case .ornith: 0.6
        }
    }

    public var officialTopK: Int {
        switch self {
        case .gemmaQATSym: 64
        case .ornith: 20
        }
    }

    public var officialTopP: Double { 0.95 }

    /// Directory name of the installed checkpoint, shared by the package-root
    /// `scratch/` layout and the Application Support fallback.
    public var directoryName: String {
        switch self {
        case .gemmaQATSym: "gemma4-qat-sym.moepack"
        case .ornith: "ornith-oq4e-g64.moepack"
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
        }
    }

    /// The name of the MTP-head sidecar directory *inside* the model
    /// directory. Ornith's head is a 503 MB sidecar, not a `.moepack` section
    /// (`docs/qwen35moe/30-MTP-HEAD-GRAFT.md` §6). The engine falls back to
    /// `QwenMTPSidecar.defaultDirectory` when this is absent, which is where
    /// the development machine keeps it.
    public static let mtpSidecarDirectoryName = "mtp-head"

    /// The architecture the manifest must validate against for this kind.
    public var archConfig: ArchConfig {
        switch self {
        case .gemmaQATSym: .gemma4_26B_A4B
        case .ornith: .ornith1_5_35B_A3B
        }
    }

    /// Contexts this kind may be loaded at. Both reach 128K; the note about
    /// Ornith's decode cliff at 128K lives in the UI, not here.
    public var contextOptions: [AppContextLengthOption] {
        AppContextLengthOption.allCases
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
        default: return nil
        }
    }
}
