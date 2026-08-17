import Foundation
import TurboFieldfareRepackCore

private let usage = """
Usage:
  TurboFieldfareRepack --output <model.gturbo> [--overwrite] [--resume]
  TurboFieldfareRepack --output <model.gturbo> --source-snapshot <dir> [--overwrite]
  TurboFieldfareRepack --discard-partial --output <model.gturbo>
  TurboFieldfareRepack --add-vision --input-gturbo <model.gturbo>
  TurboFieldfareRepack --add-draft --input-gturbo <model.gturbo>
  TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
  TurboFieldfareRepack --help

The installer streams the supported Gemma 4 checkpoint from Hugging Face and
repackages it without materializing the source checkpoint on disk. Set HF_TOKEN
only if Hugging Face requests authentication. A cancelled or interrupted
download can be continued with --resume or removed with --discard-partial.

--include-vision also installs the Gemma 4 vision tower. Its weights are not
part of the text checkpoint, so they are fetched over ranges from the pinned
upstream repository (about 1.15 GB) and written to vision/vision_weights.bin.
The install refuses to pair a tower with text weights it does not belong to.
Without this flag the output is byte-for-byte what it has always been.

--add-vision adds that same tower to a model that is already installed. The
text weights are not read, rewritten or re-downloaded, so it costs the 1.15 GB
of tower and nothing else, and the result is identical to having installed the
model with --include-vision in the first place. It refuses a model that already
has a tower, and re-verifies the whole install afterwards. An interrupted run
leaves the model untouched; just run it again.

--include-draft also installs the Gemma 4 MTP drafter (the assistant model used
for speculative decoding). It is a separate checkpoint, fetched over ranges from
its own pinned repository (about 236 MB) and written to draft/draft_weights.bin.
The install refuses a drafter that does not match the text checkpoint's K/V
geometry, and one that is not Google's QAT assistant. Without this flag the
output is byte-for-byte what it has always been.

--add-draft adds that same drafter to a model that is already installed, on the
same terms as --add-vision: 236 MB and nothing else, a result identical to
--include-draft, a refusal if a drafter is already there, and a full
re-verification afterwards.

--source-snapshot repacks from a checkpoint already staged on disk in its
distributed form (the safetensors shards plus model.safetensors.index.json,
config.json and the tokenizer files). The snapshot must be one this build
pins: its index digest selects the source, and an unrecognised digest is
rejected. Nothing is downloaded in this mode.
"""

private struct Arguments {
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputGTurbo: String?
    var sourceSnapshot: String?
    var includeVision = false
    var addVision = false
    var includeDraft = false
    var addDraft = false

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--include-vision":
                parsed.includeVision = true
                index += 1
            case "--add-vision":
                parsed.addVision = true
                index += 1
            case "--include-draft":
                parsed.includeDraft = true
                index += 1
            case "--add-draft":
                parsed.addDraft = true
                index += 1
            case "--output", "--input-gturbo", "--source-snapshot":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                switch flag {
                case "--output":     parsed.output = values[index + 1]
                case "--input-gturbo": parsed.inputGTurbo = values[index + 1]
                default:             parsed.sourceSnapshot = values[index + 1]
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        guard !(parsed.addVision && parsed.verifyInstall) else {
            throw ParseError.invalidMode("--add-vision and --verify-install are mutually exclusive")
        }
        guard !(parsed.addDraft && parsed.verifyInstall) else {
            throw ParseError.invalidMode("--add-draft and --verify-install are mutually exclusive")
        }
        // Each append mode owns the model directory for the length of its run,
        // so they are taken one at a time rather than interleaved.
        guard !(parsed.addVision && parsed.addDraft) else {
            throw ParseError.invalidMode("--add-vision and --add-draft are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, parsed.sourceSnapshot == nil,
                  !parsed.overwrite, !parsed.verifyInstall, !parsed.includeVision,
                  !parsed.addVision, !parsed.includeDraft, !parsed.addDraft else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.addVision {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, parsed.sourceSnapshot == nil,
                  !parsed.overwrite, !parsed.resume, !parsed.includeDraft else {
                throw ParseError.invalidMode("--add-vision accepts only --input-gturbo")
            }
            guard !parsed.includeVision else {
                throw ParseError.invalidMode(
                    "--add-vision already installs the tower; drop --include-vision")
            }
            return parsed
        }
        if parsed.addDraft {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, parsed.sourceSnapshot == nil,
                  !parsed.overwrite, !parsed.resume, !parsed.includeVision else {
                throw ParseError.invalidMode("--add-draft accepts only --input-gturbo")
            }
            guard !parsed.includeDraft else {
                throw ParseError.invalidMode(
                    "--add-draft already installs the drafter; drop --include-draft")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, parsed.sourceSnapshot == nil,
                  !parsed.overwrite, !parsed.resume, !parsed.includeVision,
                  !parsed.includeDraft else {
                throw ParseError.invalidMode("verification accepts only --input-gturbo")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-gturbo requires --verify-install")
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.addVision, let input = arguments.inputGTurbo {
        do {
            let result = try await VisionAppendInstaller(
                options: AddVisionOptions(
                    inputGTurbo: URL(fileURLWithPath: input).path,
                    token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                    minFreeReserveBytes: SupportedModelSource.reserveBytes)).run()
            print("Added the vision tower to \(result.modelDirectory)")
            print("Tower: \(result.tensorCount) tensors, \(result.payloadBytes) bytes")
            print("Source: \(result.visionRepoID) @ \(result.visionResolvedCommit)")
            print("Downloaded \(result.downloadedBytes) bytes")
            print("Re-verified \(result.verifiedFileCount) files "
                  + "(\(result.verifiedBytes) bytes)")
            if !result.unexpectedEntries.isEmpty {
                printError("warning: undeclared entries in the model directory: "
                           + result.unexpectedEntries.joined(separator: ", "))
            }
            return 0
        } catch {
            printError("add-vision failed: \(error)")
            return 1
        }
    }

    if arguments.addDraft, let input = arguments.inputGTurbo {
        do {
            let result = try await DraftAppendInstaller(
                options: AddDraftOptions(
                    inputGTurbo: URL(fileURLWithPath: input).path,
                    token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                    minFreeReserveBytes: SupportedModelSource.reserveBytes)).run()
            print("Added the MTP drafter to \(result.modelDirectory)")
            print("Drafter: \(result.tensorCount) tensors, \(result.payloadBytes) bytes")
            print("Source: \(result.draftRepoID) @ \(result.draftResolvedCommit)")
            print("Downloaded \(result.downloadedBytes) bytes")
            print("Re-verified \(result.verifiedFileCount) files "
                  + "(\(result.verifiedBytes) bytes)")
            if !result.unexpectedEntries.isEmpty {
                printError("warning: undeclared entries in the model directory: "
                           + result.unexpectedEntries.joined(separator: ", "))
            }
            return 0
        } catch {
            printError("add-draft failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }
    let options: RemoteStreamingRepackOptions
    if let snapshot = arguments.sourceSnapshot {
        options = RemoteStreamingRepackOptions(
            sourceSnapshotDirectory: snapshot,
            outputDir: URL(fileURLWithPath: output).path,
            minFreeReserveBytes: SupportedModelSource.reserveBytes,
            overwrite: arguments.overwrite,
            resume: arguments.resume,
            includeVision: arguments.includeVision,
            includeDraft: arguments.includeDraft,
            token: ProcessInfo.processInfo.environment["HF_TOKEN"])
    } else {
        options = SupportedModelSource.installOptions(
            outputDirectory: URL(fileURLWithPath: output),
            overwrite: arguments.overwrite,
            token: ProcessInfo.processInfo.environment["HF_TOKEN"],
            resume: arguments.resume,
            includeVision: arguments.includeVision,
            includeDraft: arguments.includeDraft)
    }
    do {
        let result = try await RemoteStreamingRepacker(options: options).run()
        print("Installed \(result.sourceDisplayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
