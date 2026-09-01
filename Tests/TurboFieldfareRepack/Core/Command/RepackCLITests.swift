import Foundation
import Testing

@Suite(.serialized)
struct RepackCLITests {
    @Test func resumeAndDiscardAreMutuallyExclusive() throws {
        let output = temporaryOutput("exclusive")
        defer { clean(output) }
        let result = try run([
            "--output", output,
            "--resume",
            "--discard-partial",
        ])

        #expect(result.status == 2)
        #expect(result.stderr.contains("mutually exclusive"))
    }

    @Test func resumeWithoutStateFailsBeforeNetwork() throws {
        let output = temporaryOutput("missing-resume")
        defer { clean(output) }
        let result = try run([
            "--output", output,
            "--resume",
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func discardWithoutStateReportsAnError() throws {
        let output = temporaryOutput("missing-discard")
        defer { clean(output) }
        let result = try run([
            "--discard-partial",
            "--output", output,
        ])

        #expect(result.status == 1)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func visionIsNotAModeForVerificationOrDiscard() throws {
        let output = temporaryOutput("vision-modes")
        defer { clean(output) }
        let discard = try run(["--discard-partial", "--output", output,
                               "--include-vision"])
        #expect(discard.status == 2)
        #expect(discard.stderr.contains("--discard-partial only accepts --output"))

        let verify = try run(["--verify-install", "--input-gturbo", output,
                              "--include-vision"])
        #expect(verify.status == 2)
        #expect(verify.stderr.contains("verification accepts only --input-gturbo"))
    }

    @Test func helpDocumentsTheVisionFlag() throws {
        let result = try run(["--help"])
        #expect(result.status == 0)
        #expect(result.stdout.contains("--include-vision"))
        #expect(result.stdout.contains("--add-vision"))
    }

    @Test func addVisionTakesOnlyAnInstalledModel() throws {
        let output = temporaryOutput("add-vision-modes")
        defer { clean(output) }

        let missingInput = try run(["--add-vision"])
        #expect(missingInput.status == 2)
        #expect(missingInput.stderr.contains("missing required argument: --input-gturbo"))

        let withOutput = try run(["--add-vision", "--input-gturbo", output,
                                  "--output", output])
        #expect(withOutput.status == 2)
        #expect(withOutput.stderr.contains("--add-vision accepts only --input-gturbo"))

        let redundant = try run(["--add-vision", "--input-gturbo", output,
                                 "--include-vision"])
        #expect(redundant.status == 2)
        #expect(redundant.stderr.contains("drop --include-vision"))

        let bothModes = try run(["--add-vision", "--verify-install",
                                 "--input-gturbo", output])
        #expect(bothModes.status == 2)
        #expect(bothModes.stderr.contains("mutually exclusive"))
    }

    /// The refusal has to happen before anything is downloaded, so a wrong path
    /// costs nothing.
    @Test func addVisionRefusesAPathThatHoldsNoModel() throws {
        let output = temporaryOutput("add-vision-missing")
        defer { clean(output) }
        let result = try run(["--add-vision", "--input-gturbo", output])
        #expect(result.status == 1)
        #expect(result.stderr.contains("no installed model at"))
    }

    private func run(_ arguments: [String]) throws
        -> (status: Int32, stdout: String, stderr: String) {
        let executable = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/debug/TurboFieldfareRepack")
        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        process.executableURL = executable
        process.arguments = arguments
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        let out = stdout.fileHandleForReading.readDataToEndOfFile()
        let err = stderr.fileHandleForReading.readDataToEndOfFile()
        return (
            process.terminationStatus,
            String(decoding: out, as: UTF8.self),
            String(decoding: err, as: UTF8.self))
    }

    private func temporaryOutput(_ tag: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-cli-\(tag)-\(UUID().uuidString).gturbo")
    }

    private func clean(_ output: String) {
        for path in [
            output,
            output + ".partial",
            output + ".install-state",
            output + ".install-state.cleanup",
            output + ".install.lock",
        ] {
            try? FileManager.default.removeItem(atPath: path)
        }
    }
}
