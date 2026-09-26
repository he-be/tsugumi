import Foundation
import Testing
@testable import TsugumiAppCore
@testable import TsugumiDecodeService
import TsugumiDecodeProtocol

@Suite struct DecodeServiceOutboxTests {
    @Test func cancellationFollowedByThrownCancellationWritesOneTerminal() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)

        let event = try firstTerminal(
            from: outbox,
            published: .cancelled(diagnostics(stopReason: .cancelled)),
            finishError: AppInferenceError.cancelled)

        #expect(event.kind == .cancelled)
        #expect(event.generationID == generationID)
    }

    @Test func failureFollowedByThrownErrorWritesOneTerminal() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)

        let event = try firstTerminal(
            from: outbox,
            published: .failed(.unknown("first"), partial: nil),
            finishError: AppInferenceError.unknown("second"))

        #expect(event.kind == .failed)
        #expect(event.generationID == generationID)
        #expect(event.error == "first")
    }

    @Test func terminalCarriesWherePrefillRan() throws {
        for (round, wire) in [(AppPairRound.pair(k: 46), DecodePairRound(k: 46)),
                              (.fellBack("pair link: closed"), DecodePairRound(fallback: "pair link: closed")),
                              (.alone, DecodePairRound())] {
            var finished = diagnostics(stopReason: .eos)
            finished.pairRound = round
            let event = try firstTerminal(from: DecodeServiceOutbox(generationID: UUID()),
                                          published: .finished(finished), finishError: AppInferenceError.cancelled)
            #expect(event.pairRound == wire)
        }
        let plain = try firstTerminal(from: DecodeServiceOutbox(generationID: UUID()),
                                      published: .finished(diagnostics(stopReason: .eos)),
                                      finishError: AppInferenceError.cancelled)
        #expect(plain.pairRound == nil)
    }

    private func firstTerminal(
        from outbox: DecodeServiceOutbox,
        published event: AppInferenceEvent,
        finishError: Error
    ) throws -> DecodeServiceEvent {
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        outbox.publish(event)
        let terminal = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        outbox.finish(error: finishError)

        #expect(writerFinished.wait(timeout: .now() + 2) == .success)
        #expect(pipe.fileHandleForReading.readDataToEndOfFile().isEmpty)
        return terminal
    }

    private func diagnostics(stopReason: AppStopReason) -> AppDiagnostics {
        AppDiagnostics(
            generatedTokens: 0,
            stopReason: stopReason,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
    }
}
