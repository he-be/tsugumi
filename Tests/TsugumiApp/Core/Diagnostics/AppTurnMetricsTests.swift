import Foundation
import Testing
@testable import TsugumiAppCore

/// The metrics line every round leaves behind: the machine at the start
/// (speedometer, measured residency) next to what the round cost.
@Suite struct AppTurnMetricsTests {
    private let gib: UInt64 = 1 << 30

    private func temporaryFile(_ name: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("turn-metrics-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent(name)
    }

    @Test func appendWritesOneJSONLinePerRecordAndReadsThemBack() throws {
        let file = temporaryFile("turn-metrics.jsonl")
        defer { try? FileManager.default.removeItem(at: file.deletingLastPathComponent()) }
        let log = AppTurnMetricsLog(fileURL: file)
        let diagnostics = AppDiagnostics(
            generatedTokens: 12, stopReason: .eos, promptTokenCount: 2_177,
            cachedPromptTokens: 0, timeToFirstTokenSeconds: 0.1,
            decodeSeconds: 0.3, tokensPerSecond: 40, peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        let headroom = AppMachineHeadroom(
            host: AppHostMemory(physicalBytes: 18 * gib, appBytes: 5 * gib, wiredBytes: 2 * gib,
                                compressedBytes: 1 * gib, cachedFileBytes: 4 * gib),
            wantedBytes: 12 * gib)
        let record = AppTurnMetricsRecord(
            recordedAt: "2026-09-03T00:00:00Z", turnID: "T", round: 1, chatID: "C",
            outcome: "finished", model: "gemma4-qat-sym", contextTokens: 32_768, slots: 32,
            mtp: true, thinking: false, network: "online", directive: nil,
            headroom: headroom, weightsResidentFraction: 0.5,
            diagnostics: diagnostics, toolCalls: 0)
        log.append(record)
        log.append(record)

        let text = try String(contentsOf: file, encoding: .utf8)
        #expect(text.split(separator: "\n").count == 2)
        let back = AppTurnMetricsLog.read(from: file)
        #expect(back == [record, record])
        #expect(back.first?.headroomLevel == headroom.level)
        #expect(back.first?.borrowableBytes == 10 * gib)
        #expect(back.first?.promptTokens == 2_177)
        #expect(back.first?.stopReason == "eos")
    }

    @Test func residencyProbeCountsPagesOfTheExpertFiles() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("residency-\(UUID().uuidString)")
        let experts = directory.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = experts.appendingPathComponent("layer_00.bin")
        try Data(repeating: 7, count: 1 << 20).write(to: file)
        _ = try Data(contentsOf: file) // just written and read: in the page cache

        let fraction = try #require(AppWeightResidencyProbe.residentFraction(modelDirectory: directory))
        #expect(fraction > 0.5 && fraction <= 1)
        #expect(AppWeightResidencyProbe.residentFraction(
            modelDirectory: directory.appendingPathComponent("missing")) == nil)
    }

    @MainActor
    @Test func aFinishedRoundLeavesOneLineWithTheMachineAndTheCost() async throws {
        let file = temporaryFile("turn-metrics.jsonl")
        defer { try? FileManager.default.removeItem(at: file.deletingLastPathComponent()) }
        let host = AppHostMemory(physicalBytes: 18 * gib, appBytes: 5 * gib, wiredBytes: 2 * gib,
                                 compressedBytes: 1 * gib, cachedFileBytes: 4 * gib)
        let client = MockInferenceClient(response: "one two", tokenDelayNanos: 0)
        let model = AppModel(client: client,
                             hostMemorySampler: AppHostMemorySampler(read: { host }),
                             turnMetricsLog: AppTurnMetricsLog(fileURL: file))
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "go"
        model.run()
        for _ in 0..<400 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        var records: [AppTurnMetricsRecord] = []
        for _ in 0..<200 where records.isEmpty {
            try? await Task.sleep(nanoseconds: 5_000_000)
            records = AppTurnMetricsLog.read(from: file)
        }
        let record = try #require(records.first)
        #expect(records.count == 1)
        #expect(record.outcome == "finished")
        #expect(record.round == 1)
        #expect(record.chatID == model.selectedChat.id.uuidString)
        #expect(record.borrowableBytes == 10 * gib)
        #expect(record.wantedBytes == model.installDescriptor.installedBytes)
        #expect(record.weightsResidentFraction == nil)
        #expect(record.generatedTokens == model.diagnostics?.generatedTokens)
        #expect(record.contextTokens == model.maxContextTokens)
    }
}
