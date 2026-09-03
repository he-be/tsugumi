import Foundation
import Testing
@testable import TsugumiAppCore

/// The gauge behind the speedometer: how much of the streamed weights the
/// page cache can hold, from the machine's memory as Activity Monitor
/// splits it.
@Suite struct AppMachineHeadroomTests {
    private let gib: UInt64 = 1 << 30

    /// vm_stat page counts add up the way Activity Monitor adds them:
    /// app = anonymous − purgeable, used = app + wired + compressed,
    /// cached = file-backed + purgeable.
    @Test func hostMemoryFollowsActivityMonitorsSplit() {
        let host = AppHostMemory(
            pageSize: 16_384, physicalBytes: 18 * gib,
            anonymousPages: 207_760, purgeablePages: 7_776,
            wiredPages: 157_661, compressorPages: 71_665,
            fileBackedPages: 641_493)
        #expect(host.appBytes == (207_760 - 7_776) * 16_384)
        #expect(host.wiredBytes == 157_661 * 16_384)
        #expect(host.compressedBytes == 71_665 * 16_384)
        #expect(host.cachedFileBytes == (641_493 + 7_776) * 16_384)
        #expect(host.usedBytes == host.appBytes + host.wiredBytes + host.compressedBytes)
        #expect(host.borrowableBytes == 18 * gib - host.usedBytes)
    }

    @Test func borrowableNeverGoesNegative() {
        let host = AppHostMemory(physicalBytes: 8 * gib, appBytes: 6 * gib,
                                 wiredBytes: 3 * gib, compressedBytes: 0, cachedFileBytes: 0)
        #expect(host.borrowableBytes == 0)
    }

    @Test func levelIsTheBorrowableShareOfWhatTheModelWants() {
        let host = AppHostMemory(physicalBytes: 18 * gib, appBytes: 4 * gib,
                                 wiredBytes: 2 * gib, compressedBytes: 0, cachedFileBytes: 10 * gib)
        let headroom = AppMachineHeadroom(host: host, wantedBytes: 12 * gib)
        #expect(headroom.borrowableBytes == 12 * gib)
        #expect(headroom.level == 1)
        #expect(headroom.shortfallBytes == 0)

        let crowded = AppMachineHeadroom(
            host: AppHostMemory(physicalBytes: 18 * gib, appBytes: 10 * gib,
                                wiredBytes: 2 * gib, compressedBytes: 0, cachedFileBytes: 4 * gib),
            wantedBytes: 12 * gib)
        #expect(crowded.level == 0.5)
        #expect(crowded.shortfallBytes == 6 * gib)
    }

    /// The page cache holding the weights is not "used": after an answer
    /// the level reads the same as before it.
    @Test func cachedWeightsDoNotLowerTheLevel() {
        let before = AppMachineHeadroom(
            host: AppHostMemory(physicalBytes: 18 * gib, appBytes: 5 * gib,
                                wiredBytes: 2 * gib, compressedBytes: 1 * gib, cachedFileBytes: 1 * gib),
            wantedBytes: 12 * gib)
        let after = AppMachineHeadroom(
            host: AppHostMemory(physicalBytes: 18 * gib, appBytes: 5 * gib,
                                wiredBytes: 2 * gib, compressedBytes: 1 * gib, cachedFileBytes: 10 * gib),
            wantedBytes: 12 * gib)
        #expect(before.level == after.level)
    }

    /// What the runtime holds for itself is named, not charged to other
    /// apps; the level does not change, the breakdown does.
    @Test func ownBytesComeOutOfOtherAppsNotOutOfTheLevel() {
        let host = AppHostMemory(physicalBytes: 18 * gib, appBytes: 3 * gib,
                                 wiredBytes: 10 * gib, compressedBytes: 0, cachedFileBytes: 4 * gib)
        let anonymous = AppMachineHeadroom(host: host, wantedBytes: 12 * gib)
        let named = AppMachineHeadroom(host: host, wantedBytes: 12 * gib, ownBytes: 9 * gib)
        #expect(named.level == anonymous.level)
        #expect(anonymous.otherBytes == 13 * gib)
        #expect(named.otherBytes == 4 * gib)
        #expect(AppMachineHeadroom(host: host, wantedBytes: 12 * gib, ownBytes: 20 * gib).otherBytes == 0)
    }

    @Test func levelIsClampedAndTolerantOfAnUnknownModelSize() {
        let roomy = AppHostMemory(physicalBytes: 64 * gib, appBytes: 4 * gib,
                                  wiredBytes: 2 * gib, compressedBytes: 0, cachedFileBytes: 0)
        #expect(AppMachineHeadroom(host: roomy, wantedBytes: 12 * gib).level == 1)
        #expect(AppMachineHeadroom(host: roomy, wantedBytes: 0).level == 1)
    }

    @Test func bandsSplitAtNinetyAndTwentyFivePercent() {
        func headroom(borrowing gib: UInt64, of wanted: UInt64) -> AppMachineHeadroom {
            AppMachineHeadroom(
                host: AppHostMemory(physicalBytes: gib * self.gib, appBytes: 0, wiredBytes: 0,
                                    compressedBytes: 0, cachedFileBytes: 0),
                wantedBytes: wanted * self.gib)
        }
        #expect(headroom(borrowing: 12, of: 12).band == .full)
        #expect(headroom(borrowing: 11, of: 12).band == .full)
        #expect(headroom(borrowing: 10, of: 12).band == .partial)
        #expect(headroom(borrowing: 3, of: 12).band == .partial)
        #expect(headroom(borrowing: 2, of: 12).band == .tight)
        #expect(headroom(borrowing: 0, of: 12).band == .tight)
    }

    @Test func streamedBytesAreTheExpertFiles() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("headroom-\(UUID().uuidString)")
        let experts = directory.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(count: 1_000).write(to: experts.appendingPathComponent("layer_00.bin"))
        try Data(count: 2_500).write(to: experts.appendingPathComponent("layer_01.bin"))
        try Data(count: 99_999).write(to: directory.appendingPathComponent("model_weights.bin"))

        #expect(AppMachineHeadroom.streamedWeightBytes(modelDirectory: directory) == 3_500)
        #expect(AppMachineHeadroom.streamedWeightBytes(
            modelDirectory: directory.appendingPathComponent("missing")) == nil)
    }

    @Test func realSamplerReadsSomethingPlausible() throws {
        let host = try #require(AppHostMemorySampler().sample())
        #expect(host.physicalBytes == ProcessInfo.processInfo.physicalMemory)
        #expect(host.usedBytes > 0)
        #expect(host.usedBytes <= host.physicalBytes)
    }
}

@Suite struct AppModelMachineHeadroomTests {
    private let gib: UInt64 = 1 << 30

    @MainActor
    @Test func refreshMeasuresTheMachineAgainstTheInstalledExperts() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("headroom-model-\(UUID().uuidString)")
        let experts = directory.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(count: 4_096).write(to: experts.appendingPathComponent("layer_00.bin"))

        let host = AppHostMemory(physicalBytes: 18 * gib, appBytes: 5 * gib,
                                 wiredBytes: 2 * gib, compressedBytes: 1 * gib, cachedFileBytes: 0)
        let model = AppModel(modelDirectory: directory,
                             hostMemorySampler: AppHostMemorySampler(read: { host }))
        #expect(model.machineHeadroom == nil)

        model.refreshMachineHeadroom()
        let headroom = try #require(model.machineHeadroom)
        #expect(headroom.wantedBytes == 4_096)
        #expect(headroom.borrowableBytes == 10 * gib)
        #expect(headroom.level == 1)
    }

    @MainActor
    @Test func refreshFallsBackToTheInstallSizeWithoutExpertFiles() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("headroom-empty-\(UUID().uuidString)")
        let host = AppHostMemory(physicalBytes: 18 * gib, appBytes: 12 * gib,
                                 wiredBytes: 2 * gib, compressedBytes: 0, cachedFileBytes: 0)
        let model = AppModel(modelDirectory: directory,
                             hostMemorySampler: AppHostMemorySampler(read: { host }))
        model.refreshMachineHeadroom()
        let headroom = try #require(model.machineHeadroom)
        #expect(headroom.wantedBytes == model.installDescriptor.installedBytes)
        #expect(headroom.borrowableBytes == 4 * gib)
        #expect(headroom.level < 1)
    }

    @MainActor
    @Test func aFailedSampleKeepsTheLastReading() throws {
        let host = AppHostMemory(physicalBytes: 18 * gib, appBytes: 5 * gib,
                                 wiredBytes: 2 * gib, compressedBytes: 0, cachedFileBytes: 0)
        let box = LockedBox<AppHostMemory?>(host)
        let model = AppModel(hostMemorySampler: AppHostMemorySampler(read: { box.value }))
        model.refreshMachineHeadroom()
        let first = try #require(model.machineHeadroom)
        box.value = nil
        model.refreshMachineHeadroom()
        #expect(model.machineHeadroom == first)
    }
}

private final class LockedBox<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value
    init(_ value: Value) { stored = value }
    var value: Value {
        get { lock.lock(); defer { lock.unlock() }; return stored }
        set { lock.lock(); stored = newValue; lock.unlock() }
    }
}

/// The gate in front of the gauge: Tsugumi's own generation must not read
/// as other apps crowding the RAM (docs/MAC_APP.md §4e, 2026-09-04).
@Suite struct AppHeadroomSampleGateTests {
    private let gib: UInt64 = 1 << 30
    /// `#expect` cannot call a mutating member on a `var`, so the gate rides
    /// in a box.
    private final class GateBox {
        var gate = AppHeadroomSampleGate()
        init() {}
        func admits(_ host: AppHostMemory, generating: Bool, at now: Date) -> Bool {
            gate.admits(host, generating: generating, at: now)
        }
        var isSettling: Bool { gate.isSettling }
    }
    private func host(wired: UInt64) -> AppHostMemory {
        AppHostMemory(physicalBytes: 18 * gib, appBytes: 3 * gib,
                      wiredBytes: wired, compressedBytes: 1 * gib, cachedFileBytes: 5 * gib)
    }
    private let t0 = Date(timeIntervalSinceReferenceDate: 1_000)

    @Test func idleSamplesAreAdmitted() {
        let gate = GateBox()
        #expect(gate.admits(host(wired: 5 * gib), generating: false, at: t0))
        #expect(gate.admits(host(wired: 12 * gib), generating: false, at: t0 + 2))
        #expect(!gate.isSettling)
    }

    @Test func nothingIsAdmittedWhileGenerating() {
        let gate = GateBox()
        #expect(gate.admits(host(wired: 5 * gib), generating: false, at: t0))
        #expect(!gate.admits(host(wired: 11 * gib), generating: true, at: t0 + 2))
        #expect(!gate.admits(host(wired: 12 * gib), generating: true, at: t0 + 4))
    }

    /// The measured shape: wired 5.5 → 10〜12 GB while decoding, back within
    /// two seconds. The first sample after the turn still carries the
    /// transient and is refused; the one that is back near the last admitted
    /// reading reopens the gate.
    @Test func afterATurnTheGateWaitsForWiredToComeBack() {
        let gate = GateBox()
        #expect(gate.admits(host(wired: 5_500 << 20), generating: false, at: t0))
        #expect(!gate.admits(host(wired: 11 * gib), generating: true, at: t0 + 2))
        #expect(!gate.admits(host(wired: 10 * gib), generating: false, at: t0 + 4))
        #expect(gate.isSettling)
        #expect(!gate.admits(host(wired: 8 * gib), generating: false, at: t0 + 6))
        #expect(gate.admits(host(wired: 5_700 << 20), generating: false, at: t0 + 8))
        #expect(!gate.isSettling)
        #expect(gate.admits(host(wired: 5_700 << 20), generating: false, at: t0 + 10))
    }

    /// A machine that really changed between turns (another app took the
    /// RAM as wired) must not be held forever: the timeout reopens the gate.
    @Test func theTimeoutReopensTheGateOnAChangedMachine() {
        let gate = GateBox()
        #expect(gate.admits(host(wired: 5 * gib), generating: false, at: t0))
        #expect(!gate.admits(host(wired: 11 * gib), generating: true, at: t0 + 2))
        #expect(!gate.admits(host(wired: 9 * gib), generating: false, at: t0 + 4))
        #expect(!gate.admits(host(wired: 9 * gib), generating: false, at: t0 + 20))
        #expect(gate.admits(host(wired: 9 * gib), generating: false,
                            at: t0 + 4 + AppHeadroomSampleGate.settleTimeout))
        #expect(!gate.isSettling)
    }

    /// A gate that never admitted anything has nothing to compare with and
    /// takes the first idle sample after a turn.
    @Test func withNoBaselineTheFirstIdleSampleIsTaken() {
        let gate = GateBox()
        #expect(!gate.admits(host(wired: 11 * gib), generating: true, at: t0))
        #expect(gate.admits(host(wired: 10 * gib), generating: false, at: t0 + 2))
    }
}
