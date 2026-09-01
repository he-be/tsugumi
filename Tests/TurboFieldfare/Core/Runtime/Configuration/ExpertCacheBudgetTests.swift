import Testing

@testable import TurboFieldfare

/// The load-time guard charges what a configuration *allocates*.
///
/// Before `docs/mtp/52-D-P7-PREFILL-QUEUE-DEPTH.md` §9 both arms were charged
/// the private-slot formula, which on the mmap arm is memory nobody asks the
/// allocator for: measured peak there is flat at 1.29 GB (32 slots) and 1.31 GB
/// (48 slots). That over-charge rejected 48 slots and a 128K context on this
/// machine's stock `iogpu.wired_limit_mb` while the process would have used a
/// fraction of it.
@Suite struct ExpertCacheBudgetTests {

    static let gigabyte: UInt64 = 1_000_000_000

    /// The 128K case that was refused: everything but the expert term adds up
    /// to 6.26 GB against a 8.59 GB working set.
    static func budget(expertCache: UInt64, residencyRequest: UInt64) -> ExpertCacheBudget {
        ExpertCacheBudget(
            residentBytes: 1_510_000_000,
            visionResidentBytes: 1_150_000_000,
            expertCacheBytes: expertCache,
            expertResidencyRequestBytes: residencyRequest,
            kvCacheBytes: 3_310_000_000,
            recurrentStateBytes: 0,
            prefillScratchBytes: 290_000_000,
            recommendedWorkingSetBytes: 8_590_000_000,
            slotCount: 32)
    }

    @Test func privateSlotsAreChargedAndCanExceedTheWorkingSet() {
        let pread = Self.budget(expertCache: 3_570_000_000, residencyRequest: 0)
        #expect(pread.totalBytes == 9_830_000_000)
        #expect(!pread.fitsRecommendedWorkingSet)
        #expect(pread.summary.contains("experts 3.57 GB"))
        #expect(!pread.summary.contains("not an allocation"))
    }

    @Test func mappedExpertsAreReportedButNotCharged() {
        let mmap = Self.budget(expertCache: 0, residencyRequest: 3_570_000_000)
        #expect(mmap.totalBytes == 6_260_000_000)
        #expect(mmap.fitsRecommendedWorkingSet)
        // Still visible in the message — the pages are real, they are just
        // file-backed and evictable, so a shortfall costs faults, not swap.
        #expect(mmap.summary.contains("3.57 GB of mapped experts"))
        #expect(mmap.summary.contains("not an allocation"))
    }

    /// Slots stop moving the guard on the mmap arm, which is what makes
    /// `--expert-cache-slots` a hit-rate control there rather than a memory one.
    /// That is also why the front ends cap the flag at the operating point
    /// (`RuntimeConfiguration.allowedExpertCacheSlots`): a slot count that would
    /// ask for more residency than the device has no longer announces itself as
    /// footprint, so the range is what keeps it out, not this arithmetic.
    @Test func slotCountDoesNotMoveTheMappedArmsTotal() {
        let small = Self.budget(expertCache: 0, residencyRequest: 1_790_000_000)
        let ceiling = Self.budget(expertCache: 0, residencyRequest: 3_570_000_000)
        #expect(small.totalBytes == ceiling.totalBytes)
        #expect(ceiling.fitsRecommendedWorkingSet)
    }

    @Test func theAcceptedRangeStopsAtTheOperatingPoint() {
        #expect(RuntimeConfiguration.allowedExpertCacheSlots == [8, 16, 24, 32])
        #expect(RuntimeConfiguration.maximumContextTokens == 131_072)
        #expect(RuntimeConfiguration().expertCacheSlots == 32)
    }

    /// The load-time refusal is read by two front ends that no longer spell the
    /// context flag the same way: the CLI takes `--max-context` and the server
    /// retired it for `-c/--ctx-size` (SPEC §11 FLAG-1). A message naming
    /// either one is wrong for the other half of its readers, so it names the
    /// levers in words and no flag at all.
    @Test func theWorkingSetErrorNamesLeversAndNoFlag() {
        let pread = ExpertCacheBudgetError
            .exceedsRecommendedWorkingSet(Self.budget(expertCache: 3_570_000_000,
                                                      residencyRequest: 0))
            .description
        let mapped = ExpertCacheBudgetError
            .exceedsRecommendedWorkingSet(Self.budget(expertCache: 0,
                                                      residencyRequest: 9_000_000_000))
            .description
        for message in [pread, mapped] {
            #expect(!message.contains("--"), "起動フラグの綴りを名指ししている: \(message)")
            #expect(message.contains("context size"))
        }
        // The private-slot arm has one more lever than the mapped arm: slots are
        // charged there, so lowering them moves this total. (The slot *count*
        // appears in both summaries; only this arm offers it as a lever.)
        #expect(pread.contains("expert-cache slots"))
        #expect(!mapped.contains("expert-cache slots"))
    }

    /// A device that reports no recommendation (zero) is not a device that
    /// rejects everything.
    @Test func anUnknownWorkingSetAcceptsAnyConfiguration() {
        let unknown = ExpertCacheBudget(
            residentBytes: 1_510_000_000,
            visionResidentBytes: 0,
            expertCacheBytes: 3_570_000_000,
            expertResidencyRequestBytes: 0,
            kvCacheBytes: 3_310_000_000,
            recurrentStateBytes: 0,
            prefillScratchBytes: 290_000_000,
            recommendedWorkingSetBytes: 0,
            slotCount: 32)
        #expect(unknown.fitsRecommendedWorkingSet)
    }
}
