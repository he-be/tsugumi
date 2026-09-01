import Testing

@testable import Tsugumi

@Suite struct PrefillRoutedGEMMPlannerTests {
    private static func group(_ expert: UInt32, _ start: UInt32, _ count: UInt32) -> PrefillMoEGroup {
        PrefillMoEGroup(expert: expert, pairStart: start, pairCount: count)
    }

    /// Every pair lands in exactly one block, once, with the block's local row
    /// matching its position in the batch — the invariant the GEMM's scratch
    /// indexing and route scatter both depend on.
    private func expectCoversPairsExactlyOnce(_ batches: [PrefillRoutedGEMMBatch],
                                              groups: [PrefillMoEGroup]) {
        var seen: [UInt32: Int] = [:]
        for batch in batches {
            var rowsSeen = Set<UInt32>()
            for block in batch.blocks {
                #expect(block.rowCount >= 1)
                #expect(block.rowCount <= UInt32(PrefillRoutedGEMMPlanner.tileRows))
                #expect(Int(block.localRow) + Int(block.rowCount) <= batch.rows)
                for i in 0..<block.rowCount {
                    #expect(rowsSeen.insert(block.localRow + i).inserted)
                    let pair = block.pairStart + i
                    seen[pair, default: 0] += 1
                    let group = groups[Int(block.localSlot)]
                    #expect(pair >= group.pairStart)
                    #expect(pair < group.pairStart + group.pairCount)
                }
            }
            #expect(rowsSeen.count == batch.rows)
        }
        let expected = groups.flatMap { g in (0..<g.pairCount).map { g.pairStart + $0 } }
        #expect(seen.count == expected.count)
        #expect(seen.values.allSatisfy { $0 == 1 })
    }

    @Test func packsWholeGroupsIntoOneBatchWhenTheyFit() {
        let groups = [Self.group(3, 0, 40), Self.group(7, 40, 20)]
        let batches = PrefillRoutedGEMMPlanner.plan(groups: groups, maxRowsPerBatch: 128)
        #expect(batches.count == 1)
        #expect(batches[0].rows == 60)
        // 40 and 20 rows are one block each: a batch boundary never splits an
        // expert that fits, so no block is smaller than it has to be.
        #expect(batches[0].blocks.count == 2)
        #expect(batches[0].blocks[1].localRow == 40)
        #expect(batches[0].blocks[1].localSlot == 1)
        expectCoversPairsExactlyOnce(batches, groups: groups)
    }

    @Test func startsANewBatchRatherThanSplittingAGroupThatFits() {
        let groups = [Self.group(0, 0, 100), Self.group(1, 100, 100)]
        let batches = PrefillRoutedGEMMPlanner.plan(groups: groups, maxRowsPerBatch: 128)
        #expect(batches.map(\.rows) == [100, 100])
        #expect(batches[0].blocks.map(\.rowCount) == [64, 36])
        expectCoversPairsExactlyOnce(batches, groups: groups)
    }

    @Test func splitsAGroupLargerThanTheBudgetOnFullBlocks() {
        let groups = [Self.group(0, 0, 300)]
        let batches = PrefillRoutedGEMMPlanner.plan(groups: groups, maxRowsPerBatch: 128)
        #expect(batches.map(\.rows) == [128, 128, 44])
        #expect(batches[0].blocks.map(\.rowCount) == [64, 64])
        #expect(batches[2].blocks.map(\.rowCount) == [44])
        expectCoversPairsExactlyOnce(batches, groups: groups)
    }

    /// The budget is floored to a whole number of 64-row blocks, so a batch
    /// never ends mid-block for a group that is still being split.
    @Test func roundsTheBudgetDownToWholeBlocks() {
        let groups = [Self.group(0, 0, 200)]
        let batches = PrefillRoutedGEMMPlanner.plan(groups: groups, maxRowsPerBatch: 100)
        #expect(batches.map(\.rows) == [64, 64, 64, 8])
        expectCoversPairsExactlyOnce(batches, groups: groups)
    }

    @Test func handlesSingleRowGroups() {
        let groups = (0..<16).map { Self.group(UInt32($0), UInt32($0), 1) }
        let batches = PrefillRoutedGEMMPlanner.plan(groups: groups, maxRowsPerBatch: 512)
        #expect(batches.count == 1)
        #expect(batches[0].blocks.count == 16)
        expectCoversPairsExactlyOnce(batches, groups: groups)
    }

    @Test func emptyGroupsProduceNoBatches() {
        #expect(PrefillRoutedGEMMPlanner.plan(groups: [], maxRowsPerBatch: 512).isEmpty)
    }
}
