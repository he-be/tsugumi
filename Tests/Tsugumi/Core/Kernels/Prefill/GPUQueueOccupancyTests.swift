import Testing
@testable import Tsugumi

@Suite struct GPUQueueOccupancyTests {
    @Test func overlappingAndTouchingIntervalsMergeAndLeaveOnlyRealGaps() {
        var queue = GPUQueueOccupancy()
        queue.record(start: 5.0, end: 6.0)
        queue.record(start: 1.0, end: 2.0)
        queue.record(start: 1.5, end: 3.0)   // overlaps the first
        queue.record(start: 3.0, end: 3.5)   // touches it
        queue.record(start: 4.0, end: 4.5)

        #expect(queue.merged.map { [$0.start, $0.end] } == [[1.0, 3.5], [4.0, 4.5], [5.0, 6.0]])
        #expect(queue.gaps.map { [$0.start, $0.end] } == [[3.5, 4.0], [4.5, 5.0]])
        let occupancy = queue.occupancy
        #expect(occupancy?.busy == 4.0)
        #expect(occupancy?.idle == 1.0)
        #expect(occupancy?.span == 5.0)
    }

    @Test func emptyQueueHasNoOccupancyAndNoGaps() {
        let queue = GPUQueueOccupancy()
        #expect(queue.occupancy == nil)
        #expect(queue.gaps.isEmpty)
    }
}
