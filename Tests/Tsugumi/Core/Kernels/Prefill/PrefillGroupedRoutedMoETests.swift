import Dispatch
import Metal
import Testing
@testable import Tsugumi
import TsugumiValidationSupport

@Suite struct PrefillGroupedRoutedMoETests {
    static func measuredPressureRoutes() throws -> PrefillMoEGroupedRoutes {
        var expertAssignments: [UInt32] = []
        expertAssignments.reserveCapacity(256)
        for expert in 0..<16 {
            let count = expert < 5 ? 15 : 14
            expertAssignments.append(contentsOf: repeatElement(UInt32(expert), count: count))
        }
        expertAssignments.append(contentsOf: repeatElement(UInt32(16), count: 27))
        #expect(expertAssignments.count == 256)

        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(256)
        for i in 0..<256 {
            pairs.append(Self.pair(token: UInt32(i / 8),
                                   expert: expertAssignments[i],
                                   rank: UInt32(i % 8)))
        }
        return try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 32,
            topK: 8,
            numExperts: 128,
            tileExpertCount: 16)
    }

    static func pair(token: UInt32, expert: UInt32, rank: UInt32) -> PrefillTokenExpertPair {
        PrefillTokenExpertPair(token: token,
                               expert: expert,
                               rank: rank,
                               weight: Float16(0.125 + Float(rank) * 0.0625))
    }

    static func fakeTensorViews(device: MTLDevice, count: Int) throws -> [TensorView] {
        guard let buffer = device.makeBuffer(length: max(count, 1) * 64,
                                             options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("fake tensor view buffer")
        }
        return (0..<count).map { index in
            TensorView(buffer: buffer,
                       offset: UInt64(index * 64),
                       length: 64,
                       scaleOffset: 0,
                       scaleLength: 0,
                       biasOffset: 0,
                       biasLength: 0,
                       shape: (0, UInt32(index), 0, 0),
                       dtype: 0)
        }
    }

    static func tileFetchRoutes() throws -> PrefillMoEGroupedRoutes {
        let pairs = [
            Self.pair(token: 0, expert: 3, rank: 0),
            Self.pair(token: 0, expert: 1, rank: 1),
            Self.pair(token: 1, expert: 5, rank: 0),
            Self.pair(token: 1, expert: 3, rank: 1),
            Self.pair(token: 2, expert: 1, rank: 0),
            Self.pair(token: 2, expert: 5, rank: 1),
        ]
        return try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 3,
            topK: 2,
            numExperts: 8,
            tileExpertCount: 3)
    }

    static func byte(_ view: TensorView, at relativeOffset: Int) -> UInt8 {
        view.buffer.contents()
            .advanced(by: Int(view.offset) + relativeOffset)
            .load(as: UInt8.self)
    }

    static func streamedViewsWithNonzeroOffsets(device: MTLDevice,
                                                        pool: SyntheticExpertPool,
                                                        expertIDs: [Int]) throws -> [TensorView] {
        try expertIDs.enumerated().map { index, expertID in
            let start = expertID * pool.stride
            let end = start + pool.stride
            let prefix = 64 + index * 16
            let suffix = 32
            var bytes = [UInt8](repeating: 0xA5, count: prefix)
            bytes.append(contentsOf: pool.bytes[start..<end])
            bytes.append(contentsOf: repeatElement(UInt8(0x5A), count: suffix))
            guard let buffer = device.makeBuffer(bytes: bytes,
                                                 length: bytes.count,
                                                 options: .storageModeShared) else {
                throw PrefillGroupedRoutedMoEError.allocationFailed("streamed expert \(expertID)")
            }
            return TensorView(buffer: buffer,
                              offset: UInt64(prefix),
                              length: UInt64(pool.stride),
                              scaleOffset: 0,
                              scaleLength: 0,
                              biasOffset: 0,
                              biasLength: 0,
                              shape: (0, UInt32(expertID), 0, 0),
                              dtype: 0)
        }
    }

    struct SyntheticExpertPool {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
        let stride: Int
    }

    static func makeSyntheticExpertPool(numExperts: Int, d: Int, f: Int) -> SyntheticExpertPool {
        precondition(numExperts > 0, "a pool needs at least one expert")
        // Experts are synthesized independently and the widest caller wants
        // 24 of them at 704x704x3, which is most of this suite's wall clock in
        // a debug build. Build them concurrently and concatenate in expert
        // order; the bytes are the ones the serial build produced.
        var perExpertBytes = [[UInt8]](repeating: [], count: numExperts)
        var perExpertOffsets = [MoEExpertOffsets](repeating: Self.zeroExpertOffsets,
                                                  count: numExperts)
        perExpertBytes.withUnsafeMutableBufferPointer { bytesSlots in
            perExpertOffsets.withUnsafeMutableBufferPointer { offsetSlots in
                // Each index is written by exactly one iteration.
                nonisolated(unsafe) let bytesBase = bytesSlots.baseAddress!
                nonisolated(unsafe) let offsetsBase = offsetSlots.baseAddress!
                DispatchQueue.concurrentPerform(iterations: numExperts) { expert in
                    let built = Self.syntheticExpert(expert: expert, d: d, f: f)
                    (bytesBase + expert).pointee = built.bytes
                    (offsetsBase + expert).pointee = built.offsets
                }
            }
        }

        let offsets = perExpertOffsets[0]
        let stride = perExpertBytes[0].count
        for expert in 1..<numExperts {
            #expect(stride == perExpertBytes[expert].count)
            #expect(offsets.gateWOff == perExpertOffsets[expert].gateWOff)
            #expect(offsets.downBOff == perExpertOffsets[expert].downBOff)
        }

        var allBytes: [UInt8] = []
        allBytes.reserveCapacity(stride * numExperts)
        for expert in 0..<numExperts {
            allBytes.append(contentsOf: perExpertBytes[expert])
        }
        return SyntheticExpertPool(bytes: allBytes, offsets: offsets, stride: stride)
    }

    static let zeroExpertOffsets = MoEExpertOffsets(gateWOff: 0, gateSOff: 0, gateBOff: 0,
                                                    upWOff: 0, upSOff: 0, upBOff: 0,
                                                    downWOff: 0, downSOff: 0, downBOff: 0)

    /// One expert's bytes, in the layout `makeSyntheticExpertPool` concatenates.
    /// Nothing here records an expectation, so it is safe off the test's task.
    static func syntheticExpert(expert: Int, d: Int, f: Int)
        -> (bytes: [UInt8], offsets: MoEExpertOffsets)
    {
        var bytes: [UInt8] = []
        // Each projection is quantized once and then emitted as its three
        // component-major regions. Quantizing per component instead would
        // redo the same work three times, and the widest caller builds
        // this pool at 24 x 704 x 704.
        let gate = Self.quantizedRows(rows: f, cols: d, expert: expert, role: 0)
        let gateWOff = UInt32(bytes.count)
        Self.appendQuantized(gate, to: &bytes, component: .packed)
        let gateSOff = UInt32(bytes.count)
        Self.appendQuantized(gate, to: &bytes, component: .scales)
        let gateBOff = UInt32(bytes.count)
        Self.appendQuantized(gate, to: &bytes, component: .biases)

        let up = Self.quantizedRows(rows: f, cols: d, expert: expert, role: 1)
        let upWOff = UInt32(bytes.count)
        Self.appendQuantized(up, to: &bytes, component: .packed)
        let upSOff = UInt32(bytes.count)
        Self.appendQuantized(up, to: &bytes, component: .scales)
        let upBOff = UInt32(bytes.count)
        Self.appendQuantized(up, to: &bytes, component: .biases)

        let down = Self.quantizedRows(rows: d, cols: f, expert: expert, role: 2)
        let downWOff = UInt32(bytes.count)
        Self.appendQuantized(down, to: &bytes, component: .packed)
        let downSOff = UInt32(bytes.count)
        Self.appendQuantized(down, to: &bytes, component: .scales)
        let downBOff = UInt32(bytes.count)
        Self.appendQuantized(down, to: &bytes, component: .biases)

        return (bytes, MoEExpertOffsets(gateWOff: gateWOff,
                                        gateSOff: gateSOff,
                                        gateBOff: gateBOff,
                                        upWOff: upWOff,
                                        upSOff: upSOff,
                                        upBOff: upBOff,
                                        downWOff: downWOff,
                                        downSOff: downSOff,
                                        downBOff: downBOff))
    }

    enum ProjectionComponent {
        case packed
        case scales
        case biases
    }

    static func quantizedRows(rows: Int, cols: Int, expert: Int, role: Int)
        -> [Quantization.Int4AffineRow]
    {
        Self.syntheticRows(rows: rows, cols: cols, expert: expert, role: role)
            .map { Quantization.quantizeInt4Affine($0) }
    }

    static func appendQuantized(_ quantized: [Quantization.Int4AffineRow],
                                to bytes: inout [UInt8],
                                component: ProjectionComponent) {
        switch component {
        case .packed:
            for row in quantized {
                bytes.append(contentsOf: row.packed)
            }
        case .scales:
            for row in quantized {
                Self.appendU16(row.scales, to: &bytes)
            }
        case .biases:
            for row in quantized {
                Self.appendU16(row.biases, to: &bytes)
            }
        }
    }

    static func syntheticRows(rows: Int, cols: Int, expert: Int, role: Int) -> [[Float]] {
        (0..<rows).map { row in
            (0..<cols).map { col in
                Float(expert + 1) * 0.001
                    + Float(role + 1) * 0.003
                    + Float((row % 7) - 3) * 0.0004
                    + Float((col % 11) - 5) * 0.0002
            }
        }
    }

    static func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
        for value in values {
            bytes.append(UInt8(truncatingIfNeeded: value))
            bytes.append(UInt8(truncatingIfNeeded: value >> 8))
        }
    }

    static func cpuSyntheticRoutePartials(routes: PrefillMoEGroupedRoutes,
                                                  hidden: [Float16],
                                                  hiddenStride: Int,
                                                  pool: SyntheticExpertPool,
                                                  topK: Int,
                                                  d: Int,
                                                  f: Int) -> [Float16] {
        // Every pair reads one token and writes its own `d`-wide slice, so the
        // pairs never overlap. The widest caller is 32 pairs over 704x704,
        // which is tens of seconds of scalar INT4 work in a debug build on one
        // core; spreading the pairs keeps the arithmetic of each dot product
        // exactly as it was.
        let pairs = routes.sortedPairs
        let offsets = pool.offsets
        let stride = pool.stride
        var out = [Float16](repeating: -99, count: routes.queryCount * topK * d)
        pool.bytes.withUnsafeBufferPointer { poolBytes in
            out.withUnsafeMutableBufferPointer { outBuffer in
                // The pool is read-only here, and each pair writes its own
                // `d`-wide slice of the output.
                nonisolated(unsafe) let bytes = poolBytes.baseAddress!
                nonisolated(unsafe) let output = outBuffer.baseAddress!
                DispatchQueue.concurrentPerform(iterations: pairs.count) { index in
                    let pair = pairs[index]
                    let expertBase = Int(pair.expert) * stride
                    let xBase = Int(pair.token) * hiddenStride
                    let x = (0..<d).map { Float(hidden[xBase + $0]) }
                    x.withUnsafeBufferPointer { xValues in
                        let xBuffer = xValues.baseAddress!
                        var gates = [Float](repeating: 0, count: f)
                        var ups = [Float](repeating: 0, count: f)
                        for row in 0..<f {
                            gates[row] = Self.cpuInt4Dot(bytes: bytes,
                                                         base: expertBase,
                                                         wOff: Int(offsets.gateWOff),
                                                         sOff: Int(offsets.gateSOff),
                                                         bOff: Int(offsets.gateBOff),
                                                         row: row,
                                                         n: d,
                                                         x: xBuffer)
                            ups[row] = Self.cpuInt4Dot(bytes: bytes,
                                                       base: expertBase,
                                                       wOff: Int(offsets.upWOff),
                                                       sOff: Int(offsets.upSOff),
                                                       bOff: Int(offsets.upBOff),
                                                       row: row,
                                                       n: d,
                                                       x: xBuffer)
                        }
                        // `geluTanh` is elementwise, so one call over the whole
                        // row vector is the per-element call it replaced.
                        let activated = MoeRef.geluTanh(gates)
                        let act = (0..<f).map { Float(Float16(activated[$0] * ups[$0])) }
                        act.withUnsafeBufferPointer { actValues in
                            let actBuffer = actValues.baseAddress!
                            let outBase = (Int(pair.token) * topK + Int(pair.rank)) * d
                            for row in 0..<d {
                                let value = Self.cpuInt4Dot(bytes: bytes,
                                                            base: expertBase,
                                                            wOff: Int(offsets.downWOff),
                                                            sOff: Int(offsets.downSOff),
                                                            bOff: Int(offsets.downBOff),
                                                            row: row,
                                                            n: f,
                                                            x: actBuffer)
                                (output + outBase + row).pointee = Float16(value)
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    static func readU16(_ bytes: UnsafePointer<UInt8>, _ offset: Int) -> UInt16 {
        UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
    }

    /// Takes raw pointers rather than arrays: at 704x704 the bounds check and
    /// the un-inlined `Array` subscript of a debug build dominate this loop.
    static func cpuInt4Dot(bytes: UnsafePointer<UInt8>,
                                   base: Int,
                                   wOff: Int,
                                   sOff: Int,
                                   bOff: Int,
                                   row: Int,
                                   n: Int,
                                   x: UnsafePointer<Float>) -> Float {
        let groupSize = Quantization.groupSize
        let groups = n / groupSize
        let rowBytes = n / 2
        let wRow = base + wOff + row * rowBytes
        let sRow = base + sOff + row * groups * MemoryLayout<UInt16>.stride
        let bRow = base + bOff + row * groups * MemoryLayout<UInt16>.stride
        var acc: Float = 0
        for group in 0..<groups {
            let scale = Quantization.bf16ToFloat(Self.readU16(bytes, sRow + group * 2))
            let bias = Quantization.bf16ToFloat(Self.readU16(bytes, bRow + group * 2))
            for k in 0..<groupSize {
                let col = group * groupSize + k
                let packed = bytes[wRow + col / 2]
                let q = (k & 1) == 0 ? Float(packed & 0x0F) : Float(packed >> 4)
                acc += (q * scale + bias) * x[col]
            }
        }
        return acc
    }
}
