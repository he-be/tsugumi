import Foundation
import MoEPackFormat

/// Little-endian binary encoders for the resident `IndexHeader` and 72-byte
/// `IndexEntry` records. Used by the resident writer and the
/// matching loader-side parsers; kept in one place so the on-disk layout
/// changes only here.
enum MoEPackBinary {

    static let indexHeaderBytes = MoEPackFormatV1.residentHeaderBytes
    static let indexEntryBytes = MoEPackFormatV1.residentEntryBytes

    /// Write `IndexHeader { indexSize, residentSize, entryCount }` (24 bytes, LE).
    static func writeIndexHeader(into buf: UnsafeMutableRawPointer,
                                        indexSize: UInt64,
                                        residentSize: UInt64,
                                        entryCount: UInt64) {
        MoEPackResidentIndexCodec.writeHeader(
            into: buf,
            header: MoEPackResidentIndexHeaderV1(
                indexSize: indexSize,
                residentSize: residentSize,
                entryCount: entryCount))
    }

    /// Write one `IndexEntry` (72 bytes, LE) at `dst`. See moepack-format.md.
    static func writeIndexEntry(into dst: UnsafeMutableRawPointer,
                                       entry: ResidentEntry,
                                       nameOffset: UInt32) {
        MoEPackResidentIndexCodec.writeEntry(
            into: dst,
            entry: MoEPackResidentIndexEntryV1(
                name: entry.name,
                dtype: entry.dtype,
                fileOffset: entry.fileOffset,
                sizeBytes: entry.sizeBytes,
                shape: entry.logicalShape4,
                scaleOffset: entry.scaleOffset,
                scaleSize: entry.scaleSize,
                biasOffset: entry.biasOffset,
                biasSize: entry.biasSize),
            nameOffset: nameOffset)
    }
}
