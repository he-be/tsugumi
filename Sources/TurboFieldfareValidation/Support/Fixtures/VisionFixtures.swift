import Foundation

/// Reader for the reference dumps `Scripts/vision/dump_vision_fixtures.py` writes.
///
/// The fixtures are the whole basis of PLAN_VISION §6-1 layer B: they let the
/// tower be measured against upstream's own intermediates without our resize in
/// the loop, so a kernel bug and a bicubic difference can never be mistaken for
/// each other. They live under `scratch/` (195 MB) rather than in the
/// repository, so everything here degrades to "not available" instead of
/// failing when they are absent.
public enum VisionFixtures {

    public static let defaultRoot = "scratch/vision-fixtures"
    static let magic = Array("TFVFIX01".utf8)

    public enum FixtureError: Error, CustomStringConvertible {
        case unreadable(String, String)
        case badMagic(String)
        case truncated(String)

        public var description: String {
            switch self {
            case let .unreadable(path, detail): return "vision fixture \(path): \(detail)"
            case let .badMagic(path): return "vision fixture \(path): not a TFVFIX01 file"
            case let .truncated(path): return "vision fixture \(path): payload shorter than its header"
            }
        }
    }

    public struct Tensor: Sendable {
        public let shape: [Int]
        public let values: [Float]
        public var count: Int { values.count }
    }

    public struct Case: Sendable {
        public let name: String
        public let directory: URL
        public let imageName: String
        public let imageWidth: Int
        public let imageHeight: Int
        public let maxSoftTokens: Int
        public let patchesWide: Int
        public let patchesHigh: Int
        public let patchCount: Int
        public let softTokenCount: Int

        public func tensor(_ stem: String) throws -> Tensor {
            try VisionFixtures.readTensor(at: directory.appendingPathComponent("\(stem).bin"))
        }
    }

    /// All fixture cases, or `nil` when the fixtures have not been generated.
    public static func load(root: String = defaultRoot) throws -> [Case]? {
        let rootURL = URL(fileURLWithPath: root)
        let manifestURL = rootURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { return nil }

        let data = try Data(contentsOf: manifestURL)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let cases = object["cases"] as? [[String: Any]] else {
            throw FixtureError.unreadable(manifestURL.path, "manifest has no `cases` array")
        }

        return try cases.map { entry in
            func int(_ key: String) throws -> Int {
                guard let value = entry[key] as? Int else {
                    throw FixtureError.unreadable(manifestURL.path, "case is missing `\(key)`")
                }
                return value
            }
            guard let name = entry["name"] as? String,
                  let dir = entry["dir"] as? String,
                  let image = entry["image"] as? String,
                  let size = entry["image_size"] as? [Int], size.count == 2,
                  let grid = entry["patch_grid"] as? [Int], grid.count == 2 else {
                throw FixtureError.unreadable(manifestURL.path, "malformed case entry")
            }
            return Case(name: name,
                        directory: rootURL.appendingPathComponent(dir),
                        imageName: image,
                        imageWidth: size[0],
                        imageHeight: size[1],
                        maxSoftTokens: try int("max_soft_tokens"),
                        patchesWide: grid[0],
                        patchesHigh: grid[1],
                        patchCount: try int("num_patches"),
                        softTokenCount: try int("num_soft_tokens"))
        }
    }

    /// `magic(8) | dtype u32 | ndim u32 | dims u32[ndim] | payload`.
    /// Integer fixtures are widened to `Float` so callers have one type.
    public static func readTensor(at url: URL) throws -> Tensor {
        let data = try Data(contentsOf: url)
        guard data.count >= 16 else { throw FixtureError.truncated(url.path) }
        guard Array(data[0..<8]) == magic else { throw FixtureError.badMagic(url.path) }

        func u32(_ offset: Int) -> Int {
            var value: UInt32 = 0
            _ = withUnsafeMutableBytes(of: &value) { data.copyBytes(to: $0, from: offset..<(offset + 4)) }
            return Int(UInt32(littleEndian: value))
        }

        let dtype = u32(8)
        let ndim = u32(12)
        guard data.count >= 16 + 4 * ndim else { throw FixtureError.truncated(url.path) }
        let shape = (0..<ndim).map { u32(16 + 4 * $0) }
        let count = shape.reduce(1, *)
        let payloadStart = 16 + 4 * ndim
        guard data.count >= payloadStart + 4 * count else { throw FixtureError.truncated(url.path) }

        let payload = data.subdata(in: payloadStart..<(payloadStart + 4 * count))
        let values: [Float]
        switch dtype {
        case 0:
            values = payload.withUnsafeBytes { Array($0.bindMemory(to: Float32.self)) }
        case 1:
            values = payload.withUnsafeBytes { $0.bindMemory(to: Int32.self).map(Float.init) }
        default:
            throw FixtureError.unreadable(url.path, "unknown dtype tag \(dtype)")
        }
        return Tensor(shape: shape, values: values)
    }

    /// Relative error that a NaN cannot pass.
    ///
    /// `RelError.compute` folds its running maximum with `max(_:_:)`, and
    /// `max(0, .nan)` is `0` in Swift, so a kernel that writes NaN scores zero
    /// error and passes (PLAN_VISION §6-3). Here any non-finite value on either
    /// side returns `.infinity`, which no threshold admits.
    public static func relativeError(actual: [Float], reference: [Float]) -> Float {
        guard actual.count == reference.count else { return .infinity }
        var maxAbsDiff: Float = 0
        var referenceNorm: Float = 0
        for i in 0..<actual.count {
            let a = actual[i], r = reference[i]
            if !a.isFinite || !r.isFinite { return .infinity }
            maxAbsDiff = max(maxAbsDiff, abs(a - r))
            referenceNorm = max(referenceNorm, abs(r))
        }
        return maxAbsDiff / max(referenceNorm, 1e-6)
    }
}
