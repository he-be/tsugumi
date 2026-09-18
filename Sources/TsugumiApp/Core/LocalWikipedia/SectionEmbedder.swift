import CoreML
import Foundation
import Synchronization

/// Sentence vectors for picking the parts of the found articles closest to the user's question
/// (docs/qwen38/39, R1-d). Vectors are unit length, so a dot product is the cosine.
public protocol SectionEmbedding: Sendable {
    func embed(query: String) throws -> [Float]
    func embed(documents: [String]) throws -> [[Float]]
}

/// Ruri v3 (`cl-nagoya/ruri-v3-*`) converted to Core ML the way kohagi lays it out: one `.mlmodelc` with a
/// `seq_<N>` function per length bucket, `tokenizer.json` beside it. Runs on the Neural Engine, so it does not
/// take the GPU from the model. Texts longer than the largest bucket are cut there. Checked against kohagi and HF
/// `tokenizers` on 890 sections (docs/qwen38/38 §2-6).
public final class RuriSectionEmbedder: SectionEmbedding, @unchecked Sendable {
    public enum LoadError: Error, CustomStringConvertible {
        case missing(String)
        public var description: String {
            switch self { case .missing(let what): "no \(what) in the section embedding directory" }
        }
    }

    static let queryPrefix = "検索クエリ: "
    static let documentPrefix = "検索文書: "

    let tokenizer: RuriUnigramTokenizer
    let buckets: [Int]
    private let models: [Int: MLModel]
    /// `MLModel` is not Sendable; one prediction at a time.
    private let predicting = NSLock()

    /// `directory` holds one `.mlmodelc` and `tokenizer.json`.
    public init(directory: URL) throws {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        guard let modelURL = files.first(where: { $0.pathExtension == "mlmodelc" }) else { throw LoadError.missing(".mlmodelc") }
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else { throw LoadError.missing("tokenizer.json") }
        tokenizer = try RuriUnigramTokenizer(json: Data(contentsOf: tokenizerURL))
        let mil = try String(contentsOf: modelURL.appendingPathComponent("model.mil"), encoding: .utf8)
        buckets = mil.components(separatedBy: "func seq_").dropFirst()
            .compactMap { Int($0.prefix { $0.isNumber }) }.sorted()
        guard !buckets.isEmpty else { throw LoadError.missing("seq_<N> function") }
        var loaded: [Int: MLModel] = [:]
        for bucket in buckets {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            configuration.functionName = "seq_\(bucket)"
            loaded[bucket] = try MLModel(contentsOf: modelURL, configuration: configuration)
        }
        models = loaded
    }

    private static let loaded = Mutex<[String: RuriSectionEmbedder]>([:])

    /// One embedder per directory for the process: loading takes seconds (the first time on a Mac, tens of
    /// seconds while Core ML compiles for the Neural Engine).
    public static func shared(directory: URL) throws -> RuriSectionEmbedder {
        if let embedder = loaded.withLock({ $0[directory.path] }) { return embedder }
        let embedder = try RuriSectionEmbedder(directory: directory)
        loaded.withLock { $0[directory.path] = embedder }
        return embedder
    }

    public func embed(query: String) throws -> [Float] {
        try vector(Self.queryPrefix + query)
    }

    public func embed(documents: [String]) throws -> [[Float]] {
        try documents.map { try vector(Self.documentPrefix + $0) }
    }

    /// Mask-aware mean of the last hidden state, L2-normalized (Ruri v3's pooling).
    func vector(_ text: String) throws -> [Float] {
        let maxLength = buckets.last!
        var ids = tokenizer.encode(text)
        if ids.count > maxLength - 2 { ids = Array(ids.prefix(maxLength - 2)) }
        ids = [RuriUnigramTokenizer.bos] + ids + [RuriUnigramTokenizer.eos]
        let count = ids.count
        let bucket = buckets.first { $0 >= count }!
        let inputIDs = try MLMultiArray(shape: [1, NSNumber(value: bucket)], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: bucket)], dataType: .int32)
        for position in 0..<bucket {
            inputIDs[position] = NSNumber(value: position < count ? ids[position] : RuriUnigramTokenizer.pad)
            mask[position] = NSNumber(value: position < count ? 1 : 0)
        }
        let input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIDs, "attention_mask": mask])
        let output = try predicting.withLock { try models[bucket]!.prediction(from: input) }
        guard let hidden = output.featureValue(for: "hidden")?.multiArrayValue else { throw LoadError.missing("hidden output") }
        let dimension = hidden.shape[2].intValue
        let tokenStride = hidden.strides[1].intValue
        let dimensionStride = hidden.strides[2].intValue
        let half = hidden.dataType == .float16
        var sum = [Float](repeating: 0, count: dimension)
        hidden.withUnsafeBytes { raw in
            for position in 0..<count {
                for d in 0..<dimension {
                    let offset = position * tokenStride + d * dimensionStride
                    sum[d] += half ? Float(raw.load(fromByteOffset: offset * 2, as: Float16.self))
                                   : raw.load(fromByteOffset: offset * 4, as: Float.self)
                }
            }
        }
        var norm: Float = 0
        for d in 0..<dimension { norm += sum[d] * sum[d] }
        norm = norm.squareRoot()
        return norm > 0 ? sum.map { $0 / norm } : sum
    }
}

/// Ruri v3's `tokenizer.json`: Unigram (Viterbi over the pieces' scores, unknown characters as UTF-8 byte pieces),
/// spaces as ▁ with none prepended, and `<s> … </s>` around the text. Matches HF `tokenizers` token for token on
/// the sections of docs/qwen38/38 §2-6; each rule below was a mismatch there.
struct RuriUnigramTokenizer {
    static let bos: Int32 = 1
    static let eos: Int32 = 2
    static let pad: Int32 = 3

    // A trie over Unicode scalars: not `Character`, which folds a combining mark or U+FE0F into the character
    // before it, and not `String` keys, whose equality merges canonically equivalent pieces.
    private var children: [[UInt32: Int32]] = [[:]]
    private var pieceID: [Int32] = [-1]
    // Double as `tokenizers` does: summed in Float, equal-score splits (4+8 vs 8+4 ideographic spaces) tie
    // the other way.
    private var pieceScore: [Double] = [0]
    private var unknownScore: Double = 0
    private var byteIDs: [UInt8: Int32] = [:]

    private struct File: Decodable { let model: Model }
    private struct Model: Decodable { let vocab: [Piece] }
    private struct Piece: Decodable {
        let text: String
        let score: Double
        init(from decoder: Decoder) throws {
            var container = try decoder.unkeyedContainer()
            text = try container.decode(String.self)
            score = try container.decode(Double.self)
        }
    }

    /// `JSONDecoder`, not `JSONSerialization`: the latter drops a leading U+FEFF from a string, which turned the
    /// piece "\u{FEFF}" into "".
    init(json: Data) throws {
        let vocab = try JSONDecoder().decode(File.self, from: json).model.vocab
        var minimum = Double.greatestFiniteMagnitude
        for (id, piece) in vocab.enumerated() {
            if id < 6 { continue }  // <unk> <s> </s> <pad> <sep> <mask> never match text
            let text = piece.text
            if text.count == 6, text.hasPrefix("<0x"), text.hasSuffix(">"),
               let byte = UInt8(text.dropFirst(3).prefix(2), radix: 16) {
                byteIDs[byte] = Int32(id)
                continue
            }
            var node = 0
            for scalar in text.unicodeScalars {
                if let next = children[node][scalar.value] {
                    node = Int(next)
                } else {
                    children.append([:]); pieceID.append(-1); pieceScore.append(0)
                    children[node][scalar.value] = Int32(children.count - 1)
                    node = children.count - 1
                }
            }
            pieceID[node] = Int32(id)
            pieceScore[node] = piece.score
            minimum = min(minimum, piece.score)
        }
        unknownScore = minimum - 10  // tokenizers' kUnkPenalty
    }

    /// The ids of `text`, without `<s>` / `</s>`.
    func encode(_ text: String) -> [Int32] {
        let scalars = Array(text.replacingOccurrences(of: " ", with: "▁").unicodeScalars)
        let n = scalars.count
        var best = [Double](repeating: -.infinity, count: n + 1)
        var from = [Int](repeating: -1, count: n + 1)
        var id = [Int32](repeating: -1, count: n + 1)  // -1: an unknown character, written as its bytes
        best[0] = 0
        for start in 0..<n where best[start] > -.infinity {
            var node = 0
            var single = false
            for end in start..<n {
                guard let next = children[node][scalars[end].value] else { break }
                node = Int(next)
                guard pieceID[node] >= 0 else { continue }
                let score = best[start] + pieceScore[node]
                if score > best[end + 1] { best[end + 1] = score; from[end + 1] = start; id[end + 1] = pieceID[node] }
                if end == start { single = true }
            }
            if !single, best[start] + unknownScore > best[start + 1] {
                best[start + 1] = best[start] + unknownScore
                from[start + 1] = start
                id[start + 1] = -1
            }
        }
        var reversed: [Int32] = []
        var end = n
        while end > 0 {
            let start = from[end]
            if id[end] >= 0 {
                reversed.append(id[end])
            } else {
                for byte in String(String.UnicodeScalarView(scalars[start..<end])).utf8.reversed() {
                    reversed.append(byteIDs[byte] ?? 0)
                }
            }
            end = start
        }
        return reversed.reversed()
    }
}
