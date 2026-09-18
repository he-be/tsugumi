import CoreML
import Foundation
// Usage: ruri_embed <.mlmodelc> <tokenizer.json> <prefix> [--group-by page] < {"id","text"[,"page"]} jsonl > {"id","n_tokens","embedding"} jsonl
// Build: swiftc -O -o ruri_embed Scripts/qwen38/ruri_embed/main.swift
// A Ruri v3 Core ML model as kohagi lays it out (one `seq_<N>` function per bucket; takahashim/ruri-v3-*-coreml
// or kohagi's own conversion), loaded once in this process (docs/qwen38/38 §2-6). Texts are cut at the largest bucket.
// Tokenizer: tokenizer.json Unigram (Viterbi, byte fallback), Metaspace without prepend, <s> A </s>.
// Pooling: mask-aware mean of `hidden`, then L2 normalize, as kohagi does.
// RURI_DUMP_IDS=1 adds the token ids; RURI_GAP_S=<s> idles before each group. stderr: load time, then one line per group (page) with its wall time.

struct Unigram {
    // Trie over Unicode scalar values (not Characters, not String keys: a combining mark or FE0F is its own
    // char to tokenizers, and String equality would merge canonically equivalent pieces).
    var children: [[UInt32: Int32]] = [[:]]
    var pieceId: [Int32] = [-1]
    // Double as tokenizers does: with Float, equal-score splits (4+8 vs 8+4 of U+3000) broke ties differently.
    var pieceScore: [Double] = [0]
    var unkScore: Double = 0
    var byteIds: [UInt8: Int32] = [:]

    struct File: Decodable { let model: Model }
    struct Model: Decodable { let vocab: [Piece] }
    struct Piece: Decodable {
        let text: String, score: Double
        init(from d: Decoder) throws { var c = try d.unkeyedContainer(); text = try c.decode(String.self); score = try c.decode(Double.self) }
    }

    // JSONDecoder, not JSONSerialization: the latter drops a leading U+FEFF from strings ("\ufeff" became "").
    init(tokenizerJSON url: URL) throws {
        let vocab = try JSONDecoder().decode(File.self, from: Data(contentsOf: url)).model.vocab
        var minScore = Double.greatestFiniteMagnitude
        for (i, e) in vocab.enumerated() {
            let p = e.text, s = e.score
            if i < 6 { continue }  // specials never match text
            if p.count == 6, p.hasPrefix("<0x"), p.hasSuffix(">"), let b = UInt8(p.dropFirst(3).prefix(2), radix: 16) {
                byteIds[b] = Int32(i); continue
            }
            var node = 0
            for c in p.unicodeScalars {
                if let next = children[node][c.value] { node = Int(next) } else {
                    children.append([:]); pieceId.append(-1); pieceScore.append(0)
                    children[node][c.value] = Int32(children.count - 1); node = children.count - 1
                }
            }
            pieceId[node] = Int32(i); pieceScore[node] = s
            minScore = min(minScore, s)
        }
        unkScore = minScore - 10  // tokenizers' Unigram: kUnkPenalty
    }

    func encode(_ text: String) -> [Int32] {
        let chars = Array(text.replacingOccurrences(of: " ", with: "▁").unicodeScalars)
        let n = chars.count
        var best = [Double](repeating: -.infinity, count: n + 1)
        var from = [Int](repeating: -1, count: n + 1)
        var pid = [Int32](repeating: -1, count: n + 1)  // -1: unknown char (byte fallback)
        best[0] = 0
        for i in 0..<n where best[i] > -.infinity {
            var node = 0
            var matchedSingle = false
            for j in i..<n {
                guard let next = children[node][chars[j].value] else { break }
                node = Int(next)
                if pieceId[node] >= 0 {
                    let v = best[i] + pieceScore[node]
                    if v > best[j + 1] { best[j + 1] = v; from[j + 1] = i; pid[j + 1] = pieceId[node] }
                    if j == i { matchedSingle = true }
                }
            }
            if !matchedSingle {
                let v = best[i] + unkScore
                if v > best[i + 1] { best[i + 1] = v; from[i + 1] = i; pid[i + 1] = -1 }
            }
        }
        var out: [Int32] = []
        var k = n
        while k > 0 {
            let i = from[k]
            if pid[k] >= 0 { out.append(pid[k]) } else {
                for b in chars[i..<k].flatMap({ Array(String($0).utf8) }).reversed() { out.append(byteIds[b] ?? 0) }
            }
            k = i
        }
        return out.reversed()
    }
}

let args = CommandLine.arguments
let modelcURL = URL(fileURLWithPath: args[1])
let tokenizerURL = URL(fileURLWithPath: args[2])
let prefix = args[3]
let groupByPage = args.count > 5 && args[4] == "--group-by" && args[5] == "page"
let mil = try String(contentsOf: modelcURL.appendingPathComponent("model.mil"), encoding: .utf8)
let buckets = mil.components(separatedBy: "func seq_").dropFirst().compactMap { Int($0.prefix { $0.isNumber }) }.sorted()
let maxSeq = buckets.last!
let dumpIds = ProcessInfo.processInfo.environment["RURI_DUMP_IDS"] == "1"  // add "ids" to each record

let tLoad0 = Date()
let tok = try Unigram(tokenizerJSON: tokenizerURL)
let tTok = Date().timeIntervalSince(tLoad0)
var models: [Int: MLModel] = [:]
for b in buckets {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = ProcessInfo.processInfo.environment["RURI_UNITS"] == "all" ? .all : .cpuAndNeuralEngine
    cfg.functionName = "seq_\(b)"
    models[b] = try MLModel(contentsOf: modelcURL, configuration: cfg)
}
FileHandle.standardError.write("buckets \(buckets)\nload: tokenizer \(String(format: "%.3f", tTok)) s, total \(String(format: "%.3f", Date().timeIntervalSince(tLoad0))) s\n".data(using: .utf8)!)

var predictSeconds = 0.0  // prediction plus pooling, per group
func embed(_ text: String) throws -> (Int, [Float], [Int32]) {
    var ids = tok.encode(prefix + text)
    if ids.count > maxSeq - 2 { ids = Array(ids.prefix(maxSeq - 2)) }
    ids = [1] + ids + [2]
    let n = ids.count
    let b = buckets.first { $0 >= n }!
    let inIds = try MLMultiArray(shape: [1, NSNumber(value: b)], dataType: .int32)
    let mask = try MLMultiArray(shape: [1, NSNumber(value: b)], dataType: .int32)
    for t in 0..<b {
        inIds[t] = NSNumber(value: t < n ? ids[t] : 3)
        mask[t] = NSNumber(value: t < n ? 1 : 0)
    }
    let tp = Date()
    defer { predictSeconds += Date().timeIntervalSince(tp) }
    let out = try models[b]!.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": inIds, "attention_mask": mask]))
    let h = out.featureValue(for: "hidden")!.multiArrayValue!
    let dim = h.shape[2].intValue
    var v = [Float](repeating: 0, count: dim)
    let strideT = h.strides[1].intValue, strideD = h.strides[2].intValue
    precondition(h.dataType == .float16 || h.dataType == .float32)
    h.withUnsafeBytes { raw in
        for t in 0..<n {
            for d in 0..<dim {
                let off = t * strideT + d * strideD
                v[d] += h.dataType == .float16
                    ? Float(raw.load(fromByteOffset: off * 2, as: Float16.self))
                    : raw.load(fromByteOffset: off * 4, as: Float.self)
            }
        }
    }
    var norm: Float = 0
    for d in 0..<dim { v[d] /= Float(n); norm += v[d] * v[d] }
    norm = norm.squareRoot()
    for d in 0..<dim { v[d] /= norm }
    return (n, v, ids)
}

struct Input: Decodable {
    let id: Int, text: String, page: String?
}
var groups: [(String, [Input])] = []
while let line = readLine() {
    let rec = try JSONDecoder().decode(Input.self, from: Data(line.utf8))
    let key = groupByPage ? rec.page ?? "" : "\(rec.id)"
    if let last = groups.last, last.0 == key { groups[groups.count - 1].1.append(rec) } else { groups.append((key, [rec])) }
}
let gap = Double(ProcessInfo.processInfo.environment["RURI_GAP_S"] ?? "0") ?? 0  // idle seconds before each group
let out = FileHandle.standardOutput
for (key, recs) in groups {
    if gap > 0 { Thread.sleep(forTimeInterval: gap) }
    let t0 = Date()
    predictSeconds = 0
    var lines: [Data] = []
    var tokens = 0
    for rec in recs {
        let (n, v, ids) = try embed(rec.text)
        tokens += n
        var o: [String: Any] = ["id": rec.id, "n_tokens": n, "embedding": v]
        if dumpIds { o["ids"] = ids }
        lines.append(try JSONSerialization.data(withJSONObject: o))
    }
    let dt = Date().timeIntervalSince(t0)
    FileHandle.standardError.write("group \(key) sections \(recs.count) tokens \(tokens) wall \(String(format: "%.3f", dt)) s predict+pool \(String(format: "%.3f", predictSeconds)) s\n".data(using: .utf8)!)
    for l in lines { out.write(l); out.write("\n".data(using: .utf8)!) }
}
