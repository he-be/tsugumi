import Foundation

/// Host sampler for Qwen3.8 (moved from the generation checks, `docs/qwen38/10` §5-1). `greedy`: argmax (lowest id on
/// ties). Otherwise the official non-thinking settings (temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5 over the
/// generated tokens, `HANDOVER-llm-server.md`), in Hugging Face's order: penalty, temperature, top_k, top_p over the
/// tempered top-k mass, draw (xorshift64*).
package struct Qwen38Sampler {
    package let greedy: Bool
    package let temperature: Float = 0.7, topK = 20, topP: Float = 0.8, presence: Float = 1.5
    package var state: UInt64
    package var seen = Set<Int>()
    private var allowed: [Bool] = []

    package init(greedy: Bool, seed: UInt64) {
        self.greedy = greedy
        state = seed &* 0x9E37_79B9_7F4A_7C15 | 1
    }

    package static func argmax(_ l: UnsafeBufferPointer<Float>) -> Int {
        var best = 0
        for i in 1..<l.count where l[i] > l[best] { best = i }
        return best
    }

    package mutating func sample(_ l: UnsafeBufferPointer<Float>) -> Int {
        let id = draw(l, allowed: nil)!
        if !greedy { seen.insert(id) }
        return id
    }

    /// `sample` under a grammar: the unconstrained draw when the gate allows it, else the draw again over the allowed
    /// ids only (the same distribution as masking first, and one mask fill only on a rejection).
    mutating func sample(_ l: UnsafeBufferPointer<Float>, gate: ConstraintGate?, position: Int) throws -> Int {
        guard let gate else { return sample(l) }
        let first = draw(l, allowed: nil)!
        var id = first
        if !gate.allows(Int32(first)) {
            if allowed.count != l.count { allowed = [Bool](repeating: false, count: l.count) }
            try allowed.withUnsafeMutableBufferPointer { try gate.fillAllowedMask($0) }
            guard let masked = allowed.withUnsafeBufferPointer({ draw(l, allowed: $0) }) else {
                throw GenerationConstraintError.noAllowedToken(position: position)
            }
            id = masked
        }
        if !greedy { seen.insert(id) }
        return id
    }

    private mutating func draw(_ l: UnsafeBufferPointer<Float>, allowed: UnsafeBufferPointer<Bool>?) -> Int? {
        if greedy {
            guard let allowed else { return Qwen38Sampler.argmax(l) }
            var best = -1
            for i in 0..<l.count where allowed[i] && (best < 0 || l[i] > l[best]) { best = i }
            return best < 0 ? nil : best
        }
        var top: [(id: Int, z: Float)] = []
        top.reserveCapacity(topK + 1)
        for i in 0..<l.count {
            if let allowed, !allowed[i] { continue }
            let z = seen.contains(i) ? l[i] - presence : l[i]
            if top.count == topK, z <= top[topK - 1].z { continue }
            var at = top.count
            while at > 0 && top[at - 1].z < z { at -= 1 }
            top.insert((i, z), at: at)
            if top.count > topK { top.removeLast() }
        }
        guard !top.isEmpty else { return nil }
        let zMax = top[0].z
        var p = top.map { Double(exp(($0.z - zMax) / temperature)) }
        let sum = p.reduce(0, +)
        p = p.map { $0 / sum }
        var keep = 0, acc = 0.0
        while keep < p.count { acc += p[keep]; keep += 1; if acc >= Double(topP) { break } }
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27
        let u = Double((state &* 0x2545_F491_4F6C_DD1D) >> 11) / Double(1 << 53) * acc
        var run = 0.0
        var id = top[0].id
        for j in 0..<keep { run += p[j]; if u < run { id = top[j].id; break } }
        return id
    }
}
