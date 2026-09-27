// SPDX-License-Identifier: AGPL-3.0-or-later
// Decode probe for the MTP verify on the M6 (docs/qwen38-27b/16 §14). Loads the pack once and,
// after prefilling a prompt file:
//   1. times one backbone forward of M = 1, 2, 3 tokens (logits at every position, the drafted
//      loop's verify), with the cache restored after each, median of `reps`;
//   2. splits the same forward by layer kind with an eval after every layer: embedding,
//      delta-net layers, attention layers, final norm + output head;
//   3. generates `gen` tokens from the prompt, three seeds each, with drafting off and with the
//      MTP head (whole-block restore and partial keep of a rejected block), the server's official
//      thinking-off sampling (0.7 / 20 / 0.8 / 0.0 / presence 1.5), and reports tok/s and the
//      acceptance.
// Step 1 runs with MLX, FewRowQMV and its half2 variant (mtp.patch) for the two-row products,
// after a check of their verify logits against MLX. Step 2 (split) is defined but not run.
//
//   ishizuki-decode-probe <pack> <prompt.txt> [reps 10] [gen 300]
//
// It is a model process: run it only while no ishizuki-serve holds the model.

import Foundation
import IshizukiKit
import MLX

let args = CommandLine.arguments
guard args.count >= 3 else {
  FileHandle.standardError.write(
    "usage: ishizuki-decode-probe <pack> <prompt.txt> [reps] [gen]\n".data(using: .utf8)!)
  exit(2)
}
let pack = URL(fileURLWithPath: args[1])
let promptText = try String(contentsOfFile: args[2], encoding: .utf8)
let reps = args.count > 3 ? Int(args[3]) ?? 10 : 10
let genTokens = args.count > 4 ? Int(args[4]) ?? 300 : 300

func ms(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 }
func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }
func say(_ s: String) { print(s); fflush(stdout) }

let model = try BonsaiModel(path: pack)
let text = model.text!
say("mtp head: \(model.hasMTP)")
let tokens = model.tokenizer.encode(promptText)

// Prefill once into a cache the forwards extend and give back.
let cache = text.makeCache()
var index = 0
while index < tokens.count {
  let end = min(index + 512, tokens.count)
  let chunk = MLXArray(tokens[index..<end].map { Int32($0) }).reshaped([1, end - index])
  eval(text.trunk(inputs: chunk, cache: cache))
  index = end
}
say("prefilled \(tokens.count) tokens, cache offset \(cache.offset)")

let probe = [tokens[tokens.count / 2], tokens[tokens.count / 3], tokens[tokens.count / 4]]

func forward(_ m: Int) -> Double {
  let ids = MLXArray(probe.prefix(m).map { Int32($0) }).reshaped([1, m])
  var runs: [Double] = []
  for _ in 0..<(reps + 2) {
    let snapshot = cache.snapshot()
    let start = DispatchTime.now().uptimeNanoseconds
    let h = text.trunk(inputs: ids, cache: cache)
    let logits = text.lmHead(text.normed(h))
    eval(logits)
    runs.append(ms(start))
    cache.restore(snapshot)
  }
  return median(Array(runs.dropFirst(2)))
}

func split(_ m: Int) -> [String: Double] {
  let ids = MLXArray(probe.prefix(m).map { Int32($0) }).reshaped([1, m])
  var parts: [String: [Double]] = [:]
  for _ in 0..<(reps + 2) {
    let snapshot = cache.snapshot()
    var t = DispatchTime.now().uptimeNanoseconds
    var h = text.embedTokens(ids)
    eval(h)
    var sums: [String: Double] = ["embed": ms(t)]
    let compute = h.dtype
    let mask = causalMask(length: m, offset: cache.offset, dtype: compute)
    h = h.asType(.float32)
    var linear = 0.0
    var attention = 0.0
    for (i, layer) in text.layers.enumerated() {
      t = DispatchTime.now().uptimeNanoseconds
      h = layer(h, mask: mask, cache: cache.layers[i], positions: nil, compute: compute)
      eval(h)
      if layer.isLinear { linear += ms(t) } else { attention += ms(t) }
    }
    t = DispatchTime.now().uptimeNanoseconds
    let logits = text.lmHead(text.normed(h.asType(compute)))
    eval(logits)
    sums["head"] = ms(t)
    sums["deltanet"] = linear
    sums["attention"] = attention
    for (k, v) in sums { parts[k, default: []].append(v) }
    cache.restore(snapshot)
  }
  return parts.mapValues { median(Array($0.dropFirst(2))) }
}

// The two-row kernels against MLX's own path on the real forward: the same 2-token verify.
do {
  let ids = MLXArray(probe.prefix(2).map { Int32($0) }).reshaped([1, 2])
  var outs: [MLXArray] = []
  for (few, magic) in [(false, false), (true, false), (true, true)] {
    FewRowQMV.enabled = few
    FewRowQMV.magicEnabled = magic
    let snapshot = cache.snapshot()
    let logits = text.lmHead(text.normed(text.trunk(inputs: ids, cache: cache))).asType(.float32)
    eval(logits)
    outs.append(logits)
    cache.restore(snapshot)
  }
  for (i, name) in [(1, "FewRowQMV"), (2, "magic")] {
    let diff = abs(outs[0] - outs[i]).max().item(Float.self)
    let same = (outs[0].argMax(axis: -1) .== outs[i].argMax(axis: -1)).all().item(Bool.self)
    say(String(format: "%@ vs MLX, 2-token verify logits: max |diff| %.4f, argmax equal %@", name, diff, same ? "yes" : "no"))
  }
}

for (few, magic, name) in [(false, false, "MLX"), (true, false, "FewRowQMV"), (true, true, "magic")] {
  FewRowQMV.enabled = few
  FewRowQMV.magicEnabled = magic
  var cells: [String] = []
  for m in 1...2 { cells.append(String(format: "M=%d %.1f ms", m, forward(m))) }
  say("forward (\(name)): \(cells.joined(separator: ", "))")
}

// End to end, as the server samples: thinking off, official values, three seeds per setting.
// All with the partial keep (ISHIZUKI_SPEC_KEEP=1). magic: FewRowQMV.applyMagic for the two-row
// verify; penalty: the MTP head's argmax under the target's penalties (ISHIZUKI_MTP_DRAFT_PENALTY).
BonsaiRuntime.lookupMinMatch = Int(Int32.max)
setenv("ISHIZUKI_SPEC_KEEP", "1", 1)
let settings: [(String, Bool, Bool, String)] = [
  ("draft off                 ", false, false, "0"),
  ("mtp FewRowQMV             ", true, false, "0"),
  ("mtp magic                 ", true, true, "0"),
  ("mtp magic + draft penalty ", true, true, "1"),
  ("mtp FewRowQMV + penalty   ", true, false, "1"),
]
FewRowQMV.enabled = true
for (label, drafting, magic, penalty) in settings {
  FewRowQMV.magicEnabled = magic
  BonsaiRuntime.speculativeDecode = drafting
  setenv("ISHIZUKI_MTP_DRAFT_PENALTY", penalty, 1)
  var rates: [String] = []
  var accepted = 0
  var proposed = 0
  for seed in [1, 2, 3] as [UInt64] {
    var official = SamplingOptions(
      temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0, presencePenalty: 1.5)
    official.seed = seed
    let result = Generator(model: model).generate(
      promptTokens: tokens, options: official, maxTokens: genTokens)
    rates.append(String(format: "%.2f", result.stats.generationTokensPerSecond))
    accepted += result.speculative?.accepted ?? 0
    proposed += result.speculative?.proposed ?? 0
  }
  say("generate \(label): tok/s \(rates.joined(separator: " / ")), accepted \(accepted)/\(proposed)")
}
