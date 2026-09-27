// SPDX-License-Identifier: AGPL-3.0-or-later
// Few-row matmul timing for the MTP verify on the M6 (docs/qwen38-27b/16 §14). No model is
// loaded: every weight is random, quantized the way the Bonsai 2 27B pack holds it (2-bit
// affine, group 128), at each shape its backbone multiplies once per token. The output head
// (248320x5120) is left out: it would need 5 GB of random floats beside the resident server,
// and the kernel's own table already takes it at two rows.
//
//   ishizuki-verify-bench [iterations]
//
// Per shape and row count M: MLX's quantizedMM, and ishizuki's VerifyMatmul with its row
// floor lowered to M (mtp.patch), and FewRowQMV (mtp.patch) for M 2 to 4. Each timing chains 16 products into one eval and divides,
// the median of `iterations` such evals. The kernel's output is checked against MLX's.

import Foundation
import IshizukiKit
import MLX
import MLXRandom

let iterations = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1]) ?? 20 : 20
let chain = 16

// (label, output n, input k, count per token in the 64-layer backbone)
let shapes: [(String, Int, Int, Int)] = [
  ("mlp.gate/up      17408x5120", 17408, 5120, 128),
  ("mlp.down         5120x17408", 5120, 17408, 64),
  ("gdn.in_proj_qkv  10240x5120", 10240, 5120, 48),
  ("gdn.in_proj_z    6144x5120", 6144, 5120, 48),
  ("gdn/attn.out     5120x6144", 5120, 6144, 64),
  ("attn.q           12288x5120", 12288, 5120, 16),
  ("attn.k/v         1024x5120", 1024, 5120, 32),
]

func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }

func time(_ body: () -> [MLXArray]) -> Double {
  eval(body())
  var runs: [Double] = []
  for _ in 0..<iterations {
    let start = DispatchTime.now().uptimeNanoseconds
    eval(body())
    runs.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3 / Double(chain))
  }
  return median(runs)
}

MLXRandom.seed(7)
var perToken: [Int: (mlx: Double, kernel: Double, few: Double)] = [:]
print("shape                         M   mlx us   verify us   max|diff|   few-row us   max|diff|")
for (label, n, k, count) in shapes {
  let packed = quantized(
    (MLXRandom.normal([n, k]) * 0.02).asType(.float16), groupSize: 128, bits: 2)
  let (w, scales, biases) = (packed.wq, packed.scales, packed.biases!)
  eval(w, scales, biases)
  for m in 1...3 {
    let xs = (0..<chain).map { _ in MLXRandom.normal([m, k]).asType(.float16) }
    eval(xs)
    let mlx = time {
      xs.map {
        quantizedMM($0, w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
      }
    }
    var kernel = Double.nan
    var diff = Float.nan
    if m >= 2 {
      VerifyMatmul.rowFloor = m
      if let got = VerifyMatmul.apply(xs[0], w, scales: scales, biases: biases, groupSize: 128, bits: 2) {
        let ref = quantizedMM(xs[0], w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
        diff = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
        kernel = time {
          xs.map { VerifyMatmul.apply($0, w, scales: scales, biases: biases, groupSize: 128, bits: 2)! }
        }
      }
    }
    var few = Double.nan
    var fewDiff = Float.nan
    if let got = FewRowQMV.apply(xs[0], w, scales: scales, biases: biases, groupSize: 128, bits: 2) {
      let ref = quantizedMM(xs[0], w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
      fewDiff = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
      few = time {
        xs.map { FewRowQMV.apply($0, w, scales: scales, biases: biases, groupSize: 128, bits: 2)! }
      }
    }
    print(String(format: "%@  %d  %8.1f  %9.1f   %9.3g   %10.1f   %.3g", label, m, mlx, kernel, diff, few, fewDiff))
    let sofar = perToken[m] ?? (0, 0, 0)
    perToken[m] = (
      sofar.mlx + mlx * Double(count),
      sofar.kernel + (kernel.isNaN ? mlx : min(mlx, kernel)) * Double(count),
      sofar.few + (few.isNaN ? mlx : few) * Double(count))
  }
}
for m in 1...3 {
  let t = perToken[m]!
  print(String(format: "M=%d  one backbone pass of matmuls: mlx %.1f ms, best of mlx/verify %.1f ms, few-row %.1f ms", m, t.mlx / 1e3, t.kernel / 1e3, t.few / 1e3))
}

// FewRowNAX (tensor ops, mtp.patch): the verify's rows on matmul2d, the lanes only decoding.
print("\nFewRowNAX (us, max|diff| against MLX): shape / M=1 qmv, M=2 nax, M=4 nax, M=8 nax")
for (label, n, k, _) in shapes {
  let packed = quantized(
    (MLXRandom.normal([n, k]) * 0.02).asType(.float16), groupSize: 128, bits: 2)
  let (w, scales, biases) = (packed.wq, packed.scales, packed.biases!)
  eval(w, scales, biases)
  var cells: [String] = []
  for m in [1, 2, 4, 8] {
    let xs = (0..<chain).map { _ in MLXRandom.normal([m, k]).asType(.float16) }
    eval(xs)
    if m == 1 {
      let t = time {
        xs.map { quantizedMM($0, w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine) }
      }
      cells.append(String(format: "%.0f", t))
      continue
    }
    guard let got = FewRowNAX.apply(xs[0], w, scales: scales, biases: biases, groupSize: 128, bits: 2) else {
      cells.append("declined"); continue
    }
    let ref = quantizedMM(xs[0], w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
    let diff = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
    let t = time { xs.map { FewRowNAX.apply($0, w, scales: scales, biases: biases, groupSize: 128, bits: 2)! } }
    cells.append(String(format: "%.0f (%.3g)", t, diff))
  }
  print("\(label)  \(cells.joined(separator: "  "))")
}

// MLX alone at larger row counts: from its qmv batch limit on, quantizedMM is qmm, which on a GPU
// with neural accelerators is the NAX (tensor ops) kernel. A verify padded up to such a row count
// pays that kernel's time, whatever the padding.
print("\nMLX quantizedMM by rows (us): shape / M=1 2 4 6 8 12 16 32 64")
for (label, n, k, _) in shapes {
  let packed = quantized(
    (MLXRandom.normal([n, k]) * 0.02).asType(.float16), groupSize: 128, bits: 2)
  let (w, scales, biases) = (packed.wq, packed.scales, packed.biases!)
  eval(w, scales, biases)
  var cells: [String] = []
  for m in [1, 2, 4, 6, 8, 12, 16, 32, 64] {
    let xs = (0..<chain).map { _ in MLXRandom.normal([m, k]).asType(.float16) }
    eval(xs)
    let t = time {
      xs.map {
        quantizedMM($0, w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
      }
    }
    cells.append(String(format: "%.0f", t))
  }
  print("\(label)  \(cells.joined(separator: " "))")
}
