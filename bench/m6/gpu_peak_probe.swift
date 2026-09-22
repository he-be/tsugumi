// M6 prefill G0 probe: dense GEMM ceiling of *this* GPU through Apple's own
// MPSMatrixMultiplication, fp16 and fp32, timed on the GPU clock.
// docs/m6-prefill/04-GATES.md G0-2. `swift bench/m6/gpu_peak_probe.swift [n]`
//
// This is a ceiling for "what Apple's kernels reach here", not what our kernels
// reach. A number from another project is not evidence for this repo; this is.
import Foundation
import Metal
import MetalPerformanceShaders

guard let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue() else {
    print("FAIL: no Metal device"); exit(1)
}
let n = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1]) ?? 4096 : 4096
let encodesPerBuffer = 4
let timedBuffers = 5
print("device: \(device.name)  n: \(n)  encodes/buffer: \(encodesPerBuffer)  timed buffers: \(timedBuffers)")

func run(dataType: MPSDataType, elemSize: Int, label: String) {
    let bytes = n * n * elemSize
    guard let a = device.makeBuffer(length: bytes, options: .storageModeShared),
          let b = device.makeBuffer(length: bytes, options: .storageModeShared),
          let c = device.makeBuffer(length: bytes, options: .storageModeShared) else {
        print("\(label): FAIL buffer"); return
    }
    // Values in [0, 1/64]: row sums stay far from fp16 overflow, no denormals.
    if elemSize == 2 {
        let p = a.contents().bindMemory(to: Float16.self, capacity: n * n)
        let q = b.contents().bindMemory(to: Float16.self, capacity: n * n)
        for i in 0..<(n * n) { p[i] = Float16(Float.random(in: 0..<0.015625)); q[i] = Float16(Float.random(in: 0..<0.015625)) }
    } else {
        let p = a.contents().bindMemory(to: Float.self, capacity: n * n)
        let q = b.contents().bindMemory(to: Float.self, capacity: n * n)
        for i in 0..<(n * n) { p[i] = Float.random(in: 0..<0.015625); q[i] = Float.random(in: 0..<0.015625) }
    }
    let desc = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * elemSize, dataType: dataType)
    let ma = MPSMatrix(buffer: a, descriptor: desc)
    let mb = MPSMatrix(buffer: b, descriptor: desc)
    let mc = MPSMatrix(buffer: c, descriptor: desc)
    let mm = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                     resultRows: n, resultColumns: n, interiorColumns: n, alpha: 1, beta: 0)
    func submit() -> MTLCommandBuffer {
        let cb = queue.makeCommandBuffer()!
        for _ in 0..<encodesPerBuffer { mm.encode(commandBuffer: cb, leftMatrix: ma, rightMatrix: mb, resultMatrix: mc) }
        cb.commit()
        return cb
    }
    // warm-up (clocks, first-use compilation)
    for _ in 0..<3 { submit().waitUntilCompleted() }
    var tflops: [Double] = []
    for _ in 0..<timedBuffers {
        let cb = submit()
        cb.waitUntilCompleted()
        let gpuSeconds = cb.gpuEndTime - cb.gpuStartTime
        let flop = 2.0 * Double(n) * Double(n) * Double(n) * Double(encodesPerBuffer)
        tflops.append(flop / gpuSeconds / 1e12)
    }
    let sorted = tflops.sorted()
    let fmt = tflops.map { String(format: "%.2f", $0) }.joined(separator: " ")
    print("\(label): median \(String(format: "%.2f", sorted[sorted.count / 2])) TFLOP/s  max \(String(format: "%.2f", sorted.last!))  (per buffer: \(fmt))")
}

run(dataType: .float16, elemSize: 2, label: "fp16 GEMM (MPS)")
run(dataType: .float32, elemSize: 4, label: "fp32 GEMM (MPS)")
