// M6 prefill B1 probe: memory bandwidth ceiling of *this* GPU, timed on the GPU
// clock. docs/m6-prefill/04-GATES.md B1 (replaces the nominal 153 GB/s).
// `swift bench/m6/gpu_bw_probe.swift [MiB]`
//
// Four patterns over one private buffer pair: blit copy, a float4 read-only
// reduction, a float4 write-only fill, and a float4 copy kernel. Bytes counted
// are the bytes the pattern must move (copy = read + write). This is what a
// streaming kernel can reach here, not what our kernels reach.
import Foundation
import Metal

guard let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue() else {
    print("FAIL: no Metal device"); exit(1)
}
let mib = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1]) ?? 512 : 512
let bytes = mib << 20
let float4s = bytes / 16
let passesPerBuffer = 4
let timedBuffers = 7
print("device: \(device.name)  buffer: \(mib) MiB x2  passes/buffer: \(passesPerBuffer)  timed buffers: \(timedBuffers)")

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void read_sum(device const float4 *a [[buffer(0)]], device float *out [[buffer(1)]],
                     constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                     uint grid [[threads_per_grid]]) {
    float4 s = 0;
    for (uint i = gid; i < n; i += grid) s += a[i];
    float t = s.x + s.y + s.z + s.w;
    if (t == 12345.0f) out[gid] = t;  // keep the loads live, almost never store
}
kernel void write_fill(device float4 *b [[buffer(0)]], constant uint &n [[buffer(2)]],
                       uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    for (uint i = gid; i < n; i += grid) b[i] = float4(1.0f);
}
kernel void copy4(device const float4 *a [[buffer(0)]], device float4 *b [[buffer(1)]],
                  constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                  uint grid [[threads_per_grid]]) {
    for (uint i = gid; i < n; i += grid) b[i] = a[i];
}
"""
let lib: MTLLibrary
do { lib = try device.makeLibrary(source: src, options: nil) } catch { print("FAIL: compile \(error)"); exit(1) }
func pso(_ name: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
}
guard let a = device.makeBuffer(length: bytes, options: .storageModePrivate),
      let b = device.makeBuffer(length: bytes, options: .storageModePrivate),
      let sink = device.makeBuffer(length: 1 << 24, options: .storageModePrivate) else {
    print("FAIL: buffer"); exit(1)
}
// Touch both buffers once so first-use page mapping is outside the timing.
do {
    let cb = queue.makeCommandBuffer()!
    let bl = cb.makeBlitCommandEncoder()!
    bl.fill(buffer: a, range: 0..<bytes, value: 1)
    bl.fill(buffer: b, range: 0..<bytes, value: 2)
    bl.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}
var n32 = UInt32(float4s)
// Enough threads to saturate, few enough that every thread streams many float4s.
let threads = min(float4s, device.maxThreadsPerThreadgroup.width * 2048)

func measure(_ label: String, movedBytes: Int, encode: (MTLCommandBuffer) -> Void) {
    func submit() -> MTLCommandBuffer {
        let cb = queue.makeCommandBuffer()!
        for _ in 0..<passesPerBuffer { encode(cb) }
        cb.commit(); return cb
    }
    for _ in 0..<3 { submit().waitUntilCompleted() }
    var gbs: [Double] = []
    for _ in 0..<timedBuffers {
        let cb = submit(); cb.waitUntilCompleted()
        if let e = cb.error { print("\(label): FAIL \(e)"); return }
        gbs.append(Double(movedBytes * passesPerBuffer) / (cb.gpuEndTime - cb.gpuStartTime) / 1e9)
    }
    let s = gbs.sorted()
    let all = gbs.map { String(format: "%.1f", $0) }.joined(separator: " ")
    print(String(format: "%@: median %.1f GB/s  max %.1f  min %.1f  [%@]", label, s[s.count / 2], s.last!, s.first!, all))
}
func compute(_ p: MTLComputePipelineState, _ bufs: [MTLBuffer]) -> (MTLCommandBuffer) -> Void {
    { cb in
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(p)
        for (i, buf) in bufs.enumerated() { e.setBuffer(buf, offset: 0, index: i) }
        e.setBytes(&n32, length: 4, index: 2)
        let tg = min(p.maxTotalThreadsPerThreadgroup, 256)
        e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        e.endEncoding()
    }
}
print("threads: \(threads)")
measure("blit copy (read+write)", movedBytes: 2 * bytes) { cb in
    let bl = cb.makeBlitCommandEncoder()!
    bl.copy(from: a, sourceOffset: 0, to: b, destinationOffset: 0, size: bytes)
    bl.endEncoding()
}
measure("kernel read", movedBytes: bytes, encode: compute(pso("read_sum"), [a, sink]))
measure("kernel write", movedBytes: bytes, encode: compute(pso("write_fill"), [b]))
measure("kernel copy (read+write)", movedBytes: 2 * bytes, encode: compute(pso("copy4"), [a, b]))
