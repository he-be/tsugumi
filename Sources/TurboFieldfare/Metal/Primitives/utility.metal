#include <metal_stdlib>
using namespace metal;

// Kept in the shared library so both INT4 and INT8 shared-expert paths use
// the same Gemma activation without compiling a private shader module.
[[kernel, max_total_threads_per_threadgroup(256)]]
void gelu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half(gelu_pytorch_tanh(g) * u);
}

// In-place elementwise scale for the MTP drafter's layer tail
// (`hidden *= layer_scalar`). The drafter's sandwich residual otherwise
// reuses the vision tower's norm-residual join, so this is the only piece
// of its layer loop without an existing kernel.
[[kernel, max_total_threads_per_threadgroup(256)]]
void scale_inplace_fp16(
    device half*        x     [[buffer(0)]],
    constant float&     scale [[buffer(1)]],
    constant uint&      count [[buffer(2)]],
    uint                tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    x[tid] = half(float(x[tid]) * scale);
}
