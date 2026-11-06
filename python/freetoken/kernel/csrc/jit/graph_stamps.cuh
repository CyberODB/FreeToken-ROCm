// In-graph timestamps for decode profiling (FREETOKEN_GRAPH_STAMPS).
//
// One thread reads the GPU's fixed-rate wall clock and adds the time since the previous
// stamp on the same stream to acc[slot]. Captured into the decode graph at phase
// boundaries, it gives per-phase GPU time with no host involvement on replay.
//
// state layout (int64): [0] = last stamp, [1 .. 1+n) = acc ticks, [1+n .. 1+2n) = counts.
// slot < 0 only sets the last stamp. Deltas above max_delta (idle between requests) are dropped.
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

__global__ void graph_stamp_kernel(int64_t* state, int64_t num_slots, int64_t slot, int64_t max_delta) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
#if FREETOKEN_USE_ROCM
    const int64_t t = static_cast<int64_t>(wall_clock64());
#else
    uint64_t ns;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
    const int64_t t = static_cast<int64_t>(ns);
#endif
    const int64_t last = state[0];
    const int64_t d = t - last;
    if (slot >= 0 && slot < num_slots && last != 0 && d >= 0 && d <= max_delta) {
        state[1 + slot] += d;
        state[1 + num_slots + slot] += 1;
    }
    state[0] = t;
}

inline void graph_stamp(tvm::ffi::TensorView state, int64_t num_slots, int64_t slot, int64_t max_delta) {
    using namespace host;
    LaunchKernel(1, 1, state.device())(
        ::graph_stamp_kernel, static_cast<int64_t*>(state.data_ptr()), num_slots, slot, max_delta);
}

// Ticks per millisecond of the stamp clock on `device_id`.
inline int64_t graph_stamp_ticks_per_ms(int64_t device_id) {
#if FREETOKEN_USE_ROCM
    int khz = 0;
    if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, static_cast<int>(device_id)) != hipSuccess) {
        return 0;
    }
    return static_cast<int64_t>(khz);
#else
    (void)device_id;
    return 1000000;  // %globaltimer counts nanoseconds
#endif
}
