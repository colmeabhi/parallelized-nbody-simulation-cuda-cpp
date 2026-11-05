#pragma once
#include <string>
#include <cuda_runtime.h>
#include "common.h"

// Default threads per block.  Any multiple of 32 in [32, 1024] works; the
// block-size sweep in scripts/run_benchmarks.sh tries 64/128/256/512.
#define BLOCK_SIZE  256

// Kept for backwards compatibility with older code; same value as kSoftening.
#define SOFTENING   kSoftening

enum class GpuKernel {
    Naive,   // every thread streams all N positions from global memory
    Tiled    // positions staged through shared memory, blockDim bodies/tile
};
GpuKernel   parse_gpu_kernel(const std::string& s);   // "naive" | "tiled"
const char* to_string(GpuKernel k);

// ─── NBodyGPU ───────────────────────────────────────────────────────────────
// Device state: pos (x,y,z,m), vel (vx,vy,vz,0), acc (ax,ay,az,0).
//
// Each timestep is split into kernels so that *no kernel writes the array
// another thread of the same kernel is reading*:
//
//   kick_drift_kernel : v += a·dt/2 ; x += v·dt        (touches only body i)
//   accel_*_kernel    : a  = F(x)   ; v += a·dt/2      (reads ALL x, writes
//                                                       only acc[i], vel[i])
//
// The kernel-launch boundary is the grid-wide barrier: every block has
// finished reading the old positions before any position is updated.  The
// original single fused kernel wrote pos[i] while other blocks were still
// loading tiles of pos[], so results depended on block scheduling.
// ────────────────────────────────────────────────────────────────────────────
class NBodyGPU {
public:
    NBodyGPU() = default;
    ~NBodyGPU() { free(); }
    NBodyGPU(const NBodyGPU&) = delete;
    NBodyGPU& operator=(const NBodyGPU&) = delete;

    void allocate(int n);
    void upload(const vec4* pos, const vec4* vel);   // also invalidates acc
    void download(vec4* pos, vec4* vel) const;       // synchronises
    void free();

    void set_kernel(GpuKernel k)  { kernel_ = k; }
    void set_block_size(int b);                      // multiple of 32, ≤ 1024

    // Kick-drift-kick leapfrog (one force evaluation per step).
    void step_kdk(float dt);
    // Semi-implicit Euler (the original scheme), for comparison only.
    void step_euler(float dt);
    // Backwards-compatible alias for step_kdk.
    void step(float dt) { step_kdk(dt); }

    // acc = F(pos) only — used to time the force kernel in isolation.
    void compute_accel();

    // Total energy in double precision (potential computed on the GPU).
    Energy energy();

    void sync() const;
    int  n() const { return n_; }
    int  block_size() const { return block_; }
    GpuKernel kernel() const { return kernel_; }

private:
    void launch_accel(float kick_dt);   // kick_dt = 0 → no velocity update

    float4* d_pos = nullptr;
    float4* d_vel = nullptr;
    float4* d_acc = nullptr;
    double* d_phi = nullptr;            // per-body potential (energy only)
    int     n_     = 0;
    int     block_ = BLOCK_SIZE;
    GpuKernel kernel_ = GpuKernel::Tiled;
    bool    acc_valid_ = false;
};
