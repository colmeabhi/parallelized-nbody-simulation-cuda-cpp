#pragma once
#include <cuda_runtime.h>

// ─── Kernel tuning constants ────────────────────────────────────────────────
// Threads per block – must be a power of 2; 256 is a good default.
#define BLOCK_SIZE  256

// Gravitational softening ε: prevents 1/r singularity when r → 0.
// In simulation units (total-mass = N, length = disk radius = 1).
#define SOFTENING   0.02f

// ─── NBodyGPU ───────────────────────────────────────────────────────────────
// Manages GPU memory and runs one leapfrog timestep via a tiled,
// shared-memory force kernel.
//
// Memory layout (float4 = 4 × float, 16 bytes, loads/stores in one transaction):
//   pos[i] = { x, y, z, mass }    – position + mass
//   vel[i] = { vx, vy, vz, 0  }  – velocity  (w unused)
// ────────────────────────────────────────────────────────────────────────────
class NBodyGPU {
public:
    NBodyGPU() : d_pos(nullptr), d_vel(nullptr), n_(0) {}

    // Allocate device buffers for n bodies.
    void allocate(int n);

    // Copy initial conditions from host arrays to device.
    void upload(const float4* pos, const float4* vel);

    // Copy current state from device back to host (for output).
    void download(float4* pos, float4* vel) const;

    // Run one integration step: compute forces then update positions/velocities.
    void step(float dt);

    // Release device memory.
    void free();

    int n() const { return n_; }

private:
    float4* d_pos;   // device: positions (x, y, z, mass)
    float4* d_vel;   // device: velocities (vx, vy, vz, 0)
    int     n_;      // number of bodies
};
