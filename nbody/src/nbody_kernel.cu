// nbody_kernel.cu  —  Core CUDA N-body force calculation and integration.
//
// Algorithm: O(N²) direct summation using shared-memory tiling.
//
// Each block loads BLOCK_SIZE body positions into shared memory (one "tile"),
// then every thread accumulates the gravitational force from every body in
// that tile before moving to the next tile.  This way each global-memory
// position is read once per tile instead of once per body → BLOCK_SIZE-fold
// reduction in global memory traffic.
//
// Integration: leapfrog (Kick-Drift)
//   v(t+dt) = v(t) + a(t)*dt
//   x(t+dt) = x(t) + v(t+dt)*dt
//
// Units: G = 1 (gravitational constant absorbed into masses).

#include "nbody_kernel.cuh"
#include <cstdio>
#include <cstdlib>

// ─── Error-checking helper ───────────────────────────────────────────────────
#define CUDA_CHECK(err)                                                    \
    do {                                                                   \
        if ((err) != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// ─── Device helper: gravitational acceleration from body j onto body i ───────
//
//   a_ij = G * m_j * (r_j - r_i) / |r_j - r_i|^3        (G = 1)
//
// soft2 = ε² is added to |r|² to prevent the singularity at r = 0.
// For a self-interaction (i == j) the displacement is (0,0,0), so a = 0. ✓
// For padded bodies (mass == 0) the result is 0. ✓
// ────────────────────────────────────────────────────────────────────────────
__device__ inline float3 body_body_accel(float4 pi, float4 pj, float soft2)
{
    float dx = pj.x - pi.x;
    float dy = pj.y - pi.y;
    float dz = pj.z - pi.z;

    float r2     = dx*dx + dy*dy + dz*dz + soft2;
    float inv_r  = rsqrtf(r2);                    // 1 / sqrt(r² + ε²)
    float scale  = pj.w * inv_r * inv_r * inv_r;  // m_j / |r|³

    return { dx * scale, dy * scale, dz * scale };
}

// ─── Main kernel ─────────────────────────────────────────────────────────────
//
// One thread handles one body (index i).
// Shared memory 'tile' holds BLOCK_SIZE positions for the current tile.
//
// Loop over tiles:
//   1. Cooperatively load tile into shared memory  (1 global load per thread)
//   2. Compute forces from every body in the tile  (reads from L1/shared mem)
//   3. Advance to next tile
//
// After accumulating all forces, apply leapfrog integration.
// ────────────────────────────────────────────────────────────────────────────
__global__ void nbody_step_kernel(
    float4* __restrict__ pos,   // in/out: body positions  (x, y, z, mass)
    float4* __restrict__ vel,   // in/out: body velocities (vx, vy, vz, 0)
    int   n,                    // total number of bodies
    float dt,                   // timestep
    float soft2)                // softening squared (ε²)
{
    // Dynamic shared memory: one tile of positions.
    extern __shared__ float4 tile[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load this body's position (or a zero dummy if i ≥ n).
    float4 pi  = (i < n) ? pos[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    float3 acc = { 0.f, 0.f, 0.f };

    int n_tiles = (n + blockDim.x - 1) / blockDim.x;

    for (int t = 0; t < n_tiles; t++) {
        // ── Step 1: load tile ──────────────────────────────────────────────
        // Each thread loads body  j = t*BLOCK_SIZE + threadIdx.x  (or zero).
        int j = t * blockDim.x + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? pos[j] : make_float4(0.f, 0.f, 0.f, 0.f);
        __syncthreads();  // wait for all threads to finish loading

        // ── Step 2: accumulate forces from this tile ───────────────────────
        // #pragma unroll 4 tells the compiler to unroll 4 iterations at a time,
        // improving instruction-level parallelism with minimal register cost.
        #pragma unroll 4
        for (int k = 0; k < blockDim.x; k++) {
            float3 a = body_body_accel(pi, tile[k], soft2);
            acc.x += a.x;
            acc.y += a.y;
            acc.z += a.z;
        }
        __syncthreads();  // ensure tile is fully consumed before next load
    }

    // ── Leapfrog (Kick-Drift) integration ────────────────────────────────
    // Kick: v  += a * dt
    // Drift: x += v * dt
    if (i < n) {
        float4 v  = vel[i];
        v.x += acc.x * dt;
        v.y += acc.y * dt;
        v.z += acc.z * dt;
        vel[i]   = v;

        pos[i].x += v.x * dt;
        pos[i].y += v.y * dt;
        pos[i].z += v.z * dt;
    }
}

// ─── NBodyGPU host-side implementation ───────────────────────────────────────

void NBodyGPU::allocate(int n) {
    n_ = n;
    CUDA_CHECK(cudaMalloc(&d_pos, n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_vel, n * sizeof(float4)));
}

void NBodyGPU::upload(const float4* pos, const float4* vel) {
    CUDA_CHECK(cudaMemcpy(d_pos, pos, n_ * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, vel, n_ * sizeof(float4), cudaMemcpyHostToDevice));
}

void NBodyGPU::download(float4* pos, float4* vel) const {
    CUDA_CHECK(cudaMemcpy(pos, d_pos, n_ * sizeof(float4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel, d_vel, n_ * sizeof(float4), cudaMemcpyDeviceToHost));
}

void NBodyGPU::step(float dt) {
    int   blocks   = (n_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int   smem     = BLOCK_SIZE * sizeof(float4);  // shared memory per block
    float soft2    = SOFTENING * SOFTENING;

    nbody_step_kernel<<<blocks, BLOCK_SIZE, smem>>>(d_pos, d_vel, n_, dt, soft2);
    CUDA_CHECK(cudaGetLastError());
    // Note: kernel is async; download() or cudaDeviceSynchronize() will flush it.
}

void NBodyGPU::free() {
    if (d_pos) { cudaFree(d_pos); d_pos = nullptr; }
    if (d_vel) { cudaFree(d_vel); d_vel = nullptr; }
    n_ = 0;
}
