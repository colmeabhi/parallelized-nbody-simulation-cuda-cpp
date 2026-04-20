// gpu_nbody.cu - the N-body simulation on the GPU (see gpu_nbody.h).
//
// How a time step is split into kernels:
//
//     accel kernel : reads ALL positions, writes only acc[i]
//     kick kernel  : vel[i] += acc[i] * dt      (touches only body i)
//     drift kernel : pos[i] += vel[i] * dt      (touches only body i)
//
// Why not one big kernel? An earlier version computed the force and then
// updated pos[i] in the same kernel. But GPU blocks run in any order, so one
// block could overwrite positions while another block was still reading them
// to compute forces. That is a race condition: the answer depended on timing.
// Now positions are only written by the drift kernel, and a kernel only starts
// after the previous one has completely finished, so nobody reads a position
// while it is being changed.
#include "gpu_nbody.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Stop with a message if a CUDA call failed.
static void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(error));
        exit(1);
    }
}

// Adds the pull of body "other" on body "me" to (ax, ay, az).
// Same formula as the CPU version. __device__ means it runs on the GPU.
__device__ void add_gravity(Vec4 me, Vec4 other, float& ax, float& ay, float& az) {
    float dx = other.x - me.x;
    float dy = other.y - me.y;
    float dz = other.z - me.z;

    float dist_squared = dx * dx + dy * dy + dz * dz + SOFTENING_SQUARED;
    float inv_dist     = rsqrtf(dist_squared);        // 1 / sqrt(...)
    float inv_dist3    = inv_dist * inv_dist * inv_dist;

    float strength = other.w * inv_dist3;             // other.w is its mass
    ax += dx * strength;
    ay += dy * strength;
    az += dz * strength;
}

// ---------------------------------------------------------------------------
// Force kernel, naive version. One thread per body.
// Thread i loops over all n bodies and reads each one from global memory.
// ---------------------------------------------------------------------------
__global__ void accel_naive_kernel(const Vec4* pos, Vec4* acc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // which body am I?
    if (i >= n) return;                              // extra threads in the last block

    Vec4  me = pos[i];
    float ax = 0, ay = 0, az = 0;

    for (int j = 0; j < n; j++)
        add_gravity(me, pos[j], ax, ay, az);

    acc[i].x = ax;
    acc[i].y = ay;
    acc[i].z = az;
}

// ---------------------------------------------------------------------------
// Force kernel, tiled version. One thread per body.
//
// The bodies are cut into "tiles" of blockDim.x bodies. For each tile:
//   1. every thread in the block copies ONE body of the tile into shared
//      memory, so together the block loads the whole tile
//   2. every thread then loops over the tile in shared memory
//
// Shared memory sits on the chip next to the cores, so step 2 is fast, and
// each position is fetched from global memory once per block instead of once
// per thread.
// ---------------------------------------------------------------------------
__global__ void accel_tiled_kernel(const Vec4* pos, Vec4* acc, int n) {
    extern __shared__ Vec4 tile[];   // blockDim.x entries, size given at launch

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Threads past the end (i >= n) must NOT return early: they still have to
    // help load tiles and reach every __syncthreads(). They just use a dummy
    // body and never store a result.
    Vec4 me = { 0, 0, 0, 0 };
    if (i < n) me = pos[i];
    float ax = 0, ay = 0, az = 0;

    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        // 1. Load my one body of this tile. Past the end we load a body with
        //    zero mass, which pulls on nobody.
        int j = tile_start + threadIdx.x;
        if (j < n) tile[threadIdx.x] = pos[j];
        else       tile[threadIdx.x] = { 0, 0, 0, 0 };

        __syncthreads();   // wait until the whole tile is loaded

        // 2. Use the tile.
        for (int k = 0; k < blockDim.x; k++)
            add_gravity(me, tile[k], ax, ay, az);

        __syncthreads();   // wait until everyone is done before loading the next tile
    }

    if (i < n) {
        acc[i].x = ax;
        acc[i].y = ay;
        acc[i].z = az;
    }
}

// "Kick": change the velocity using the acceleration.
__global__ void kick_kernel(Vec4* vel, const Vec4* acc, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vel[i].x += acc[i].x * dt;
    vel[i].y += acc[i].y * dt;
    vel[i].z += acc[i].z * dt;
}

// "Drift": move the position using the velocity.
__global__ void drift_kernel(Vec4* pos, const Vec4* vel, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    pos[i].x += vel[i].x * dt;
    pos[i].y += vel[i].y * dt;
    pos[i].z += vel[i].z * dt;
}

// ---------------------------------------------------------------------------
// Functions called from the CPU side.
// ---------------------------------------------------------------------------

GpuBodies gpu_create(int n) {
    GpuBodies g;
    g.n = n;
    check(cudaMalloc(&g.pos, n * sizeof(Vec4)), "cudaMalloc pos");
    check(cudaMalloc(&g.vel, n * sizeof(Vec4)), "cudaMalloc vel");
    check(cudaMalloc(&g.acc, n * sizeof(Vec4)), "cudaMalloc acc");
    return g;
}

void gpu_upload(GpuBodies& g, const Vec4* pos, const Vec4* vel) {
    check(cudaMemcpy(g.pos, pos, g.n * sizeof(Vec4), cudaMemcpyHostToDevice), "upload pos");
    check(cudaMemcpy(g.vel, vel, g.n * sizeof(Vec4), cudaMemcpyHostToDevice), "upload vel");
}

void gpu_download(const GpuBodies& g, Vec4* pos, Vec4* vel) {
    // cudaMemcpy waits for running kernels to finish first.
    check(cudaMemcpy(pos, g.pos, g.n * sizeof(Vec4), cudaMemcpyDeviceToHost), "download pos");
    check(cudaMemcpy(vel, g.vel, g.n * sizeof(Vec4), cudaMemcpyDeviceToHost), "download vel");
}

void gpu_destroy(GpuBodies& g) {
    cudaFree(g.pos);
    cudaFree(g.vel);
    cudaFree(g.acc);
}

void gpu_wait() {
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

// Enough blocks to give every body a thread (rounded up).
static int blocks_needed(int n, int block_size) {
    return (n + block_size - 1) / block_size;
}

void gpu_compute_accel(GpuBodies& g, bool use_tiled, int block_size) {
    int blocks = blocks_needed(g.n, block_size);
    if (use_tiled) {
        // The third launch argument is the shared memory per block, in bytes:
        // room for one tile of block_size bodies.
        int shared_bytes = block_size * sizeof(Vec4);
        accel_tiled_kernel<<<blocks, block_size, shared_bytes>>>(g.pos, g.acc, g.n);
    } else {
        accel_naive_kernel<<<blocks, block_size>>>(g.pos, g.acc, g.n);
    }
    check(cudaGetLastError(), "accel kernel launch");
}

void gpu_step_euler(GpuBodies& g, float dt, bool use_tiled, int block_size) {
    int blocks = blocks_needed(g.n, block_size);
    gpu_compute_accel(g, use_tiled, block_size);
    kick_kernel <<<blocks, block_size>>>(g.vel, g.acc, g.n, dt);
    drift_kernel<<<blocks, block_size>>>(g.pos, g.vel, g.n, dt);
}

void gpu_step_kdk(GpuBodies& g, float dt, bool use_tiled, int block_size) {
    int blocks = blocks_needed(g.n, block_size);
    kick_kernel <<<blocks, block_size>>>(g.vel, g.acc, g.n, dt / 2);   // half kick
    drift_kernel<<<blocks, block_size>>>(g.pos, g.vel, g.n, dt);       // full drift
    gpu_compute_accel(g, use_tiled, block_size);                       // new acceleration
    kick_kernel <<<blocks, block_size>>>(g.vel, g.acc, g.n, dt / 2);   // half kick
}
