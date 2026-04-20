// gpu_nbody.h - the N-body simulation on the GPU with CUDA.
// Same maths as cpu_nbody.h, but each body gets its own GPU thread.
#pragma once
#include "types.h"

// The three arrays that live in GPU memory.
struct GpuBodies {
    Vec4* pos;   // x, y, z, mass
    Vec4* vel;
    Vec4* acc;
    int   n;
};

GpuBodies gpu_create(int n);                                       // allocate GPU memory
void gpu_upload(GpuBodies& g, const Vec4* pos, const Vec4* vel);   // CPU -> GPU
void gpu_download(const GpuBodies& g, Vec4* pos, Vec4* vel);       // GPU -> CPU
void gpu_destroy(GpuBodies& g);                                    // free GPU memory

// Kernel launches return before the GPU has finished. Call this to wait
// (needed before stopping a timer).
void gpu_wait();

// There are two versions of the force kernel so we can compare them:
//   use_tiled = false : "naive", every thread reads every position straight
//                       from global memory
//   use_tiled = true  : "tiled", each block first copies a chunk of positions
//                       into fast shared memory, then all its threads use it
// block_size is the number of threads per block (for example 256).
void gpu_compute_accel(GpuBodies& g, bool use_tiled, int block_size);

// One time step. Same two methods as on the CPU.
// For gpu_step_kdk, call gpu_compute_accel() once before the first step.
void gpu_step_euler(GpuBodies& g, float dt, bool use_tiled, int block_size);
void gpu_step_kdk(GpuBodies& g, float dt, bool use_tiled, int block_size);
