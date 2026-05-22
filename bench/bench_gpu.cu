// bench_gpu.cu - times the GPU version.
//
// Usage:  ./build/bench_gpu <naive|tiled> <n_bodies> <block_size> <n_steps>
// Prints one CSV line:  impl,n,block,ms_per_step,interactions_per_s,gflops
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "galaxy.h"
#include "gpu_nbody.h"

int main(int argc, char** argv) {
    if (argc != 5) {
        printf("usage: %s <naive|tiled> <n_bodies> <block_size> <n_steps>\n", argv[0]);
        return 1;
    }
    bool use_tiled  = strcmp(argv[1], "tiled") == 0;
    int  n          = atoi(argv[2]);
    int  block_size = atoi(argv[3]);
    int  steps      = atoi(argv[4]);
    float dt        = 0.01f;

    std::vector<Vec4> pos(n), vel(n);
    make_galaxy(n, pos.data(), vel.data());

    GpuBodies gpu = gpu_create(n);
    gpu_upload(gpu, pos.data(), vel.data());

    // Warm up: the first kernel launches are slower (the GPU has to load the
    // code), so run a few steps before starting the timer.
    gpu_compute_accel(gpu, use_tiled, block_size);
    for (int s = 0; s < 3; s++)
        gpu_step_kdk(gpu, dt, use_tiled, block_size);
    gpu_wait();

    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; s++)
        gpu_step_kdk(gpu, dt, use_tiled, block_size);
    gpu_wait();   // kernels run in the background: wait before stopping the clock
    auto end = std::chrono::steady_clock::now();

    double seconds_per_step = std::chrono::duration<double>(end - start).count() / steps;
    double interactions     = (double)n * n;
    printf("%s,%d,%d,%.6f,%.4e,%.2f\n",
           use_tiled ? "gpu_tiled" : "gpu_naive", n, block_size,
           seconds_per_step * 1000,
           interactions / seconds_per_step,
           interactions * FLOPS_PER_INTERACTION / seconds_per_step / 1e9);

    gpu_destroy(gpu);
    return 0;
}
