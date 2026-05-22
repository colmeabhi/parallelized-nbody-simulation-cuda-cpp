// bench_cpu.cpp - times the CPU version.
//
// Usage:  ./build/bench_cpu <serial|omp> <n_bodies> <n_steps>
// Prints one CSV line:  impl,n,block,ms_per_step,interactions_per_s,gflops
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cpu_nbody.h"
#include "galaxy.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: %s <serial|omp> <n_bodies> <n_steps>\n", argv[0]);
        return 1;
    }
    bool use_openmp = strcmp(argv[1], "omp") == 0;
    int  n          = atoi(argv[2]);
    int  steps      = atoi(argv[3]);
    float dt        = 0.01f;

    std::vector<Vec4> pos(n), vel(n), acc(n);
    make_galaxy(n, pos.data(), vel.data());

    // KDK needs the acceleration before the first step. This call also
    // "warms up" the CPU so the first timed step is not slower than the rest.
    compute_accel_cpu(pos.data(), acc.data(), n, use_openmp);

    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; s++)
        step_kdk_cpu(pos.data(), vel.data(), acc.data(), n, dt, use_openmp);
    auto end = std::chrono::steady_clock::now();

    double seconds_per_step = std::chrono::duration<double>(end - start).count() / steps;
    double interactions     = (double)n * n;   // every body with every body
    printf("%s,%d,0,%.6f,%.4e,%.2f\n",
           use_openmp ? "cpu_openmp" : "cpu_serial", n,
           seconds_per_step * 1000,
           interactions / seconds_per_step,
           interactions * FLOPS_PER_INTERACTION / seconds_per_step / 1e9);
    return 0;
}
