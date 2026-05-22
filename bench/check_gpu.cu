// check_gpu.cu - makes sure the GPU gives the same forces as the CPU.
//
// Usage:  ./build/check_gpu
// Computes the acceleration of every body three ways (CPU, GPU naive, GPU
// tiled) and prints the biggest difference. Uses a body count that is NOT a
// multiple of the block size, to also test the half-empty last block.
#include <cmath>
#include <cstdio>
#include <vector>
#include "cpu_nbody.h"
#include "galaxy.h"
#include "gpu_nbody.h"

// Largest difference between two acceleration arrays, relative to the
// largest acceleration.
static double biggest_difference(const std::vector<Vec4>& a, const std::vector<Vec4>& b) {
    double biggest_diff = 0, biggest_value = 0;
    for (size_t i = 0; i < a.size(); i++) {
        biggest_diff  = fmax(biggest_diff, fabs(a[i].x - b[i].x));
        biggest_diff  = fmax(biggest_diff, fabs(a[i].y - b[i].y));
        biggest_diff  = fmax(biggest_diff, fabs(a[i].z - b[i].z));
        biggest_value = fmax(biggest_value, fabs(a[i].x));
    }
    return biggest_diff / biggest_value;
}

int main() {
    const int n = 5000, block_size = 256;

    std::vector<Vec4> pos(n), vel(n), cpu_acc(n), gpu_acc(n), unused(n);
    make_galaxy(n, pos.data(), vel.data());
    compute_accel_cpu(pos.data(), cpu_acc.data(), n, true);

    GpuBodies gpu = gpu_create(n);
    gpu_upload(gpu, pos.data(), vel.data());

    bool all_ok = true;
    for (int use_tiled = 0; use_tiled <= 1; use_tiled++) {
        gpu_compute_accel(gpu, use_tiled, block_size);

        // gpu_download copies pos and vel. To read back the accelerations we
        // temporarily point "vel" at the acc array.
        GpuBodies view = gpu;
        view.vel = gpu.acc;
        gpu_download(view, unused.data(), gpu_acc.data());

        double diff = biggest_difference(cpu_acc, gpu_acc);
        bool   ok   = diff < 1e-3;   // float rounding differs a little between CPU and GPU
        printf("%-5s kernel vs CPU: biggest relative difference = %.2e  %s\n",
               use_tiled ? "tiled" : "naive", diff, ok ? "OK" : "MISMATCH");
        all_ok = all_ok && ok;
    }

    gpu_destroy(gpu);
    return all_ok ? 0 : 1;
}
