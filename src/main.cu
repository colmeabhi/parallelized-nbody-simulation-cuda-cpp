// main.cu - runs the galaxy simulation on the GPU and saves snapshots.
//
// Usage:  ./build/galaxy_nbody [n_bodies] [n_steps] [kdk|euler] [dt]
// Defaults: 4096 bodies, 5000 steps, kdk, dt = 0.01
//
// Output: simulation.bin (positions every 50 steps, for plots/visualize.py)
// and the energy error printed on screen.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cpu_nbody.h"   // total_energy()
#include "galaxy.h"
#include "gpu_nbody.h"

const int  BLOCK_SIZE     = 256;
const bool USE_TILED      = true;
const int  SNAPSHOT_EVERY = 50;    // save positions every 50 steps

int main(int argc, char** argv) {
    int   n         = (argc > 1) ? atoi(argv[1]) : 4096;
    int   steps     = (argc > 2) ? atoi(argv[2]) : 5000;
    bool  use_euler = (argc > 3) && strcmp(argv[3], "euler") == 0;
    float dt        = (argc > 4) ? atof(argv[4]) : 0.01f;

    printf("Galaxy N-body on the GPU: %d bodies, %d steps, dt = %g, %s\n",
           n, steps, dt, use_euler ? "Euler" : "KDK leapfrog");

    // Build the galaxy on the CPU and copy it to the GPU.
    std::vector<Vec4> pos(n), vel(n);
    make_galaxy(n, pos.data(), vel.data());
    double start_energy = total_energy(pos.data(), vel.data(), n);

    GpuBodies gpu = gpu_create(n);
    gpu_upload(gpu, pos.data(), vel.data());
    gpu_compute_accel(gpu, USE_TILED, BLOCK_SIZE);   // KDK needs this before step 1

    // File format: n, number of snapshots, then x y z for every body in every snapshot.
    FILE* file = fopen("simulation.bin", "wb");
    int n_snapshots = steps / SNAPSHOT_EVERY;
    fwrite(&n, sizeof(int), 1, file);
    fwrite(&n_snapshots, sizeof(int), 1, file);

    auto start = std::chrono::steady_clock::now();

    for (int s = 1; s <= steps; s++) {
        if (use_euler) gpu_step_euler(gpu, dt, USE_TILED, BLOCK_SIZE);
        else           gpu_step_kdk  (gpu, dt, USE_TILED, BLOCK_SIZE);

        if (s % SNAPSHOT_EVERY == 0) {
            gpu_download(gpu, pos.data(), vel.data());
            for (int i = 0; i < n; i++)
                fwrite(&pos[i], sizeof(float), 3, file);   // x, y, z only

            double energy = total_energy(pos.data(), vel.data(), n);
            double error  = (energy - start_energy) / fabs(start_energy);
            printf("\rstep %d / %d   energy error = %+.2e", s, steps, error);
            fflush(stdout);
        }
    }
    gpu_wait();
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    fclose(file);

    gpu_download(gpu, pos.data(), vel.data());
    double end_energy = total_energy(pos.data(), vel.data(), n);
    printf("\nDone in %.2f s (includes saving snapshots and energy checks).\n", seconds);
    printf("Energy: start = %.6f, end = %.6f, relative error = %+.2e\n",
           start_energy, end_energy, (end_energy - start_energy) / fabs(start_energy));
    printf("Wrote simulation.bin. View it with: python3 plots/visualize.py simulation.bin\n");

    gpu_destroy(gpu);
    return 0;
}
