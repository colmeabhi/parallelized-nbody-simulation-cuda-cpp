// bench_gpu.cu  —  Benchmark driver for the CUDA kernels.
//
//   ./nbody_bench_gpu --kernel naive|tiled --n 16384 --block 256 --steps 20
//                     [--warmup 3] [--integrator kdk|euler] [--dt 0.01]
//                     [--energy] [--header]
//
// Times full KDK steps (force kernel + integration kernel) with CUDA events
// and prints one CSV row (schema in cli.h).
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "cli.h"
#include "cpu_nbody.h"
#include "galaxy_ics.h"
#include "nbody_kernel.cuh"

int main(int argc, char** argv) {
    Cli cli(argc, argv);
    if (cli.has("header")) { print_bench_header(stdout); if (!cli.has("n")) return 0; }

    GalaxyConfig g;
    g.n_bodies = (int)cli.num("n", 16384);
    const int   steps  = (int)cli.num("steps", 20);
    const int   warmup = (int)cli.num("warmup", 3);
    const float dt     = (float)cli.real("dt", 0.01);

    NBodyGPU gpu;
    Integrator integ;
    try {
        gpu.set_kernel(parse_gpu_kernel(cli.str("kernel", "tiled")));
        gpu.set_block_size((int)cli.num("block", BLOCK_SIZE));
        integ = parse_integrator(cli.str("integrator", "kdk"));
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    std::vector<vec4> pos, vel;
    make_galaxy(g, pos, vel);
    gpu.allocate(g.n_bodies);
    gpu.upload(pos.data(), vel.data());

    const bool want_energy = cli.has("energy");
    Energy e0;
    if (want_energy) e0 = gpu.energy();

    auto do_step = [&] {
        if (integ == Integrator::KDK) gpu.step_kdk(dt); else gpu.step_euler(dt);
    };
    for (int s = 0; s < warmup; s++) do_step();
    gpu.sync();

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int s = 0; s < steps; s++) do_step();
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    double drift = NAN;
    if (want_energy) {
        Energy e1 = gpu.energy();
        drift = (e1.total() - e0.total()) / std::fabs(e0.total());
    }

    double sec_per_step = ms * 1e-3 / steps;
    double n2 = (double)g.n_bodies * g.n_bodies;
    printf("%s,%s,%d,%d,0,%d,%.6f,%.6e,%.4f,%.6e,nan\n",
           to_string(gpu.kernel()), to_string(integ), g.n_bodies, gpu.block_size(),
           steps, sec_per_step * 1e3, n2 / sec_per_step,
           kFlopsPerInteraction * n2 / sec_per_step * 1e-9, drift);
    return 0;
}
