// bench_cpu.cpp  —  Benchmark / validation driver for the CPU solvers.
//
//   ./nbody_bench_cpu --impl serial|omp|bh --n 4096 --steps 10
//                     [--threads T] [--theta 0.5] [--integrator kdk|euler]
//                     [--dt 0.01] [--warmup 1] [--energy] [--force-error]
//                     [--energy-log file.csv --energy-every 10] [--header]
//
// Prints one CSV row (schema in cli.h).  interactions/s and GFLOP/s are
// N²/t_step and 20·N²/t_step; for Barnes–Hut they are "direct-equivalent"
// figures (what a direct code would need to match its time), not work done.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include "cli.h"
#include "cpu_nbody.h"
#include "galaxy_ics.h"
#ifdef _OPENMP
#include <omp.h>
#endif

// RMS relative acceleration error of Barnes–Hut vs exact direct summation,
// estimated on a random sample of bodies.
static double bh_force_error(const std::vector<vec4>& pos, float soft2,
                             float theta, int sample)
{
    int n = (int)pos.size();
    std::vector<vec4> a_bh(n);
    accel_barnes_hut(pos.data(), a_bh.data(), n, soft2, theta);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> pick(0, n - 1);
    sample = std::min(sample, n);
    std::vector<int> idx(sample);
    for (int& i : idx) i = pick(rng);
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int s = 0; s < sample; s++) {
        const int i = idx[s];
        double ax = 0, ay = 0, az = 0;
        for (int j = 0; j < n; j++) {
            double dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y, dz = pos[j].z - pos[i].z;
            double r2 = dx * dx + dy * dy + dz * dz + soft2;
            double sc = pos[j].w / (r2 * std::sqrt(r2));
            ax += dx * sc; ay += dy * sc; az += dz * sc;
        }
        double ex = a_bh[i].x - ax, ey = a_bh[i].y - ay, ez = a_bh[i].z - az;
        double amag2 = ax * ax + ay * ay + az * az;
        if (amag2 > 0) sum += (ex * ex + ey * ey + ez * ez) / amag2;
    }
    return std::sqrt(sum / sample);
}

int main(int argc, char** argv) {
    Cli cli(argc, argv);
    if (cli.has("header")) { print_bench_header(stdout); if (!cli.has("n")) return 0; }

    GalaxyConfig g;
    g.n_bodies = (int)cli.num("n", 4096);
    const CpuMethod  method = parse_cpu_method(cli.str("impl", "omp"));
    const Integrator integ  = parse_integrator(cli.str("integrator", "kdk"));
    const int   steps  = (int)cli.num("steps", 10);
    const int   warmup = (int)cli.num("warmup", 1);
    const float dt     = (float)cli.real("dt", 0.01);
    const float theta  = (float)cli.real("theta", 0.5);
    const float soft2  = kSoftening * kSoftening;

    int threads = 1;
#ifdef _OPENMP
    if (cli.has("threads")) omp_set_num_threads((int)cli.num("threads", 1));
    if (method == CpuMethod::Serial) omp_set_num_threads(1);  // energy/err helpers too
    threads = omp_get_max_threads();
#endif

    std::vector<vec4> pos, vel;
    make_galaxy(g, pos, vel);

    double force_err = NAN;
    if (method == CpuMethod::BarnesHut && cli.has("force-error"))
        force_err = bh_force_error(pos, soft2, theta, (int)cli.num("sample", 512));

    CpuNBody sim(pos, vel, method, soft2, theta);

    const bool want_energy = cli.has("energy") || cli.has("energy-log");
    Energy e0;
    if (want_energy) e0 = sim.energy();

    FILE* elog = nullptr;
    int   every = (int)cli.num("energy-every", 10);
    if (cli.has("energy-log")) {
        elog = std::fopen(cli.str("energy-log", "energy.csv").c_str(), "w");
        if (!elog) { std::perror("energy-log"); return 1; }
        std::fprintf(elog, "impl,integrator,n,dt,step,time,kinetic,potential,total,rel_drift\n");
        std::fprintf(elog, "%s,%s,%d,%g,0,0,%.10e,%.10e,%.10e,0\n", to_string(method),
                     to_string(integ), g.n_bodies, dt, e0.kinetic, e0.potential, e0.total());
    }

    for (int s = 0; s < warmup && !elog; s++) sim.step(integ, dt);

    auto t0 = std::chrono::steady_clock::now();
    for (int s = 1; s <= steps; s++) {
        sim.step(integ, dt);
        if (elog && s % every == 0) {
            Energy e = sim.energy();
            std::fprintf(elog, "%s,%s,%d,%g,%d,%g,%.10e,%.10e,%.10e,%.6e\n",
                         to_string(method), to_string(integ), g.n_bodies, dt, s, s * dt,
                         e.kinetic, e.potential, e.total(),
                         (e.total() - e0.total()) / std::fabs(e0.total()));
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    if (elog) std::fclose(elog);

    double sec_per_step = std::chrono::duration<double>(t1 - t0).count() / steps;
    double n2 = (double)g.n_bodies * g.n_bodies;

    double drift = NAN;
    if (want_energy) {
        Energy e1 = sim.energy();
        drift = (e1.total() - e0.total()) / std::fabs(e0.total());
    }

    std::printf("%s,%s,%d,0,%d,%d,%.6f,%.6e,%.4f,%.6e,%.6e\n",
                to_string(method), to_string(integ), g.n_bodies, threads, steps,
                sec_per_step * 1e3, n2 / sec_per_step,
                kFlopsPerInteraction * n2 / sec_per_step * 1e-9, drift, force_err);
    return 0;
}
