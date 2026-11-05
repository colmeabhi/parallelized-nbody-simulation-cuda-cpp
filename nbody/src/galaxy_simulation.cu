// galaxy_simulation.cu  —  Time loop, snapshot output and energy diagnostics
//                          for the GPU galaxy simulation.
//
// Initial conditions come from galaxy_ics.cpp; all CUDA work is delegated to
// NBodyGPU (nbody_kernel.cu).  This file never calls the CUDA API directly.
#include "galaxy_simulation.cuh"
#include "cpu_nbody.h"     // to_string(Integrator)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

GalaxySimulation::GalaxySimulation(const SimConfig& cfg) : cfg_(cfg) {}

void GalaxySimulation::init() {
    const GalaxyConfig& g = cfg_.galaxy;
    int n_snap = cfg_.n_steps / cfg_.out_interval;

    printf("━━━  Galaxy N-Body Simulation (CUDA)  ━━━\n");
    printf("  Bodies    : %d  (1 BH + %d disk + %d bulge)\n",
           g.n_bodies, galaxy_n_disk(g), galaxy_n_bulge(g));
    printf("  BH mass   : %.4f  (%.0f%% of M_total=1)\n",
           g.bh_mass_frac, g.bh_mass_frac * 100.f);
    printf("  dt        : %g  |  steps: %d  |  snapshots: %d\n",
           cfg_.dt, cfg_.n_steps, n_snap);
    printf("  Integrator: %s  |  kernel: %s  |  block: %d\n",
           to_string(cfg_.integrator), to_string(cfg_.kernel), cfg_.block_size);
    printf("  Total time: %.1f  (outer orbit ≈ 2π ≈ 6.3 time units)\n\n",
           cfg_.dt * cfg_.n_steps);

    make_galaxy(g, h_pos_, h_vel_);
    gpu_.set_kernel(cfg_.kernel);
    gpu_.set_block_size(cfg_.block_size);
    gpu_.allocate(g.n_bodies);
    gpu_.upload(h_pos_.data(), h_vel_.data());
}

// Format per body: float x, y, z  (mass is constant, not saved).
void GalaxySimulation::write_snapshot(std::ofstream& file) const {
    for (const vec4& p : h_pos_) {
        float xyz[3] = { p.x, p.y, p.z };
        file.write(reinterpret_cast<const char*>(xyz), 3 * sizeof(float));
    }
}

void GalaxySimulation::run() {
    std::ofstream out(cfg_.out_file, std::ios::binary);
    if (!out) {
        fprintf(stderr, "ERROR: cannot open '%s' for writing\n", cfg_.out_file.c_str());
        return;
    }
    const int n = cfg_.galaxy.n_bodies;
    int n_snap = cfg_.n_steps / cfg_.out_interval;
    out.write(reinterpret_cast<const char*>(&n),      sizeof(int));
    out.write(reinterpret_cast<const char*>(&n_snap), sizeof(int));

    FILE* elog = nullptr;
    if (!cfg_.energy_file.empty()) {
        elog = fopen(cfg_.energy_file.c_str(), "w");
        if (elog) fprintf(elog, "step,time,kinetic,potential,total,rel_drift\n");
    }

    const Energy e0 = gpu_.energy();
    double max_drift = 0.0;
    if (elog) fprintf(elog, "0,0,%.10e,%.10e,%.10e,0\n", e0.kinetic, e0.potential, e0.total());

    printf("Running…\n");
    double t_energy = 0.0;
    auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step < cfg_.n_steps; step++) {
        if (cfg_.integrator == Integrator::KDK) gpu_.step_kdk(cfg_.dt);
        else                                    gpu_.step_euler(cfg_.dt);

        if (step % cfg_.out_interval == 0) {
            gpu_.download(h_pos_.data(), h_vel_.data());   // synchronises
            write_snapshot(out);

            auto te0 = std::chrono::steady_clock::now();
            Energy e = gpu_.energy();
            t_energy += std::chrono::duration<double>(std::chrono::steady_clock::now() - te0).count();
            double drift = (e.total() - e0.total()) / std::fabs(e0.total());
            max_drift = std::fmax(max_drift, std::fabs(drift));
            if (elog) fprintf(elog, "%d,%g,%.10e,%.10e,%.10e,%.6e\n", step + 1,
                              (step + 1) * cfg_.dt, e.kinetic, e.potential, e.total(), drift);

            printf("\r  [%3d%%] step %5d / %d   dE/E0 = %+.3e",
                   (step * 100) / cfg_.n_steps, step, cfg_.n_steps, drift);
            fflush(stdout);
        }
    }
    gpu_.sync();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const Energy e1 = gpu_.energy();
    double final_drift = (e1.total() - e0.total()) / std::fabs(e0.total());
    max_drift = std::fmax(max_drift, std::fabs(final_drift));
    if (elog) fclose(elog);

    printf("\r  [100%%] done in %.2f s (%.2f s of it energy diagnostics)%20s\n",
           s, t_energy, "");
    printf("  Energy  : E0 = %.8e   E_end = %.8e\n", e0.total(), e1.total());
    printf("            final dE/E0 = %+.3e   max |dE/E0| = %.3e\n", final_drift, max_drift);

    double mb = (8.0 + (double)n * n_snap * 3 * 4) / 1e6;
    printf("  Output  : %s  (%.1f MB)", cfg_.out_file.c_str(), mb);
    if (!cfg_.energy_file.empty()) printf(", %s", cfg_.energy_file.c_str());
    printf("\n\n  Visualize: python3 visualize.py %s\n", cfg_.out_file.c_str());
}
