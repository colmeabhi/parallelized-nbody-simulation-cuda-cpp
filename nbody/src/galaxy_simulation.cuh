#pragma once
#include <string>
#include <vector>
#include "galaxy_ics.h"
#include "nbody_kernel.cuh"

// ─── SimConfig ───────────────────────────────────────────────────────────────
// Galaxy model (galaxy_ics.h) plus integration / output settings.
struct SimConfig {
    GalaxyConfig galaxy;

    float      dt           = 0.01f;  // timestep (v_c ≈ 1, outer orbit ≈ 6.3)
    int        n_steps      = 5000;   // total number of timesteps
    int        out_interval = 50;     // save a snapshot every N steps
    Integrator integrator   = Integrator::KDK;
    GpuKernel  kernel       = GpuKernel::Tiled;
    int        block_size   = BLOCK_SIZE;

    std::string out_file    = "simulation.bin";
    std::string energy_file = "energy.csv";   // "" disables the energy log
};

// ─── GalaxySimulation ────────────────────────────────────────────────────────
//   GalaxySimulation  →  initial conditions, time loop, output, diagnostics
//   NBodyGPU          →  CUDA memory and kernels
class GalaxySimulation {
public:
    explicit GalaxySimulation(const SimConfig& cfg);

    void init();   // generate ICs + upload to GPU
    void run();    // time integration loop + file output + energy check

private:
    void write_snapshot(std::ofstream& file) const;

    SimConfig         cfg_;
    std::vector<vec4> h_pos_, h_vel_;
    NBodyGPU          gpu_;
};
