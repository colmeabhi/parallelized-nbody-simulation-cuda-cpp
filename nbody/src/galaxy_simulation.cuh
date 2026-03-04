#pragma once
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "nbody_kernel.cuh"

// ─── GalaxyConfig ────────────────────────────────────────────────────────────
// All tunable simulation parameters in one place.
// Units: G = 1, length = disk_radius, mass = individual star mass = 1.
// ────────────────────────────────────────────────────────────────────────────
struct GalaxyConfig {
    // Population
    int   n_bodies     = 4096;   // total bodies (1 BH + disk stars + bulge stars)
    float bh_mass_frac = 0.05f;  // BH mass as fraction of total system mass
    float bulge_frac   = 0.10f;  // fraction of stars placed in the central bulge

    // Geometry
    float disk_radius    = 1.0f;  // outer edge of the stellar disk
    float disk_thickness = 0.02f; // scale height of the disk (thin = realistic)

    // Integration
    float dt           = 0.01f;  // timestep (in simulation units; v_c ≈ 1)
    int   n_steps      = 5000;   // total number of timesteps
    int   out_interval = 50;     // save a snapshot every N steps

    // Output
    std::string out_file = "simulation.bin";
};

// ─── GalaxySimulation ────────────────────────────────────────────────────────
// Owns the initial conditions (host-side) and delegates force computation
// to NBodyGPU.  The two concerns are kept strictly separate:
//
//   GalaxySimulation  →  knows about stars, orbits, disk geometry
//   NBodyGPU          →  knows about CUDA, shared-memory tiles, kernels
//
// Typical usage:
//   GalaxySimulation sim(cfg);
//   sim.init();   // build initial conditions, upload to GPU
//   sim.run();    // integrate and write output file
// ────────────────────────────────────────────────────────────────────────────
class GalaxySimulation {
public:
    explicit GalaxySimulation(const GalaxyConfig& cfg);
    ~GalaxySimulation();

    void init();   // generate ICs + upload to GPU
    void run();    // time integration loop + file output

private:
    // ── Initial condition builders ───────────────────────────────────────
    void setup_black_hole();  // body 0: massive central BH at origin
    void setup_disk();        // bodies [1, n_disk]: thin rotating stellar disk
    void setup_bulge();       // bodies [n_disk+1, n-1]: dispersion-supported bulge

    // ── Output helper ────────────────────────────────────────────────────
    void write_snapshot(std::ofstream& file) const;  // write xyz to binary file

    // ── Data ─────────────────────────────────────────────────────────────
    GalaxyConfig        cfg_;
    std::vector<float4> h_pos_;  // host positions  (x, y, z, mass)
    std::vector<float4> h_vel_;  // host velocities (vx, vy, vz, 0)
    NBodyGPU            gpu_;    // GPU solver (the only CUDA-aware component)
};
