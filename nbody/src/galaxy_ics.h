// galaxy_ics.h  —  Initial conditions for a simplified Milky-Way-like galaxy.
//                  Pure C++ (no CUDA), shared by every solver.
#pragma once
#include <cstdint>
#include <vector>
#include "common.h"

// Units: G = 1, total mass = 1, length = disk radius = 1.
// v_c ≈ 1 at the disk edge, so one outer orbit takes ≈ 2π ≈ 6.3 time units.
struct GalaxyConfig {
    int      n_bodies       = 4096;   // 1 BH + disk stars + bulge stars
    float    bh_mass_frac   = 0.05f;  // BH mass / total mass
    float    bulge_frac     = 0.10f;  // fraction of bodies in the bulge
    float    disk_radius    = 1.0f;   // outer edge of the stellar disk
    float    disk_thickness = 0.02f;  // vertical scale height of the disk
    uint32_t seed           = 42;     // RNG seed (reproducible ICs)
};

// Fills pos/vel (resized to cfg.n_bodies):
//   body 0                 : central black hole at rest at the origin
//   bodies [1, n_disk]     : thin disk on near-circular orbits
//   bodies [n_disk+1, n)   : dispersion-supported bulge
void make_galaxy(const GalaxyConfig& cfg,
                 std::vector<vec4>& pos, std::vector<vec4>& vel);

int galaxy_n_bulge(const GalaxyConfig& cfg);
int galaxy_n_disk (const GalaxyConfig& cfg);
