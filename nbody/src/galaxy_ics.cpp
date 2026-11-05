// galaxy_ics.cpp  —  Simplified Milky-Way-like initial conditions.
//
// N-body units: G = 1, total system mass = 1, characteristic length = 1.
//   star_mass = (1 − bh_mass_frac) / n_stars
//   bh_mass   = bh_mass_frac
#include "galaxy_ics.h"
#include <cmath>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

struct Rng {
    std::mt19937 gen;
    explicit Rng(uint32_t seed) : gen(seed) {}
    float uniform(float lo, float hi) {
        return std::uniform_real_distribution<float>(lo, hi)(gen);
    }
    float normal(float mean, float sigma) {
        return std::normal_distribution<float>(mean, sigma)(gen);
    }
};

// Circular orbit speed at radius r for enclosed mass M_enc (G = 1).
float v_circ(float r, float M_enc) {
    return (r > 1e-6f) ? std::sqrt(M_enc / r) : 0.f;
}

// Mass enclosed within r: central BH + uniform-surface-density disk of
// total mass M_disk and outer radius R.
float mass_enclosed(float r, float M_bh, float M_disk, float R) {
    float frac = (r < R) ? (r / R) * (r / R) : 1.f;
    return M_bh + M_disk * frac;
}

} // namespace

int galaxy_n_bulge(const GalaxyConfig& cfg) {
    return (int)(cfg.n_bodies * cfg.bulge_frac);
}
int galaxy_n_disk(const GalaxyConfig& cfg) {
    return cfg.n_bodies - 1 - galaxy_n_bulge(cfg);
}

void make_galaxy(const GalaxyConfig& cfg,
                 std::vector<vec4>& pos, std::vector<vec4>& vel)
{
    const int   n         = cfg.n_bodies;
    const int   n_disk    = galaxy_n_disk(cfg);
    const float M_bh      = cfg.bh_mass_frac;
    const float M_disk    = 1.0f - cfg.bh_mass_frac;      // total stellar mass
    const float star_mass = M_disk / (float)(n - 1);
    const float R         = cfg.disk_radius;
    const float R_bulge   = R * 0.12f;

    pos.assign(n, make_vec4(0, 0, 0, 0));
    vel.assign(n, make_vec4(0, 0, 0, 0));
    Rng rng(cfg.seed);

    // ── Body 0: central black hole ────────────────────────────────────────
    pos[0] = make_vec4(0.f, 0.f, 0.f, M_bh);

    // ── Disk: uniform surface density, r = R·sqrt(U), r ∈ [0.05R, R] ──────
    for (int i = 1; i <= n_disk; i++) {
        float r     = R * std::sqrt(rng.uniform(0.0025f, 1.f));
        float theta = rng.uniform(0.f, (float)(2.0 * M_PI));
        float z     = rng.normal(0.f, cfg.disk_thickness);
        pos[i] = make_vec4(r * std::cos(theta), r * std::sin(theta), z, star_mass);

        float vc   = v_circ(r, mass_enclosed(r, M_bh, M_disk, R));
        float disp = vc * 0.04f;                          // 4 % dispersion
        vel[i] = make_vec4(-vc * std::sin(theta) + rng.normal(0.f, disp),
                            vc * std::cos(theta) + rng.normal(0.f, disp),
                            rng.normal(0.f, disp * 0.2f),
                            0.f);
    }

    // ── Bulge: uniform sphere, isotropic velocities ~ local virial speed ──
    for (int i = n_disk + 1; i < n; i++) {
        float r   = R_bulge * std::cbrt(rng.uniform(1e-3f, 1.f));
        float th  = rng.uniform(0.f, (float)(2.0 * M_PI));
        float phi = std::acos(rng.uniform(-1.f, 1.f));
        pos[i] = make_vec4(r * std::sin(phi) * std::cos(th),
                           r * std::sin(phi) * std::sin(th),
                           r * std::cos(phi),
                           star_mass);

        float v_vir = v_circ(r, mass_enclosed(r, M_bh, M_disk, R));
        float vmag  = std::fabs(rng.normal(0.f, v_vir * 0.5f));
        float vth   = rng.uniform(0.f, (float)(2.0 * M_PI));
        float vphi  = std::acos(rng.uniform(-1.f, 1.f));
        vel[i] = make_vec4(vmag * std::sin(vphi) * std::cos(vth),
                           vmag * std::sin(vphi) * std::sin(vth),
                           vmag * std::cos(vphi),
                           0.f);
    }
}
