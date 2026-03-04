// galaxy_simulation.cu  —  Simplified Milky Way-like galaxy initial conditions
//                          and time-integration loop.
//
// ── Unit system ────────────────────────────────────────────────────────────
// N-body units: G = 1, total system mass = 1, characteristic length = 1.
//   • star_mass = (1 − bh_mass_frac) / n_stars       ← tiny, sums to (1−f)
//   • bh_mass   = bh_mass_frac                        ← 5 % of total
//   • M_total   = 1.0
//
// This gives circular velocity v_c ≈ 1 at r = disk_radius, so one full
// outer-disk orbit takes T ≈ 2π ≈ 6.3 time units.
// With dt = 0.01 and n_steps = 5000, we integrate for 50 time units ≈ 8 orbits.
//
// ── Galaxy model ────────────────────────────────────────────────────────────
//   Body 0          : supermassive black hole (Sgr A* analogue) at origin.
//   Bodies [1, N_d] : thin rotating stellar disk, uniform surface density.
//   Bodies [N_d, N) : dispersion-supported central bulge.
//
// Force computation is entirely delegated to NBodyGPU (nbody_kernel.cu).
// This file never calls any CUDA API directly.

#include "galaxy_simulation.cuh"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── Random number helpers (seeded for reproducibility) ──────────────────────
static std::mt19937 rng(42);

static float randf(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng);
}
static float randg(float mean, float sigma) {
    return std::normal_distribution<float>(mean, sigma)(rng);
}

// ─── Physics helpers ─────────────────────────────────────────────────────────

// Circular orbit speed at radius r given enclosed mass M_enc (G = 1).
static float v_circ(float r, float M_enc) {
    return (r > 1e-6f) ? sqrtf(M_enc / r) : 0.f;
}

// Mass enclosed within radius r for a uniform-surface-density stellar disk
// of total mass M_disk and outer radius R, plus a central black hole M_bh.
static float mass_enclosed(float r, float M_bh, float M_disk, float R) {
    float frac = (r < R) ? (r / R) * (r / R) : 1.f;
    return M_bh + M_disk * frac;
}

// ─── Constructor / Destructor ────────────────────────────────────────────────
GalaxySimulation::GalaxySimulation(const GalaxyConfig& cfg)
    : cfg_(cfg), h_pos_(cfg.n_bodies), h_vel_(cfg.n_bodies)
{}

GalaxySimulation::~GalaxySimulation() {
    gpu_.free();
}

// ─── Initial condition builders ───────────────────────────────────────────────

// Body 0: central black hole at origin.
// In N-body units (M_total = 1): M_bh = bh_mass_frac.
void GalaxySimulation::setup_black_hole() {
    float M_bh = cfg_.bh_mass_frac;      // e.g., 0.05 (5 % of total mass)
    h_pos_[0]  = make_float4(0.f, 0.f, 0.f, M_bh);
    h_vel_[0]  = make_float4(0.f, 0.f, 0.f, 0.f);
}

// Disk stars: placed in a thin disk on near-circular orbits.
// Surface density is uniform: r sampled as r = R * sqrt(U) where U ~ Uniform[0,1].
void GalaxySimulation::setup_disk() {
    int   n_stars  = cfg_.n_bodies - 1;
    int   n_bulge  = (int)(cfg_.n_bodies * cfg_.bulge_frac);
    int   n_disk   = cfg_.n_bodies - 1 - n_bulge;   // bodies [1 .. n_disk]

    // N-body unit masses: M_bh + M_disk_total = 1
    float M_bh     = cfg_.bh_mass_frac;
    float M_disk   = 1.0f - cfg_.bh_mass_frac;      // total stellar mass
    float star_mass = M_disk / (float)n_stars;       // individual star mass
    float R         = cfg_.disk_radius;

    for (int i = 1; i <= n_disk; i++) {
        // Uniform surface density: r ~ R * sqrt(U),   theta ~ Uniform[0, 2π].
        // Minimum r = 0.05*R avoids very tight inner orbits with few steps/period.
        float r     = R * sqrtf(randf(0.0025f, 1.f));   // r in [0.05R, R]
        float theta = randf(0.f, (float)(2.0 * M_PI));
        float z     = randg(0.f, cfg_.disk_thickness);  // thin disk scatter

        h_pos_[i] = make_float4(r * cosf(theta), r * sinf(theta), z, star_mass);

        // Circular speed at r (Keplerian + enclosed disk contribution).
        float vc   = v_circ(r, mass_enclosed(r, M_bh, M_disk, R));
        float disp = vc * 0.04f;   // 4 % random velocity dispersion

        // Tangential velocity for counter-clockwise (prograde) orbit.
        h_vel_[i] = make_float4(
            -vc * sinf(theta) + randg(0.f, disp),    // vx
             vc * cosf(theta) + randg(0.f, disp),    // vy
             randg(0.f, disp * 0.2f),                // vz (small vertical kick)
             0.f);
    }
}

// Bulge stars: placed in a 3-D sphere with isotropic random velocities
// scaled to the local virial speed.
void GalaxySimulation::setup_bulge() {
    int   n_stars  = cfg_.n_bodies - 1;
    int   n_bulge  = (int)(cfg_.n_bodies * cfg_.bulge_frac);
    int   n_disk   = cfg_.n_bodies - 1 - n_bulge;   // bulge: [n_disk+1 .. n-1]

    float M_bh     = cfg_.bh_mass_frac;
    float M_disk   = 1.0f - cfg_.bh_mass_frac;
    float star_mass = M_disk / (float)n_stars;
    float R         = cfg_.disk_radius;
    float R_bulge   = R * 0.12f;    // bulge scale radius ≈ 12 % of disk radius

    for (int i = n_disk + 1; i < cfg_.n_bodies; i++) {
        // Uniform sphere: r = R_bulge * cbrt(U),  uniform on sphere via (θ, φ).
        float r   = R_bulge * cbrtf(randf(1e-3f, 1.f));
        float th  = randf(0.f, (float)(2.0 * M_PI));
        float phi = acosf(randf(-1.f, 1.f));   // uniform on sphere

        h_pos_[i] = make_float4(
            r * sinf(phi) * cosf(th),
            r * sinf(phi) * sinf(th),
            r * cosf(phi),
            star_mass);

        // Isotropic random velocity: |v| drawn from Gaussian, direction uniform.
        float v_vir = v_circ(r, mass_enclosed(r, M_bh, M_disk, R));
        float vmag  = fabsf(randg(0.f, v_vir * 0.5f));
        float vth   = randf(0.f, (float)(2.0 * M_PI));
        float vphi  = acosf(randf(-1.f, 1.f));

        h_vel_[i] = make_float4(
            vmag * sinf(vphi) * cosf(vth),
            vmag * sinf(vphi) * sinf(vth),
            vmag * cosf(vphi),
            0.f);
    }
}

// ─── init(): build ICs, print summary, upload to GPU ─────────────────────────
void GalaxySimulation::init() {
    int   n_bulge = (int)(cfg_.n_bodies * cfg_.bulge_frac);
    int   n_disk  = cfg_.n_bodies - 1 - n_bulge;
    float M_bh    = cfg_.bh_mass_frac;
    int   n_snap  = cfg_.n_steps / cfg_.out_interval;

    printf("━━━  Galaxy N-Body Simulation (CUDA)  ━━━\n");
    printf("  Bodies  : %d  (1 BH + %d disk + %d bulge)\n",
           cfg_.n_bodies, n_disk, n_bulge);
    printf("  BH mass : %.4f  (%.0f%% of M_total=1)\n",
           M_bh, M_bh * 100.f);
    printf("  dt      : %g  |  steps: %d  |  snapshots: %d\n",
           cfg_.dt, cfg_.n_steps, n_snap);
    printf("  Total time: %.1f  (outer orbit ≈ 2π ≈ 6.3 time units)\n\n",
           cfg_.dt * cfg_.n_steps);

    setup_black_hole();
    setup_disk();
    setup_bulge();

    gpu_.allocate(cfg_.n_bodies);
    gpu_.upload(h_pos_.data(), h_vel_.data());
}

// ─── write_snapshot(): append one frame of positions to a binary file ─────────
// Format per body: float x, float y, float z  (mass is constant, not saved).
void GalaxySimulation::write_snapshot(std::ofstream& file) const {
    for (int i = 0; i < cfg_.n_bodies; i++) {
        float xyz[3] = { h_pos_[i].x, h_pos_[i].y, h_pos_[i].z };
        file.write(reinterpret_cast<char*>(xyz), 3 * sizeof(float));
    }
}

// ─── run(): time-integration loop ─────────────────────────────────────────────
void GalaxySimulation::run() {
    // ── Open output file and write header ─────────────────────────────────
    std::ofstream out(cfg_.out_file, std::ios::binary);
    if (!out) {
        fprintf(stderr, "ERROR: cannot open '%s' for writing\n",
                cfg_.out_file.c_str());
        return;
    }
    int n_snap = cfg_.n_steps / cfg_.out_interval;
    out.write(reinterpret_cast<char*>(&cfg_.n_bodies), sizeof(int));
    out.write(reinterpret_cast<char*>(&n_snap),        sizeof(int));

    printf("Running…\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < cfg_.n_steps; step++) {
        // ── One leapfrog step (asynchronous GPU kernel) ───────────────────
        gpu_.step(cfg_.dt);

        // ── Save snapshot every out_interval steps ────────────────────────
        // download() implicitly synchronises the GPU (cudaMemcpy is blocking).
        if (step % cfg_.out_interval == 0) {
            gpu_.download(h_pos_.data(), h_vel_.data());
            write_snapshot(out);

            int pct = (step * 100) / cfg_.n_steps;
            printf("\r  [%3d%%] step %5d / %d", pct, step, cfg_.n_steps);
            fflush(stdout);
        }
    }

    auto t1  = std::chrono::high_resolution_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();

    printf("\r  [100%%] done in %.2f s\n", s);

    double mb = (8.0 + (double)cfg_.n_bodies * n_snap * 3 * 4) / 1e6;
    printf("  Output : %s  (%.1f MB)\n", cfg_.out_file.c_str(), mb);
    printf("\n  Visualize: python3 visualize.py %s\n", cfg_.out_file.c_str());
}
