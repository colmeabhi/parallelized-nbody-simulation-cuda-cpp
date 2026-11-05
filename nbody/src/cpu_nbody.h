// cpu_nbody.h  —  CPU reference solvers: single-thread direct summation,
//                 OpenMP direct summation, and Barnes–Hut (OpenMP).
#pragma once
#include <string>
#include <vector>
#include "common.h"

enum class CpuMethod  { Serial, OpenMP, BarnesHut };
enum class Integrator { KDK, Euler };

CpuMethod   parse_cpu_method(const std::string& s);   // "serial" | "omp" | "bh"
Integrator  parse_integrator(const std::string& s);   // "kdk" | "euler"
const char* to_string(CpuMethod m);
const char* to_string(Integrator m);

// ── Accelerations ────────────────────────────────────────────────────────────
// a_i = Σ_j m_j (r_j − r_i) / (|r_j − r_i|² + ε²)^{3/2}       (G = 1)
// The self term is exactly zero (r_j − r_i = 0), so no j ≠ i branch is needed.

// One thread, SIMD-vectorised inner loop.  O(N²).
void accel_direct_serial(const vec4* pos, vec4* acc, int n, float soft2);

// OpenMP over i, same inner loop.  O(N²).
void accel_direct_omp(const vec4* pos, vec4* acc, int n, float soft2);

// Barnes–Hut octree with monopole approximation, opening angle theta.
// Tree build is serial; force traversal is OpenMP-parallel over bodies.
// O(N log N).
void accel_barnes_hut(const vec4* pos, vec4* acc, int n, float soft2,
                      float theta);

// ── Energy (double precision, OpenMP) ────────────────────────────────────────
// E = Σ ½ m_i v_i²  −  Σ_{i<j} m_i m_j / sqrt(r_ij² + ε²)
// The potential uses the same Plummer softening as the force, so E is the
// quantity a symplectic integrator of this softened system should conserve.
Energy total_energy_cpu(const vec4* pos, const vec4* vel, int n, float soft2);

// ── Simulation driver ────────────────────────────────────────────────────────
class CpuNBody {
public:
    CpuNBody(std::vector<vec4> pos, std::vector<vec4> vel, CpuMethod method,
             float soft2 = kSoftening * kSoftening, float theta = 0.5f);

    // Kick-drift-kick leapfrog (2nd order, symplectic, time-reversible):
    //   v_{n+½} = v_n + a(x_n)·dt/2
    //   x_{n+1} = x_n + v_{n+½}·dt
    //   v_{n+1} = v_{n+½} + a(x_{n+1})·dt/2
    // Uses one force evaluation per step (a(x_{n+1}) is reused next step).
    void step_kdk(float dt);

    // Semi-implicit (symplectic) Euler — what the original kernel did:
    //   v_{n+1} = v_n + a(x_n)·dt ;  x_{n+1} = x_n + v_{n+1}·dt
    // 1st order; kept only for the integrator comparison.
    void step_euler(float dt);

    void step(Integrator it, float dt) {
        if (it == Integrator::KDK) step_kdk(dt); else step_euler(dt);
    }

    Energy energy() const;
    const std::vector<vec4>& pos() const { return pos_; }
    const std::vector<vec4>& vel() const { return vel_; }

    void compute_accel();   // acc_ = a(pos_)

private:
    std::vector<vec4> pos_, vel_, acc_;
    CpuMethod method_;
    float     soft2_, theta_;
    bool      acc_valid_ = false;
};
