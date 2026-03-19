// cpu_nbody.cpp - the N-body simulation on the CPU (see cpu_nbody.h).
#include "cpu_nbody.h"
#include <cmath>

void compute_accel_cpu(const Vec4* pos, Vec4* acc, int n, bool use_openmp) {
    // Each body i is independent of the others, so the outer loop can be
    // split across CPU cores. "if (use_openmp)" turns that on or off.
    #pragma omp parallel for if (use_openmp)
    for (int i = 0; i < n; i++) {
        float ax = 0, ay = 0, az = 0;

        for (int j = 0; j < n; j++) {
            // Vector pointing from body i to body j.
            float dx = pos[j].x - pos[i].x;
            float dy = pos[j].y - pos[i].y;
            float dz = pos[j].z - pos[i].z;

            // 1 / distance^3, with softening so it never divides by zero.
            // When j == i the vector is (0,0,0), so that term adds nothing
            // and we do not need a special "skip myself" check.
            float dist_squared = dx * dx + dy * dy + dz * dz + SOFTENING_SQUARED;
            float inv_dist     = 1.0f / std::sqrt(dist_squared);
            float inv_dist3    = inv_dist * inv_dist * inv_dist;

            float strength = pos[j].w * inv_dist3;   // pos[j].w is the mass of j
            ax += dx * strength;
            ay += dy * strength;
            az += dz * strength;
        }

        acc[i] = { ax, ay, az, 0 };
    }
}

// "Kick": change the velocities using the accelerations.
static void kick(Vec4* vel, const Vec4* acc, int n, float dt) {
    for (int i = 0; i < n; i++) {
        vel[i].x += acc[i].x * dt;
        vel[i].y += acc[i].y * dt;
        vel[i].z += acc[i].z * dt;
    }
}

// "Drift": move the positions using the velocities.
static void drift(Vec4* pos, const Vec4* vel, int n, float dt) {
    for (int i = 0; i < n; i++) {
        pos[i].x += vel[i].x * dt;
        pos[i].y += vel[i].y * dt;
        pos[i].z += vel[i].z * dt;
    }
}

void step_euler_cpu(Vec4* pos, Vec4* vel, Vec4* acc, int n, float dt, bool use_openmp) {
    compute_accel_cpu(pos, acc, n, use_openmp);
    kick(vel, acc, n, dt);
    drift(pos, vel, n, dt);
}

void step_kdk_cpu(Vec4* pos, Vec4* vel, Vec4* acc, int n, float dt, bool use_openmp) {
    kick(vel, acc, n, dt / 2);                    // half kick with the old acceleration
    drift(pos, vel, n, dt);                       // full drift
    compute_accel_cpu(pos, acc, n, use_openmp);   // acceleration at the new positions
    kick(vel, acc, n, dt / 2);                    // half kick with the new acceleration
}

double total_energy(const Vec4* pos, const Vec4* vel, int n) {
    double kinetic = 0, potential = 0;

    // Doubles are used here on purpose: we are adding up millions of small
    // numbers and want the sum to be accurate.
    #pragma omp parallel for reduction(+ : kinetic, potential)
    for (int i = 0; i < n; i++) {
        double mass = pos[i].w;

        // Kinetic energy: 1/2 * m * v^2
        double v_squared = (double)vel[i].x * vel[i].x + (double)vel[i].y * vel[i].y
                         + (double)vel[i].z * vel[i].z;
        kinetic += 0.5 * mass * v_squared;

        // Potential energy: -m_i * m_j / distance for every pair, counted once
        // (j starts at i + 1). Uses the same softening as the force.
        for (int j = i + 1; j < n; j++) {
            double dx = (double)pos[j].x - pos[i].x;
            double dy = (double)pos[j].y - pos[i].y;
            double dz = (double)pos[j].z - pos[i].z;
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING_SQUARED);
            potential -= mass * pos[j].w / dist;
        }
    }
    return kinetic + potential;
}
