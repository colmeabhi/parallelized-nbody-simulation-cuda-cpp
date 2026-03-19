// cpu_nbody.h - the N-body simulation on the CPU.
// This is the reference version: simple loops, optionally run in parallel
// with OpenMP. The GPU code in gpu_nbody.cu does exactly the same maths.
#pragma once
#include "types.h"

// Computes the gravitational acceleration on every body:
//
//     acc[i] = sum over j of  m_j * (pos_j - pos_i) / (r^2 + softening^2)^(3/2)
//
// Every body looks at every other body, so this costs N * N interactions.
// use_openmp = false runs on one thread, true uses all the CPU cores.
void compute_accel_cpu(const Vec4* pos, Vec4* acc, int n, bool use_openmp);

// One time step with the Euler method:
//     acc = gravity(pos);  vel += acc * dt;  pos += vel * dt
void step_euler_cpu(Vec4* pos, Vec4* vel, Vec4* acc, int n, float dt, bool use_openmp);

// One time step with kick-drift-kick (KDK) leapfrog:
//     vel += acc * dt/2;  pos += vel * dt;  acc = gravity(pos);  vel += acc * dt/2
// IMPORTANT: acc must already hold gravity(pos) when this is called, so call
// compute_accel_cpu() once before the first step.
void step_kdk_cpu(Vec4* pos, Vec4* vel, Vec4* acc, int n, float dt, bool use_openmp);

// Total energy = kinetic + potential. With no outside forces this number
// should stay constant, so how much it changes tells us how accurate the
// simulation is.
double total_energy(const Vec4* pos, const Vec4* vel, int n);
