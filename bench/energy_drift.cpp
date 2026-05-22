// energy_drift.cpp - measures how well each integrator conserves energy.
//
// Usage:  ./build/energy_drift <kdk|euler> <dt>
// Runs a 1024-body galaxy for 20 time units on the CPU and prints CSV lines:
//     integrator,dt,time,relative_energy_error
// A perfect simulation would print 0 in the last column the whole time.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cpu_nbody.h"
#include "galaxy.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage: %s <kdk|euler> <dt>\n", argv[0]);
        return 1;
    }
    bool  use_euler = strcmp(argv[1], "euler") == 0;
    float dt        = atof(argv[2]);

    const int   n            = 1024;
    const float end_time     = 20.0f;   // about 3 orbits of the outer disk
    const int   steps        = (int)std::round(end_time / dt);
    const int   print_every  = steps / 100;

    std::vector<Vec4> pos(n), vel(n), acc(n);
    make_galaxy(n, pos.data(), vel.data());
    compute_accel_cpu(pos.data(), acc.data(), n, true);

    double start_energy = total_energy(pos.data(), vel.data(), n);

    for (int s = 1; s <= steps; s++) {
        if (use_euler) step_euler_cpu(pos.data(), vel.data(), acc.data(), n, dt, true);
        else           step_kdk_cpu  (pos.data(), vel.data(), acc.data(), n, dt, true);

        if (s % print_every == 0) {
            double energy = total_energy(pos.data(), vel.data(), n);
            double error  = (energy - start_energy) / std::fabs(start_energy);
            printf("%s,%g,%g,%.6e\n", argv[1], dt, s * dt, error);
        }
    }
    return 0;
}
