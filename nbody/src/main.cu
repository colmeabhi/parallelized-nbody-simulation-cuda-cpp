// main.cu  —  Entry point.
//
// Usage: ./galaxy_nbody [n_bodies] [n_steps]
//
// Defaults produce a 4096-body Milky-Way-like disk galaxy simulated for
// 5000 steps, saving 100 position snapshots to simulation.bin.

#include <cstdio>
#include <cstdlib>
#include "galaxy_simulation.cuh"

int main(int argc, char* argv[]) {
    GalaxyConfig cfg;

    // Allow quick tuning from the command line.
    if (argc > 1) cfg.n_bodies = atoi(argv[1]);
    if (argc > 2) cfg.n_steps  = atoi(argv[2]);

    GalaxySimulation sim(cfg);
    sim.init();
    sim.run();

    return 0;
}
