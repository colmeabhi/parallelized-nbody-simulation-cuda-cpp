// main.cu  —  Galaxy simulation entry point.
//
// Usage: ./galaxy_nbody [n_bodies] [n_steps]
//                       [--integrator kdk|euler] [--kernel tiled|naive]
//                       [--block 256] [--dt 0.01] [--out simulation.bin]
//                       [--energy-log energy.csv]
//
// Defaults: 4096 bodies, 5000 KDK steps, 100 snapshots → simulation.bin,
// energy time series → energy.csv.
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "cli.h"
#include "cpu_nbody.h"
#include "galaxy_simulation.cuh"

int main(int argc, char* argv[]) {
    Cli cli(argc, argv);
    SimConfig cfg;
    try {
        const auto& pos = cli.positional();
        if (pos.size() > 0) cfg.galaxy.n_bodies = std::atoi(pos[0].c_str());
        if (pos.size() > 1) cfg.n_steps         = std::atoi(pos[1].c_str());
        cfg.galaxy.n_bodies = (int)cli.num("n", cfg.galaxy.n_bodies);
        cfg.n_steps         = (int)cli.num("steps", cfg.n_steps);
        cfg.dt              = (float)cli.real("dt", cfg.dt);
        cfg.integrator      = parse_integrator(cli.str("integrator", "kdk"));
        cfg.kernel          = parse_gpu_kernel(cli.str("kernel", "tiled"));
        cfg.block_size      = (int)cli.num("block", cfg.block_size);
        cfg.out_file        = cli.str("out", cfg.out_file);
        cfg.energy_file     = cli.str("energy-log", cfg.energy_file);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    GalaxySimulation sim(cfg);
    sim.init();
    sim.run();
    return 0;
}
