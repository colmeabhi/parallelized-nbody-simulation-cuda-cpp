// galaxy.h - builds the starting positions and velocities of the galaxy.
#pragma once
#include "types.h"

// Fills pos[0..n-1] and vel[0..n-1].
//   body 0        : a heavy black hole sitting still at the centre
//   about 90%     : disk stars on near-circular orbits
//   about 10%     : bulge stars in a small ball around the centre
//
// Units: G = 1, total mass = 1, disk radius = 1. In these units a star at the
// edge of the disk moves at speed ~1 and one orbit takes about 2*pi = 6.3.
void make_galaxy(int n, Vec4* pos, Vec4* vel);
