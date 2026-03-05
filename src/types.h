// types.h - the few types and constants shared by the CPU and GPU code.
#pragma once

// Four floats. We use it for three things:
//   position     : x, y, z and w = mass
//   velocity     : x, y, z (w unused)
//   acceleration : x, y, z (w unused)
// Keeping the mass next to the position means the force loop only has to
// read one array.
struct Vec4 {
    float x, y, z, w;
};

// Softening length. Gravity goes like 1/r^2, which blows up when two bodies
// get very close. Adding SOFTENING^2 to r^2 keeps the force finite.
const float SOFTENING = 0.02f;
const float SOFTENING_SQUARED = SOFTENING * SOFTENING;

// Used only to report GFLOP/s. People who benchmark N-body codes usually
// count one body-body interaction as 20 floating point operations.
const double FLOPS_PER_INTERACTION = 20.0;
