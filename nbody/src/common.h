// common.h  —  Types and physics shared by the CPU (C++/OpenMP) and GPU (CUDA)
//              code paths.  This header must compile with both g++ and nvcc.
#pragma once
#include <cmath>

#if defined(__CUDACC__)
#  define NB_HD __host__ __device__ __forceinline__
#else
#  define NB_HD inline
#endif

// 16-byte aligned 4-vector.  Same memory layout as CUDA's float4, so device
// code can reinterpret a vec4* as a float4* and use 128-bit loads/stores.
//   positions : { x,  y,  z,  mass }
//   velocities: { vx, vy, vz, 0    }
//   accels    : { ax, ay, az, 0    }
struct alignas(16) vec4 {
    float x, y, z, w;
};

NB_HD vec4 make_vec4(float x, float y, float z, float w) {
    vec4 v; v.x = x; v.y = y; v.z = z; v.w = w; return v;
}

// Plummer softening length ε (simulation units, disk radius = 1).
constexpr float kSoftening = 0.02f;

// FLOP count per body–body interaction used for GFLOP/s figures.
// 20 FLOPs/interaction is the usual convention for direct-summation N-body
// codes (Nyland, Harris & Prins, "Fast N-Body Simulation with CUDA",
// GPU Gems 3, ch. 31).  It is a convention, not a hardware counter reading,
// so use it to compare implementations, not to compare against peak FLOP/s.
constexpr double kFlopsPerInteraction = 20.0;

// Time integrators: kick-drift-kick leapfrog, or semi-implicit Euler.
enum class Integrator { KDK, Euler };

// Total energy split into kinetic and potential parts (double precision).
struct Energy {
    double kinetic   = 0.0;
    double potential = 0.0;
    double total() const { return kinetic + potential; }
};
