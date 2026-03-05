// galaxy.cpp - starting conditions for a simple Milky-Way-like galaxy.
#include "galaxy.h"
#include <cmath>
#include <random>

const float PI = 3.14159265f;

const float BLACK_HOLE_MASS = 0.05f;   // 5% of the total mass
const float STARS_MASS      = 0.95f;   // the rest is shared by all the stars
const float BULGE_FRACTION  = 0.10f;   // 10% of the bodies go in the bulge
const float BULGE_RADIUS    = 0.12f;
const float DISK_THICKNESS  = 0.02f;

// Mass inside radius r: the black hole plus the part of the disk inside r.
// (The disk has uniform surface density, so its mass grows like r^2.)
static float mass_inside(float r) {
    float disk_part = (r < 1.0f) ? r * r : 1.0f;
    return BLACK_HOLE_MASS + STARS_MASS * disk_part;
}

// Speed needed for a circular orbit at radius r:  v = sqrt(G * M / r), G = 1.
static float circular_speed(float r) {
    return std::sqrt(mass_inside(r) / r);
}

void make_galaxy(int n, Vec4* pos, Vec4* vel) {
    std::mt19937 rng(42);   // fixed seed: every run starts from the same galaxy
    auto uniform = [&](float lo, float hi) {
        return std::uniform_real_distribution<float>(lo, hi)(rng);
    };
    auto gaussian = [&](float sigma) {
        return std::normal_distribution<float>(0.0f, sigma)(rng);
    };

    int   n_bulge   = (int)(n * BULGE_FRACTION);
    int   n_disk    = n - 1 - n_bulge;
    float star_mass = STARS_MASS / (n - 1);

    // Body 0: the black hole, at rest in the centre.
    pos[0] = { 0, 0, 0, BLACK_HOLE_MASS };
    vel[0] = { 0, 0, 0, 0 };

    // Disk stars: random spot in a thin disk, moving on a circle around the
    // centre with a little random motion on top.
    for (int i = 1; i <= n_disk; i++) {
        float r     = std::sqrt(uniform(0.0025f, 1.0f));   // 0.05 <= r <= 1
        float angle = uniform(0, 2 * PI);
        float z     = gaussian(DISK_THICKNESS);
        pos[i] = { r * std::cos(angle), r * std::sin(angle), z, star_mass };

        float v      = circular_speed(r);
        float jitter = 0.04f * v;
        vel[i] = { -v * std::sin(angle) + gaussian(jitter),
                    v * std::cos(angle) + gaussian(jitter),
                    gaussian(0.2f * jitter),
                    0 };
    }

    // Bulge stars: random spot inside a small ball, random direction of motion.
    for (int i = n_disk + 1; i < n; i++) {
        float r     = BULGE_RADIUS * std::cbrt(uniform(0.001f, 1.0f));
        float angle = uniform(0, 2 * PI);
        float tilt  = std::acos(uniform(-1, 1));
        pos[i] = { r * std::sin(tilt) * std::cos(angle),
                   r * std::sin(tilt) * std::sin(angle),
                   r * std::cos(tilt),
                   star_mass };

        float speed   = std::fabs(gaussian(0.5f * circular_speed(r)));
        float v_angle = uniform(0, 2 * PI);
        float v_tilt  = std::acos(uniform(-1, 1));
        vel[i] = { speed * std::sin(v_tilt) * std::cos(v_angle),
                   speed * std::sin(v_tilt) * std::sin(v_angle),
                   speed * std::cos(v_tilt),
                   0 };
    }
}
