// cpu_nbody.cpp  —  CPU reference solvers (see cpu_nbody.h).
#include "cpu_nbody.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// ─── Parsing helpers ─────────────────────────────────────────────────────────
CpuMethod parse_cpu_method(const std::string& s) {
    if (s == "serial") return CpuMethod::Serial;
    if (s == "omp" || s == "openmp") return CpuMethod::OpenMP;
    if (s == "bh" || s == "barnes-hut") return CpuMethod::BarnesHut;
    throw std::invalid_argument("unknown CPU method '" + s + "' (serial|omp|bh)");
}
Integrator parse_integrator(const std::string& s) {
    if (s == "kdk" || s == "leapfrog") return Integrator::KDK;
    if (s == "euler") return Integrator::Euler;
    throw std::invalid_argument("unknown integrator '" + s + "' (kdk|euler)");
}
const char* to_string(CpuMethod m) {
    switch (m) {
        case CpuMethod::Serial:    return "cpu_serial";
        case CpuMethod::OpenMP:    return "cpu_openmp";
        case CpuMethod::BarnesHut: return "cpu_barnes_hut";
    }
    return "?";
}
const char* to_string(Integrator m) {
    return m == Integrator::KDK ? "kdk" : "euler";
}

// ─── Direct summation ────────────────────────────────────────────────────────
// Positions are converted to structure-of-arrays once per call (O(N)) so the
// O(N) inner loop is a clean unit-stride loop the compiler vectorises.
namespace {

struct SoA {
    std::vector<float> x, y, z, m;
    void load(const vec4* p, int n) {
        x.resize(n); y.resize(n); z.resize(n); m.resize(n);
        for (int i = 0; i < n; i++) {
            x[i] = p[i].x; y[i] = p[i].y; z[i] = p[i].z; m[i] = p[i].w;
        }
    }
};

inline vec4 accel_on(float xi, float yi, float zi, const SoA& s, int n,
                     float soft2)
{
    const float* __restrict__ X = s.x.data();
    const float* __restrict__ Y = s.y.data();
    const float* __restrict__ Z = s.z.data();
    const float* __restrict__ M = s.m.data();
    float ax = 0.f, ay = 0.f, az = 0.f;
    #pragma omp simd reduction(+:ax, ay, az)
    for (int j = 0; j < n; j++) {
        float dx = X[j] - xi;
        float dy = Y[j] - yi;
        float dz = Z[j] - zi;
        float r2    = dx * dx + dy * dy + dz * dz + soft2;
        float inv_r = 1.0f / std::sqrt(r2);
        float sc    = M[j] * inv_r * inv_r * inv_r;
        ax += dx * sc;  ay += dy * sc;  az += dz * sc;
    }
    return make_vec4(ax, ay, az, 0.f);
}

} // namespace

void accel_direct_serial(const vec4* pos, vec4* acc, int n, float soft2) {
    static thread_local SoA s;
    s.load(pos, n);
    for (int i = 0; i < n; i++)
        acc[i] = accel_on(pos[i].x, pos[i].y, pos[i].z, s, n, soft2);
}

void accel_direct_omp(const vec4* pos, vec4* acc, int n, float soft2) {
    static SoA s;   // shared, read-only inside the parallel region
    s.load(pos, n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        acc[i] = accel_on(pos[i].x, pos[i].y, pos[i].z, s, n, soft2);
}

// ─── Barnes–Hut ──────────────────────────────────────────────────────────────
namespace {

constexpr int kLeafCap  = 8;    // max bodies in a leaf
constexpr int kMaxDepth = 40;   // guards against coincident bodies

struct BHNode {
    float cx, cy, cz, half;      // cube centre and half-width
    float mx, my, mz, m;         // centre of mass and total mass
    int   child[8];              // -1 if absent
    int   first, count;          // body range in perm[] (leaves only)
    bool  leaf;
};

class Octree {
public:
    void build(const vec4* pos, int n) {
        pos_ = pos;
        nodes_.clear();
        nodes_.reserve(2 * (size_t)n / kLeafCap + 64);
        perm_.resize(n);
        tmp_.resize(n);
        for (int i = 0; i < n; i++) perm_[i] = i;

        float lo[3] = { pos[0].x, pos[0].y, pos[0].z };
        float hi[3] = { lo[0], lo[1], lo[2] };
        for (int i = 1; i < n; i++) {
            lo[0] = std::min(lo[0], pos[i].x); hi[0] = std::max(hi[0], pos[i].x);
            lo[1] = std::min(lo[1], pos[i].y); hi[1] = std::max(hi[1], pos[i].y);
            lo[2] = std::min(lo[2], pos[i].z); hi[2] = std::max(hi[2], pos[i].z);
        }
        float half = 0.5f * std::max({ hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2] });
        half = half * 1.0001f + 1e-6f;
        build_node(0, n, 0.5f * (lo[0] + hi[0]), 0.5f * (lo[1] + hi[1]),
                   0.5f * (lo[2] + hi[2]), half, 0);
    }

    // Acceleration on a point (px,py,pz) with opening angle theta.
    vec4 accel(float px, float py, float pz, float theta2, float soft2) const {
        float ax = 0.f, ay = 0.f, az = 0.f;
        int stack[8 * kMaxDepth + 8];
        int sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            const BHNode& nd = nodes_[stack[--sp]];
            if (nd.leaf) {
                for (int k = nd.first; k < nd.first + nd.count; k++) {
                    const vec4& b = pos_[perm_[k]];
                    float dx = b.x - px, dy = b.y - py, dz = b.z - pz;
                    float r2 = dx * dx + dy * dy + dz * dz + soft2;
                    float inv_r = 1.0f / std::sqrt(r2);
                    float sc = b.w * inv_r * inv_r * inv_r;
                    ax += dx * sc; ay += dy * sc; az += dz * sc;
                }
                continue;
            }
            float dx = nd.mx - px, dy = nd.my - py, dz = nd.mz - pz;
            float d2 = dx * dx + dy * dy + dz * dz;
            float s  = 2.f * nd.half;
            if (s * s < theta2 * d2) {          // far enough: use monopole
                float r2 = d2 + soft2;
                float inv_r = 1.0f / std::sqrt(r2);
                float sc = nd.m * inv_r * inv_r * inv_r;
                ax += dx * sc; ay += dy * sc; az += dz * sc;
            } else {                             // open the node
                for (int c = 0; c < 8; c++)
                    if (nd.child[c] >= 0) stack[sp++] = nd.child[c];
            }
        }
        return make_vec4(ax, ay, az, 0.f);
    }

    size_t n_nodes() const { return nodes_.size(); }

private:
    int build_node(int begin, int end, float cx, float cy, float cz,
                   float half, int depth)
    {
        int idx = (int)nodes_.size();
        nodes_.push_back(BHNode{});
        BHNode nd{};
        nd.cx = cx; nd.cy = cy; nd.cz = cz; nd.half = half;
        for (int c = 0; c < 8; c++) nd.child[c] = -1;
        nd.first = begin; nd.count = end - begin;

        if (end - begin <= kLeafCap || depth >= kMaxDepth) {
            nd.leaf = true;
            double m = 0, mx = 0, my = 0, mz = 0;
            for (int k = begin; k < end; k++) {
                const vec4& b = pos_[perm_[k]];
                m += b.w; mx += b.w * b.x; my += b.w * b.y; mz += b.w * b.z;
            }
            set_com(nd, m, mx, my, mz);
            nodes_[idx] = nd;
            return idx;
        }

        // Counting sort of perm_[begin,end) into the 8 octants.
        int counts[8] = { 0 };
        auto octant = [&](int b) {
            const vec4& p = pos_[b];
            return (p.x >= cx ? 1 : 0) | (p.y >= cy ? 2 : 0) | (p.z >= cz ? 4 : 0);
        };
        for (int k = begin; k < end; k++) counts[octant(perm_[k])]++;
        int offs[9]; offs[0] = begin;
        for (int c = 0; c < 8; c++) offs[c + 1] = offs[c] + counts[c];
        int fill[8];
        for (int c = 0; c < 8; c++) fill[c] = offs[c];
        for (int k = begin; k < end; k++) {
            int b = perm_[k];
            tmp_[fill[octant(b)]++] = b;
        }
        std::copy(tmp_.begin() + begin, tmp_.begin() + end, perm_.begin() + begin);

        nd.leaf = false;
        double m = 0, mx = 0, my = 0, mz = 0;
        float h = 0.5f * half;
        for (int c = 0; c < 8; c++) {
            if (counts[c] == 0) continue;
            int ch = build_node(offs[c], offs[c + 1],
                                cx + ((c & 1) ? h : -h),
                                cy + ((c & 2) ? h : -h),
                                cz + ((c & 4) ? h : -h), h, depth + 1);
            nd.child[c] = ch;
            const BHNode& cn = nodes_[ch];
            m  += cn.m;
            mx += (double)cn.m * cn.mx;
            my += (double)cn.m * cn.my;
            mz += (double)cn.m * cn.mz;
        }
        set_com(nd, m, mx, my, mz);
        nodes_[idx] = nd;
        return idx;
    }

    void set_com(BHNode& nd, double m, double mx, double my, double mz) {
        nd.m = (float)m;
        if (m > 0) { nd.mx = (float)(mx / m); nd.my = (float)(my / m); nd.mz = (float)(mz / m); }
        else       { nd.mx = nd.cx; nd.my = nd.cy; nd.mz = nd.cz; }
    }

    const vec4*          pos_ = nullptr;
    std::vector<BHNode>  nodes_;
    std::vector<int>     perm_, tmp_;
};

} // namespace

void accel_barnes_hut(const vec4* pos, vec4* acc, int n, float soft2,
                      float theta)
{
    static Octree tree;
    tree.build(pos, n);
    const float theta2 = theta * theta;
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; i++)
        acc[i] = tree.accel(pos[i].x, pos[i].y, pos[i].z, theta2, soft2);
}

// ─── Energy ──────────────────────────────────────────────────────────────────
Energy total_energy_cpu(const vec4* pos, const vec4* vel, int n, float soft2) {
    double ke = 0.0, pe = 0.0;
    #pragma omp parallel for reduction(+:ke) schedule(static)
    for (int i = 0; i < n; i++) {
        double v2 = (double)vel[i].x * vel[i].x + (double)vel[i].y * vel[i].y
                  + (double)vel[i].z * vel[i].z;
        ke += 0.5 * pos[i].w * v2;
    }
    #pragma omp parallel for reduction(+:pe) schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
        double xi = pos[i].x, yi = pos[i].y, zi = pos[i].z, phi = 0.0;
        for (int j = i + 1; j < n; j++) {
            double dx = pos[j].x - xi, dy = pos[j].y - yi, dz = pos[j].z - zi;
            phi += pos[j].w / std::sqrt(dx * dx + dy * dy + dz * dz + soft2);
        }
        pe -= pos[i].w * phi;
    }
    Energy e; e.kinetic = ke; e.potential = pe;
    return e;
}

// ─── CpuNBody ────────────────────────────────────────────────────────────────
CpuNBody::CpuNBody(std::vector<vec4> pos, std::vector<vec4> vel,
                   CpuMethod method, float soft2, float theta)
    : pos_(std::move(pos)), vel_(std::move(vel)), acc_(pos_.size()),
      method_(method), soft2_(soft2), theta_(theta) {}

void CpuNBody::compute_accel() {
    int n = (int)pos_.size();
    switch (method_) {
        case CpuMethod::Serial:    accel_direct_serial(pos_.data(), acc_.data(), n, soft2_); break;
        case CpuMethod::OpenMP:    accel_direct_omp   (pos_.data(), acc_.data(), n, soft2_); break;
        case CpuMethod::BarnesHut: accel_barnes_hut   (pos_.data(), acc_.data(), n, soft2_, theta_); break;
    }
    acc_valid_ = true;
}

void CpuNBody::step_kdk(float dt) {
    if (!acc_valid_) compute_accel();       // a(x_0), only on the first step
    const int n = (int)pos_.size();
    const float h = 0.5f * dt;
    for (int i = 0; i < n; i++) {           // kick ½ + drift
        vel_[i].x += acc_[i].x * h;  vel_[i].y += acc_[i].y * h;  vel_[i].z += acc_[i].z * h;
        pos_[i].x += vel_[i].x * dt; pos_[i].y += vel_[i].y * dt; pos_[i].z += vel_[i].z * dt;
    }
    compute_accel();                        // a(x_{n+1})
    for (int i = 0; i < n; i++) {           // kick ½
        vel_[i].x += acc_[i].x * h;  vel_[i].y += acc_[i].y * h;  vel_[i].z += acc_[i].z * h;
    }
}

void CpuNBody::step_euler(float dt) {
    compute_accel();
    const int n = (int)pos_.size();
    for (int i = 0; i < n; i++) {
        vel_[i].x += acc_[i].x * dt;  vel_[i].y += acc_[i].y * dt;  vel_[i].z += acc_[i].z * dt;
        pos_[i].x += vel_[i].x * dt;  pos_[i].y += vel_[i].y * dt;  pos_[i].z += vel_[i].z * dt;
    }
    acc_valid_ = false;  // acc_ now refers to the old positions
}

Energy CpuNBody::energy() const {
    return total_energy_cpu(pos_.data(), vel_.data(), (int)pos_.size(), soft2_);
}
