// nbody_kernel.cu  —  CUDA direct-summation N-body solver.
//
// Force:  a_i = Σ_j m_j (r_j − r_i) / (|r_j − r_i|² + ε²)^{3/2}     (G = 1)
//
// Two force kernels are provided so their performance can be compared:
//   accel_naive_kernel : each thread reads all N positions from global memory
//                        (relies on L1/L2 caches for reuse).
//   accel_tiled_kernel : each block stages blockDim.x positions at a time in
//                        shared memory; every global load is reused by all
//                        blockDim.x threads of the block.
//
// Integration: kick-drift-kick leapfrog, split across two kernels so that no
// kernel reads positions that the same kernel writes (see nbody_kernel.cuh).
//
//   step n → n+1:
//     kick_drift_kernel     v += a(x_n)·dt/2 ;  x += v·dt
//     accel_*_kernel<KICK>  a = a(x_{n+1})   ;  v += a·dt/2
#include "nbody_kernel.cuh"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// Capture `err` once so CUDA_CHECK(cudaGetLastError()) is not evaluated twice.
#define CUDA_CHECK(err)                                                    \
    do {                                                                   \
        cudaError_t _e = (err);                                            \
        if (_e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

GpuKernel parse_gpu_kernel(const std::string& s) {
    if (s == "naive") return GpuKernel::Naive;
    if (s == "tiled") return GpuKernel::Tiled;
    throw std::invalid_argument("unknown GPU kernel '" + s + "' (naive|tiled)");
}
const char* to_string(GpuKernel k) {
    return k == GpuKernel::Naive ? "gpu_naive" : "gpu_tiled";
}

// ─── Device helper: acceleration contribution of body j on body i ───────────
// Self-interaction: displacement is 0 → contribution is exactly 0.
// Padding bodies (mass 0) contribute 0.
__device__ __forceinline__ void accumulate(float3& acc, float4 pi, float4 pj,
                                           float soft2)
{
    float dx = pj.x - pi.x;
    float dy = pj.y - pi.y;
    float dz = pj.z - pi.z;
    float r2    = dx * dx + dy * dy + dz * dz + soft2;
    float inv_r = rsqrtf(r2);
    float s     = pj.w * inv_r * inv_r * inv_r;
    acc.x += dx * s;
    acc.y += dy * s;
    acc.z += dz * s;
}

// Store a_i, and optionally finish the leapfrog step with the second half-kick.
template <bool KICK>
__device__ __forceinline__ void store_result(int i, float3 a,
                                             float4* __restrict__ acc,
                                             float4* __restrict__ vel,
                                             float kick_dt)
{
    acc[i] = make_float4(a.x, a.y, a.z, 0.f);
    if (KICK) {
        float4 v = vel[i];
        v.x += a.x * kick_dt;
        v.y += a.y * kick_dt;
        v.z += a.z * kick_dt;
        vel[i] = v;
    }
}

// ─── Naive force kernel ─────────────────────────────────────────────────────
template <bool KICK>
__global__ void accel_naive_kernel(const float4* __restrict__ pos,
                                   float4* __restrict__ acc,
                                   float4* __restrict__ vel,
                                   int n, float soft2, float kick_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 pi = pos[i];
    float3 a  = make_float3(0.f, 0.f, 0.f);
    #pragma unroll 4
    for (int j = 0; j < n; j++)
        accumulate(a, pi, pos[j], soft2);
    store_result<KICK>(i, a, acc, vel, kick_dt);
}

// ─── Tiled (shared-memory) force kernel ─────────────────────────────────────
// Dynamic shared memory: blockDim.x float4 = one tile of positions.
template <bool KICK>
__global__ void accel_tiled_kernel(const float4* __restrict__ pos,
                                   float4* __restrict__ acc,
                                   float4* __restrict__ vel,
                                   int n, float soft2, float kick_dt)
{
    extern __shared__ float4 tile[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Threads with i ≥ n still help load tiles and hit every __syncthreads.
    float4 pi = (i < n) ? pos[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    float3 a  = make_float3(0.f, 0.f, 0.f);

    for (int base = 0; base < n; base += blockDim.x) {
        int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? pos[j] : make_float4(0.f, 0.f, 0.f, 0.f);
        __syncthreads();                       // tile fully loaded

        #pragma unroll 8
        for (int k = 0; k < blockDim.x; k++)
            accumulate(a, pi, tile[k], soft2);
        __syncthreads();                       // tile fully consumed
    }
    if (i < n) store_result<KICK>(i, a, acc, vel, kick_dt);
}

// ─── Integration kernel: v += a·kick_dt ; x += v·drift_dt ───────────────────
// Each thread reads and writes only body i → no cross-thread hazard.
__global__ void kick_drift_kernel(float4* __restrict__ pos,
                                  float4* __restrict__ vel,
                                  const float4* __restrict__ acc,
                                  int n, float kick_dt, float drift_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 a = acc[i], v = vel[i], p = pos[i];
    v.x += a.x * kick_dt;  v.y += a.y * kick_dt;  v.z += a.z * kick_dt;
    p.x += v.x * drift_dt; p.y += v.y * drift_dt; p.z += v.z * drift_dt;
    vel[i] = v;
    pos[i] = p;
}

// ─── Potential kernel (energy diagnostics, double precision) ────────────────
// phi_i = −Σ_{j≠i} m_j / sqrt(r_ij² + ε²).  Here j ≠ i *is* required: the
// softened self-term would be −m_i/ε, not 0.
__global__ void potential_kernel(const float4* __restrict__ pos,
                                 double* __restrict__ phi, int n, float soft2)
{
    extern __shared__ float4 tile[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 pi = (i < n) ? pos[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    double sum = 0.0;
    for (int base = 0; base < n; base += blockDim.x) {
        int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? pos[j] : make_float4(0.f, 0.f, 0.f, 0.f);
        __syncthreads();
        int kmax = min((int)blockDim.x, n - base);
        for (int k = 0; k < kmax; k++) {
            if (base + k == i) continue;
            double dx = (double)tile[k].x - pi.x;
            double dy = (double)tile[k].y - pi.y;
            double dz = (double)tile[k].z - pi.z;
            sum -= tile[k].w / sqrt(dx * dx + dy * dy + dz * dz + (double)soft2);
        }
        __syncthreads();
    }
    if (i < n) phi[i] = sum;
}

// ─── NBodyGPU host-side implementation ───────────────────────────────────────

void NBodyGPU::allocate(int n) {
    free();
    n_ = n;
    CUDA_CHECK(cudaMalloc(&d_pos, n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_vel, n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_acc, n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_phi, n * sizeof(double)));
    acc_valid_ = false;
}

void NBodyGPU::upload(const vec4* pos, const vec4* vel) {
    CUDA_CHECK(cudaMemcpy(d_pos, pos, n_ * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, vel, n_ * sizeof(float4), cudaMemcpyHostToDevice));
    acc_valid_ = false;
}

void NBodyGPU::download(vec4* pos, vec4* vel) const {
    CUDA_CHECK(cudaMemcpy(pos, d_pos, n_ * sizeof(float4), cudaMemcpyDeviceToHost));
    if (vel)
        CUDA_CHECK(cudaMemcpy(vel, d_vel, n_ * sizeof(float4), cudaMemcpyDeviceToHost));
}

void NBodyGPU::free() {
    if (d_pos) { cudaFree(d_pos); d_pos = nullptr; }
    if (d_vel) { cudaFree(d_vel); d_vel = nullptr; }
    if (d_acc) { cudaFree(d_acc); d_acc = nullptr; }
    if (d_phi) { cudaFree(d_phi); d_phi = nullptr; }
    n_ = 0;
    acc_valid_ = false;
}

void NBodyGPU::set_block_size(int b) {
    if (b < 32 || b > 1024 || b % 32 != 0)
        throw std::invalid_argument("block size must be a multiple of 32 in [32, 1024]");
    block_ = b;
}

void NBodyGPU::sync() const { CUDA_CHECK(cudaDeviceSynchronize()); }

void NBodyGPU::launch_accel(float kick_dt) {
    const int   blocks = (n_ + block_ - 1) / block_;
    const float soft2  = kSoftening * kSoftening;
    const bool  kick   = kick_dt != 0.f;

    if (kernel_ == GpuKernel::Tiled) {
        size_t smem = block_ * sizeof(float4);
        if (kick) accel_tiled_kernel<true ><<<blocks, block_, smem>>>(d_pos, d_acc, d_vel, n_, soft2, kick_dt);
        else      accel_tiled_kernel<false><<<blocks, block_, smem>>>(d_pos, d_acc, d_vel, n_, soft2, kick_dt);
    } else {
        if (kick) accel_naive_kernel<true ><<<blocks, block_>>>(d_pos, d_acc, d_vel, n_, soft2, kick_dt);
        else      accel_naive_kernel<false><<<blocks, block_>>>(d_pos, d_acc, d_vel, n_, soft2, kick_dt);
    }
    CUDA_CHECK(cudaGetLastError());
    acc_valid_ = true;
}

void NBodyGPU::compute_accel() { launch_accel(0.f); }

void NBodyGPU::step_kdk(float dt) {
    if (!acc_valid_) launch_accel(0.f);              // a(x_0), first step only
    const int blocks = (n_ + block_ - 1) / block_;
    kick_drift_kernel<<<blocks, block_>>>(d_pos, d_vel, d_acc, n_, 0.5f * dt, dt);
    CUDA_CHECK(cudaGetLastError());
    launch_accel(0.5f * dt);                         // a(x_{n+1}) + half-kick
}

void NBodyGPU::step_euler(float dt) {
    launch_accel(0.f);                               // a(x_n)
    const int blocks = (n_ + block_ - 1) / block_;
    kick_drift_kernel<<<blocks, block_>>>(d_pos, d_vel, d_acc, n_, dt, dt);
    CUDA_CHECK(cudaGetLastError());
    acc_valid_ = false;                              // acc refers to old x
}

Energy NBodyGPU::energy() {
    const int blocks = (n_ + block_ - 1) / block_;
    potential_kernel<<<blocks, block_, block_ * sizeof(float4)>>>(
        d_pos, d_phi, n_, kSoftening * kSoftening);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> phi(n_);
    std::vector<vec4>   pos(n_), vel(n_);
    CUDA_CHECK(cudaMemcpy(phi.data(), d_phi, n_ * sizeof(double), cudaMemcpyDeviceToHost));
    download(pos.data(), vel.data());

    Energy e;
    for (int i = 0; i < n_; i++) {
        double v2 = (double)vel[i].x * vel[i].x + (double)vel[i].y * vel[i].y
                  + (double)vel[i].z * vel[i].z;
        e.kinetic   += 0.5 * pos[i].w * v2;
        e.potential += 0.5 * pos[i].w * phi[i];       // ½: each pair counted twice
    }
    return e;
}
