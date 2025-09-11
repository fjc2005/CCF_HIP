#include "main.h"
#include <cstdio>
#include <cstdlib>

#ifndef HIP_CHECK
#define HIP_CHECK(call) do { \
    hipError_t _hip_status = (call); \
    if (_hip_status != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s:%d: %s (code %d)\n", __FILE__, __LINE__, hipGetErrorString(_hip_status), static_cast<int>(_hip_status)); \
        std::abort(); \
    } \
} while (0)
#endif

// 使用数值稳定的 softmax 实现：
// m = max(x), t_i = exp(x_i - m), S = sum(t_i), y_i = t_i / S

__global__ void kernel_block_max(const float* __restrict__ x, int n, float* __restrict__ block_max) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const int bid = blockIdx.x;
    const int gid0 = bid * blockDim.x + tid;
    float local_max = -FLT_MAX;

    // 网格步长遍历，保证覆盖全部 n
    for (int i = gid0; i < n; i += gridDim.x * threads) {
        float v = x[i];
        local_max = fmaxf(local_max, v);
    }

    sdata[tid] = local_max;
    __syncthreads();

    // 归约（块内最大值）
    for (int stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[bid] = sdata[0];
    }
}

__global__ void kernel_reduce_max_final(const float* __restrict__ partial, int n, float* __restrict__ result) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    float local_max = -FLT_MAX;

    // 单块归约所有 partial
    for (int i = tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, partial[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        result[0] = sdata[0];
    }
}

__global__ void kernel_compute_exp_and_block_sum(const float* __restrict__ x,
                                                 int n,
                                                 const float m,
                                                 float* __restrict__ t,
                                                 float* __restrict__ block_sum) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    const int bid = blockIdx.x;
    const int gid0 = bid * blockDim.x + tid;

    float local_sum = 0.0f;
    for (int i = gid0; i < n; i += gridDim.x * threads) {
        float v = x[i] - m;
        float tv = expf(v);
        t[i] = tv;
        local_sum += tv;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sum[bid] = sdata[0];
    }
}

__global__ void kernel_reduce_sum_final(const float* __restrict__ partial, int n, float* __restrict__ result) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += partial[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        result[0] = sdata[0];
    }
}

__global__ void kernel_normalize(float* __restrict__ t, int n, const float denom, float* __restrict__ y) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        y[i] = t[i] / denom;
    }
}

static inline int compute_num_blocks(int n, int threads_per_block) {
    // 采用网格步长循环，限制网格规模，避免过多 blocks
    int max_blocks = 1024; // 保守上限
    int est_blocks = (n + threads_per_block - 1) / threads_per_block;
    if (est_blocks < 1) est_blocks = 1;
    return est_blocks > max_blocks ? max_blocks : est_blocks;
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    const int threads = 256;
    const int blocks_max = compute_num_blocks(N, threads);

    float *d_x = nullptr, *d_t = nullptr, *d_y = nullptr;
    float *d_block_max = nullptr, *d_block_sum = nullptr;
    float *d_max = nullptr, *d_sum = nullptr;

    auto start_alloc = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipMalloc(&d_x, sizeof(float) * N));
    HIP_CHECK(hipMalloc(&d_t, sizeof(float) * N));
    HIP_CHECK(hipMalloc(&d_y, sizeof(float) * N));
    HIP_CHECK(hipMalloc(&d_block_max, sizeof(float) * blocks_max));
    HIP_CHECK(hipMalloc(&d_block_sum, sizeof(float) * blocks_max));
    HIP_CHECK(hipMalloc(&d_max, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sum, sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());
    if (g_enable_timing) {
        auto end_alloc = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end_alloc - start_alloc);
        std::cerr << "[TIMER] GPU memory allocation: " << dur.count() << " us" << std::endl;
    }

    auto start_h2d = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipMemcpy(d_x, input, sizeof(float) * N, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    if (g_enable_timing) {
        auto end_h2d = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end_h2d - start_h2d);
        std::cerr << "[TIMER] Data transfer to device: " << dur.count() << " us" << std::endl;
    }

    // 1) 计算最大值 m
    auto start_gpu = std::chrono::high_resolution_clock::now();
    size_t shmem = threads * sizeof(float);
    hipLaunchKernelGGL(kernel_block_max, dim3(blocks_max), dim3(threads), shmem, 0, d_x, N, d_block_max);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kernel_reduce_max_final, dim3(1), dim3(threads), shmem, 0, d_block_max, blocks_max, d_max);
    HIP_CHECK(hipGetLastError());

    // 2) 计算 t_i 和分块和，进而得到总和 S
    // 读取 m 到主机
    float h_m = 0.0f;
    HIP_CHECK(hipMemcpy(&h_m, d_max, sizeof(float), hipMemcpyDeviceToHost));
    // 计算 t 与分块和（正确的传参）
    hipLaunchKernelGGL(kernel_compute_exp_and_block_sum, dim3(blocks_max), dim3(threads), shmem, 0, d_x, N, h_m, d_t, d_block_sum);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(kernel_reduce_sum_final, dim3(1), dim3(threads), shmem, 0, d_block_sum, blocks_max, d_sum);
    HIP_CHECK(hipGetLastError());

    // 读取 S 到主机
    float h_S = 0.0f;
    HIP_CHECK(hipMemcpy(&h_S, d_sum, sizeof(float), hipMemcpyDeviceToHost));
    if (h_S < 1e-12f) h_S = 1e-12f;

    // 3) 归一化得到 y
    hipLaunchKernelGGL(kernel_normalize, dim3(blocks_max), dim3(threads), 0, 0, d_t, N, h_S, d_y);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    if (g_enable_timing) {
        auto end_gpu = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
        std::cerr << "[TIMER] GPU computation(Solve): " << dur.count() << " us" << std::endl;
    }

    // 拷回结果
    auto start_d2h = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipMemcpy(output, d_y, sizeof(float) * N, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    if (g_enable_timing) {
        auto end_d2h = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end_d2h - start_d2h);
        std::cerr << "[TIMER] Data transfer from device: " << dur.count() << " us" << std::endl;
    }

    HIP_CHECK(hipDeviceSynchronize());

    auto start_cleanup = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_t));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_block_max));
    HIP_CHECK(hipFree(d_block_sum));
    HIP_CHECK(hipFree(d_max));
    HIP_CHECK(hipFree(d_sum));
    HIP_CHECK(hipDeviceSynchronize());
    if (g_enable_timing) {
        auto end_cleanup = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end_cleanup - start_cleanup);
        std::cerr << "[TIMER] GPU memory cleanup: " << dur.count() << " us" << std::endl;
    }
}