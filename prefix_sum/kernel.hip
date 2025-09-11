#include "main.h"
#include <cstdlib>

// 简单多层并行扫描：
// 1) 块内 inclusive 扫描并输出，每块记录块和
// 2) 递归地对块和数组做扫描
// 3) 将上一步得到的偏移（块和扫描的前一项）加回到对应块的元素上

#define BLOCK_SIZE 256

static inline void HIP_CHECK(hipError_t status, const char* context) {
    if (status != hipSuccess) {
        std::cerr << "HIP error in " << context << ": " << hipGetErrorString(status) << std::endl;
        std::exit(1);
    }
}

// 运行时实现选择：0 = Hillis-Steele, 1 = Blelloch
static int g_scan_impl = 1;

extern "C" void set_scan_impl(int impl) {
    g_scan_impl = (impl != 0) ? 1 : 0;
}

// Hillis-Steele 块内 inclusive 扫描
__global__ void block_scan_inclusive_kernel_hs(const int* __restrict__ in,
                                               int* __restrict__ out,
                                               int* __restrict__ block_sums,
                                               int length) {
    extern __shared__ int sdata[];

    const int tid = threadIdx.x;
    const int blockStart = blockIdx.x * blockDim.x;
    const int gid = blockStart + tid;

    int val = 0;
    if (gid < length) val = in[gid];
    sdata[tid] = val;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int add = 0;
        if (tid >= offset) add = sdata[tid - offset];
        __syncthreads();
        sdata[tid] += add;
        __syncthreads();
    }

    if (gid < length) out[gid] = sdata[tid];

    if (block_sums != nullptr) {
        int elementsInBlock = length - blockStart;
        if (elementsInBlock > blockDim.x) elementsInBlock = blockDim.x;
        if (elementsInBlock > 0) {
            int last = elementsInBlock - 1;
            if (tid == last) block_sums[blockIdx.x] = sdata[last];
        }
    }
}

// Blelloch 块内 work-efficient 扫描（输出 inclusive）
__global__ void block_scan_inclusive_kernel_blelloch(const int* __restrict__ in,
                                                     int* __restrict__ out,
                                                     int* __restrict__ block_sums,
                                                     int length) {
    extern __shared__ int sdata[];

    const int tid = threadIdx.x;
    const int blockStart = blockIdx.x * blockDim.x;
    const int gid = blockStart + tid;

    int val = 0;
    int local_n = blockDim.x;
    if (length - blockStart < local_n) local_n = length - blockStart;
    if (local_n < 0) local_n = 0;

    if (gid < length) val = in[gid];
    sdata[tid] = (tid < local_n) ? val : 0;
    __syncthreads();

    int nPow2 = 1;
    while (nPow2 < local_n) nPow2 <<= 1;
    if (nPow2 < 1) nPow2 = 1;

    for (int offset = 1; offset < nPow2; offset <<= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < nPow2) {
            sdata[idx] += sdata[idx - offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // 将总和保留在 sdata[nPow2-1] 的同时，做 exclusive 化
        // downsweep 之前置 0 实现 exclusive
        sdata[nPow2 - 1] = 0;
    }
    __syncthreads();

    for (int offset = nPow2 >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < nPow2) {
            int t = sdata[idx - offset];
            sdata[idx - offset] = sdata[idx];
            sdata[idx] += t;
        }
        __syncthreads();
    }

    if (gid < length && tid < local_n) {
        int inclusive = sdata[tid] + val;
        out[gid] = inclusive;
    }

    if (block_sums != nullptr) {
        if (local_n > 0) {
            int last = local_n - 1;
            if (tid == last) {
                block_sums[blockIdx.x] = sdata[last] + val;
            }
        }
    }
}

__global__ void add_block_offsets_kernel(int* __restrict__ data,
                                         const int* __restrict__ scanned_block_sums,
                                         int length) {
    const int tid = threadIdx.x;
    const int blockStart = blockIdx.x * blockDim.x;
    const int gid = blockStart + tid;

    int offset = 0;
    if (blockIdx.x > 0) offset = scanned_block_sums[blockIdx.x - 1];

    if (gid < length) data[gid] += offset;
}

static void device_scan_inclusive(const int* d_in, int* d_out, int length) {
    if (length <= 0) return;

    int numBlocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int* d_block_sums = nullptr;
    if (numBlocks > 1) {
        HIP_CHECK(hipMalloc(&d_block_sums, sizeof(int) * numBlocks), "hipMalloc d_block_sums");
    }

    if (g_scan_impl == 1) {
        hipLaunchKernelGGL(block_scan_inclusive_kernel_blelloch,
                           dim3(numBlocks), dim3(BLOCK_SIZE), BLOCK_SIZE * sizeof(int), 0,
                           d_in, d_out, d_block_sums, length);
    } else {
        hipLaunchKernelGGL(block_scan_inclusive_kernel_hs,
                           dim3(numBlocks), dim3(BLOCK_SIZE), BLOCK_SIZE * sizeof(int), 0,
                           d_in, d_out, d_block_sums, length);
    }
    HIP_CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize after block_scan");

    if (numBlocks > 1) {
        int* d_scanned_block_sums = nullptr;
        HIP_CHECK(hipMalloc(&d_scanned_block_sums, sizeof(int) * numBlocks), "hipMalloc d_scanned_block_sums");
        // recursively scan block sums
        device_scan_inclusive(d_block_sums, d_scanned_block_sums, numBlocks);
        hipLaunchKernelGGL(add_block_offsets_kernel,
                           dim3(numBlocks), dim3(BLOCK_SIZE), 0, 0,
                           d_out, d_scanned_block_sums, length);
        HIP_CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize after add_block_offsets_kernel");

        HIP_CHECK(hipFree(d_scanned_block_sums), "hipFree d_scanned_block_sums");
        HIP_CHECK(hipFree(d_block_sums), "hipFree d_block_sums");
    }
}

extern "C" void solve(const int* input, int* output, int N) {
    if (N <= 0) return;

    int *d_in = nullptr, *d_out = nullptr;
    auto host_alloc_start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipMalloc(&d_in, sizeof(int) * (size_t)N), "hipMalloc d_in");
    HIP_CHECK(hipMalloc(&d_out, sizeof(int) * (size_t)N), "hipMalloc d_out");
    auto host_alloc_end = std::chrono::high_resolution_clock::now();
    if (g_enable_timing) {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(host_alloc_end - host_alloc_start).count();
        std::cerr << "[TIMER] GPU memory allocation: " << us << " us" << std::endl;
    }

    auto host_h2d_start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipMemcpy(d_in, input, sizeof(int) * (size_t)N, hipMemcpyHostToDevice), "hipMemcpy H2D");
    HIP_CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize after H2D");
    auto host_h2d_end = std::chrono::high_resolution_clock::now();
    if (g_enable_timing) {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(host_h2d_end - host_h2d_start).count();
        std::cerr << "[TIMER] Data transfer to device: " << us << " us" << std::endl;
    }

    hipEvent_t ev_start, ev_stop;
    if (g_enable_timing) {
        HIP_CHECK(hipEventCreate(&ev_start), "hipEventCreate ev_start");
        HIP_CHECK(hipEventCreate(&ev_stop), "hipEventCreate ev_stop");
        HIP_CHECK(hipEventRecord(ev_start, 0), "hipEventRecord ev_start");
    }
    device_scan_inclusive(d_in, d_out, N);
    if (g_enable_timing) {
        HIP_CHECK(hipEventRecord(ev_stop, 0), "hipEventRecord ev_stop");
        HIP_CHECK(hipEventSynchronize(ev_stop), "hipEventSynchronize ev_stop");
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop), "hipEventElapsedTime");
        long long us = static_cast<long long>(ms * 1000.0f);
        std::cerr << "[TIMER] GPU computation(Solve): " << us << " us" << std::endl;
        HIP_CHECK(hipEventDestroy(ev_start), "hipEventDestroy ev_start");
        HIP_CHECK(hipEventDestroy(ev_stop), "hipEventDestroy ev_stop");
    }

    auto host_d2h_start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipMemcpy(output, d_out, sizeof(int) * (size_t)N, hipMemcpyDeviceToHost), "hipMemcpy D2H");
    HIP_CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize after D2H");
    auto host_d2h_end = std::chrono::high_resolution_clock::now();
    if (g_enable_timing) {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(host_d2h_end - host_d2h_start).count();
        std::cerr << "[TIMER] Data transfer from device: " << us << " us" << std::endl;
    }

    auto host_free_start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipFree(d_in), "hipFree d_in");
    HIP_CHECK(hipFree(d_out), "hipFree d_out");
    auto host_free_end = std::chrono::high_resolution_clock::now();
    if (g_enable_timing) {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(host_free_end - host_free_start).count();
        std::cerr << "[TIMER] GPU memory cleanup: " << us << " us" << std::endl;
    }
}