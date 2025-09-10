#include "main.h"
#include <chrono>
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

    std::chrono::steady_clock::time_point t_kernel_start, t_kernel_end;
    if (g_enable_timing) t_kernel_start = std::chrono::steady_clock::now();
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
    if (g_enable_timing) { t_kernel_end = std::chrono::steady_clock::now(); print_timing(g_scan_impl == 1 ? "block scan (blelloch) kernel" : "block scan (hs) kernel", std::chrono::duration<double, std::milli>(t_kernel_end - t_kernel_start).count()); }

    if (numBlocks > 1) {
        int* d_scanned_block_sums = nullptr;
        HIP_CHECK(hipMalloc(&d_scanned_block_sums, sizeof(int) * numBlocks), "hipMalloc d_scanned_block_sums");
        // recursively scan block sums
        std::chrono::steady_clock::time_point t_rec_start, t_rec_end;
        if (g_enable_timing) t_rec_start = std::chrono::steady_clock::now();
        device_scan_inclusive(d_block_sums, d_scanned_block_sums, numBlocks);
        if (g_enable_timing) { t_rec_end = std::chrono::steady_clock::now(); print_timing("scan block_sums (recursive)", std::chrono::duration<double, std::milli>(t_rec_end - t_rec_start).count()); }

        std::chrono::steady_clock::time_point t_add_start, t_add_end;
        if (g_enable_timing) t_add_start = std::chrono::steady_clock::now();
        hipLaunchKernelGGL(add_block_offsets_kernel,
                           dim3(numBlocks), dim3(BLOCK_SIZE), 0, 0,
                           d_out, d_scanned_block_sums, length);
        HIP_CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize after add_block_offsets_kernel");
        if (g_enable_timing) { t_add_end = std::chrono::steady_clock::now(); print_timing("add offsets kernel", std::chrono::duration<double, std::milli>(t_add_end - t_add_start).count()); }

        HIP_CHECK(hipFree(d_scanned_block_sums), "hipFree d_scanned_block_sums");
        HIP_CHECK(hipFree(d_block_sums), "hipFree d_block_sums");
    }
}

extern "C" void solve(const int* input, int* output, int N) {
    if (N <= 0) return;

    int *d_in = nullptr, *d_out = nullptr;
    std::chrono::steady_clock::time_point t_alloc_start, t_alloc_end;
    if (g_enable_timing) t_alloc_start = std::chrono::steady_clock::now();
    HIP_CHECK(hipMalloc(&d_in, sizeof(int) * (size_t)N), "hipMalloc d_in");
    HIP_CHECK(hipMalloc(&d_out, sizeof(int) * (size_t)N), "hipMalloc d_out");
    if (g_enable_timing) { t_alloc_end = std::chrono::steady_clock::now(); print_timing("device alloc", std::chrono::duration<double, std::milli>(t_alloc_end - t_alloc_start).count()); }

    std::chrono::steady_clock::time_point t_h2d_start, t_h2d_end;
    if (g_enable_timing) t_h2d_start = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpy(d_in, input, sizeof(int) * (size_t)N, hipMemcpyHostToDevice), "hipMemcpy H2D");
    HIP_CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize after H2D");
    if (g_enable_timing) { t_h2d_end = std::chrono::steady_clock::now(); print_timing("H2D copy", std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count()); }

    std::chrono::steady_clock::time_point t_scan_start, t_scan_end;
    if (g_enable_timing) t_scan_start = std::chrono::steady_clock::now();
    device_scan_inclusive(d_in, d_out, N);
    if (g_enable_timing) { t_scan_end = std::chrono::steady_clock::now(); print_timing("device_scan_inclusive total", std::chrono::duration<double, std::milli>(t_scan_end - t_scan_start).count()); }

    std::chrono::steady_clock::time_point t_d2h_start, t_d2h_end;
    if (g_enable_timing) t_d2h_start = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpy(output, d_out, sizeof(int) * (size_t)N, hipMemcpyDeviceToHost), "hipMemcpy D2H");
    HIP_CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize after D2H");
    if (g_enable_timing) { t_d2h_end = std::chrono::steady_clock::now(); print_timing("D2H copy", std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count()); }

    HIP_CHECK(hipFree(d_in), "hipFree d_in");
    HIP_CHECK(hipFree(d_out), "hipFree d_out");
}