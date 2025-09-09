#include "main.h"

// 简单多层并行扫描：
// 1) 块内 inclusive 扫描并输出，每块记录块和
// 2) 递归地对块和数组做扫描
// 3) 将上一步得到的偏移（块和扫描的前一项）加回到对应块的元素上

#define BLOCK_SIZE 256

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
        hipMalloc(&d_block_sums, sizeof(int) * numBlocks);
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
    hipDeviceSynchronize();

    if (numBlocks > 1) {
        int* d_scanned_block_sums = nullptr;
        hipMalloc(&d_scanned_block_sums, sizeof(int) * numBlocks);
        device_scan_inclusive(d_block_sums, d_scanned_block_sums, numBlocks);

        hipLaunchKernelGGL(add_block_offsets_kernel,
                           dim3(numBlocks), dim3(BLOCK_SIZE), 0, 0,
                           d_out, d_scanned_block_sums, length);
        hipDeviceSynchronize();

        hipFree(d_scanned_block_sums);
        hipFree(d_block_sums);
    }
}

extern "C" void solve(const int* input, int* output, int N) {
    if (N <= 0) return;

    int *d_in = nullptr, *d_out = nullptr;
    hipMalloc(&d_in, sizeof(int) * (size_t)N);
    hipMalloc(&d_out, sizeof(int) * (size_t)N);

    hipMemcpy(d_in, input, sizeof(int) * (size_t)N, hipMemcpyHostToDevice);

    device_scan_inclusive(d_in, d_out, N);

    hipMemcpy(output, d_out, sizeof(int) * (size_t)N, hipMemcpyDeviceToHost);

    hipFree(d_in);
    hipFree(d_out);
}