#include "main.h"
#include <hip/hip_runtime.h>
#include <algorithm>

#define BLOCK_SIZE 1024

__global__ void find_max(const float* input, float* max_val, int N) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = -FLT_MAX;
    if (i < N) val = input[i];
    sdata[tid] = val;
    __syncthreads();

    // reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) max_val[blockIdx.x] = sdata[0];
}

__global__ void exp_sum(const float* input, float m, float* sum_val, int N) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (i < N) val = expf(input[i] - m);
    sdata[tid] = val;
    __syncthreads();

    // reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) sum_val[blockIdx.x] = sdata[0];
}

__global__ void softmax_kernel(const float* input, float* output, float m, float S, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float t = expf(input[i] - m);
        output[i] = t / (S > 1e-12f ? S : 1e-12f);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float *d_input, *d_output;
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);

    // 1. find max
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *d_max, *h_max = new float[grid];
    hipMalloc(&d_max, grid * sizeof(float));
    hipLaunchKernelGGL(find_max, dim3(grid), dim3(BLOCK_SIZE), 0, 0, d_input, d_max, N);
    hipMemcpy(h_max, d_max, grid * sizeof(float), hipMemcpyDeviceToHost);
    float m = -FLT_MAX;
    for (int i = 0; i < grid; ++i) m = std::max(m, h_max[i]);

    // 2. sum exp
    float *d_sum, *h_sum = new float[grid];
    hipMalloc(&d_sum, grid * sizeof(float));
    hipLaunchKernelGGL(exp_sum, dim3(grid), dim3(BLOCK_SIZE), 0, 0, d_input, m, d_sum, N);
    hipMemcpy(h_sum, d_sum, grid * sizeof(float), hipMemcpyDeviceToHost);
    float S = 0.0f;
    for (int i = 0; i < grid; ++i) S += h_sum[i];

    // 3. softmax
    hipLaunchKernelGGL(softmax_kernel, dim3(grid), dim3(BLOCK_SIZE), 0, 0, d_input, d_output, m, S, N);
    hipMemcpy(output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_max);
    hipFree(d_sum);
    delete[] h_max;
    delete[] h_sum;
}