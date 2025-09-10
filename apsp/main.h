#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <hip/hip_runtime.h>
#include <fstream>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

// Constants
static constexpr int INF = 1073741823; // 2^30 - 1
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// Row-major index helper
static inline __host__ __device__ size_t idx_rc(int row, int col, int n){
    return static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col);
}

// HIP error checking
static inline void hipCheck(hipError_t status, const char* msg){
    if(status != hipSuccess){
        std::fprintf(stderr, "HIP Error: %s: %s\n", msg, hipGetErrorString(status));
        std::exit(1);
    }
}

#endif 