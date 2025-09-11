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
#include <chrono>
#include <string>

// Constants
static constexpr int INF = 1073741823; // 2^30 - 1
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// Fast APSP Constants
#ifndef BATCH_SSSP
#define BATCH_SSSP 1
#endif
#ifndef ADAPTIVE_K
#define ADAPTIVE_K 1
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 64
#endif
#ifndef NUM_COMPONENTS
#define NUM_COMPONENTS 64  // Default number of components for graph partitioning
#endif

// Edge structure for storing original graph edges
struct Edge {
    int src;
    int dst;
    int weight;
    
    Edge() : src(-1), dst(-1), weight(INF) {}
    Edge(int s, int d, int w) : src(s), dst(d), weight(w) {}
};

// Component information structure
struct ComponentInfo {
    std::vector<int> all_vertices;        // All vertices in this component
    std::vector<int> interior_vertices;   // Non-boundary vertices in this component  
    std::vector<int> boundary_vertices;   // Boundary vertices in this component
    int size() const { return static_cast<int>(all_vertices.size()); }
    int interior_size() const { return static_cast<int>(interior_vertices.size()); }
    int boundary_size() const { return static_cast<int>(boundary_vertices.size()); }
};

// CSR (Compressed Sparse Row) format structure
struct CSRGraph {
    std::vector<int> row_ptr;    // Size: V+1, row_ptr[i] = start index of vertex i's edges
    std::vector<int> col_idx;    // Size: E, column indices of edges
    std::vector<int> weights;    // Size: E, weights of edges
    int num_vertices;
    int num_edges;
    
    CSRGraph() : num_vertices(0), num_edges(0) {}
    CSRGraph(int V, int E) : num_vertices(V), num_edges(E) {
        row_ptr.resize(V + 1);
        col_idx.resize(E);
        weights.resize(E);
    }
};

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

// Helpers to consume nodiscard-returning APIs without warnings
static inline void hipSyncCheck(const char* msg){
    hipError_t st = hipDeviceSynchronize();
    if(st != hipSuccess){
        std::fprintf(stderr, "HIP Error: %s: %s\n", msg, hipGetErrorString(st));
        std::exit(1);
    }
}

static inline void hipFreeCheck(void* p, const char* name){
    if(!p) return;
    hipError_t st = hipFree(p);
    if(st != hipSuccess){
        std::fprintf(stderr, "HIP Error: hipFree %s: %s\n", name, hipGetErrorString(st));
        // Continue on free error to avoid masking earlier results
    }
}

#endif 