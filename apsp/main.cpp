#include "main.h"

__device__ __forceinline__ int min_plus(int a, int b){
    if(a >= INF || b >= INF) return INF;
    long long s = static_cast<long long>(a) + static_cast<long long>(b);
    if(s > INF) return INF;
    return static_cast<int>(s);
}

__global__ void fw_phase1(int* __restrict__ d, int n, int k, int B){
    extern __shared__ int sh[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = k * B + ty;
    int j = k * B + tx;
    if(i < n && j < n){
        sh[ty * B + tx] = d[idx_rc(i,j,n)];
    }else{
        sh[ty * B + tx] = INF;
    }
    __syncthreads();
    for(int m=0;m<B;++m){
        int via = sh[ty * B + m];
        int to = sh[m * B + tx];
        int cur = sh[ty * B + tx];
        int cand = min_plus(via, to);
        if(cand < cur) sh[ty * B + tx] = cand;
        __syncthreads();
    }
    if(i < n && j < n){
        d[idx_rc(i,j,n)] = sh[ty * B + tx];
    }
}

__global__ void fw_phase2(int* __restrict__ d, int n, int k, int B){
    extern __shared__ int sh[];
    int* pivot = sh;
    int* other = sh + B*B;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int which = blockIdx.y;

    {
        int i = k * B + ty;
        int j = k * B + tx;
        if(i < n && j < n){
            pivot[ty*B+tx] = d[idx_rc(i,j,n)];
        }else{
            pivot[ty*B+tx] = INF;
        }
    }
    __syncthreads();

    if(which == 0){
        int jtile = bx; if(jtile == k) return;
        int gi = k * B + ty;
        int gj = jtile * B + tx;
        if(gi < n && gj < n){
            other[ty*B+tx] = d[idx_rc(gi,gj,n)];
        }else{
            other[ty*B+tx] = INF;
        }
        __syncthreads();
        for(int m=0;m<B;++m){
            int via = pivot[ty*B+m];
            int to = other[m*B+tx];
            int cur = other[ty*B+tx];
            int cand = min_plus(via, to);
            if(cand < cur) other[ty*B+tx] = cand;
            __syncthreads();
        }
        if(gi < n && gj < n){
            d[idx_rc(gi,gj,n)] = other[ty*B+tx];
        }
    }else{
        int itile = bx; if(itile == k) return;
        int gi = itile * B + ty;
        int gj = k * B + tx;
        if(gi < n && gj < n){
            other[ty*B+tx] = d[idx_rc(gi,gj,n)];
        }else{
            other[ty*B+tx] = INF;
        }
        __syncthreads();
        for(int m=0;m<B;++m){
            int via = other[ty*B+m];
            int to = pivot[m*B+tx];
            int cur = other[ty*B+tx];
            int cand = min_plus(via, to);
            if(cand < cur) other[ty*B+tx] = cand;
            __syncthreads();
        }
        if(gi < n && gj < n){
            d[idx_rc(gi,gj,n)] = other[ty*B+tx];
        }
    }
}

__global__ void fw_phase3(int* __restrict__ d, int n, int k, int B){
    extern __shared__ int sh[];
    int* rowk = sh;
    int* colk = sh + B*B;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int itile = blockIdx.y;
    int jtile = blockIdx.x;
    if(itile == k || jtile == k) return;
    int gi = itile * B + ty;
    int gj = jtile * B + tx;
    {
        int ri = k * B + ty;
        int rj = jtile * B + tx;
        if(ri < n && rj < n){
            rowk[ty*B+tx] = d[idx_rc(ri,rj,n)];
        }else{
            rowk[ty*B+tx] = INF;
        }
    }
    {
        int ci = itile * B + ty;
        int cj = k * B + tx;
        if(ci < n && cj < n){
            colk[ty*B+tx] = d[idx_rc(ci,cj,n)];
        }else{
            colk[ty*B+tx] = INF;
        }
    }
    __syncthreads();
    if(gi < n && gj < n){
        int cur = d[idx_rc(gi,gj,n)];
        int best = cur;
        for(int m=0;m<B;++m){
            int via = colk[ty*B+m];
            int to = rowk[m*B+tx];
            int cand = min_plus(via, to);
            if(cand < best) best = cand;
        }
        d[idx_rc(gi,gj,n)] = best;
    }
}

static bool read_graph(const char* path, int& V, int& E, std::vector<int>& dist){
    std::ifstream fin(path);
    if(!fin.is_open()) return false;
    if(!(fin >> V >> E)) return false;
    if(V <= 0){ return false; }
    dist.assign(static_cast<size_t>(V) * static_cast<size_t>(V), INF);
    for(int i=0;i<V;++i){ dist[idx_rc(i,i,V)] = 0; }
    for(int e=0;e<E;++e){
        int s,t,w; fin >> s >> t >> w;
        if(s>=0 && s<V && t>=0 && t<V){
            size_t p = idx_rc(s,t,V);
            if(w < dist[p]) dist[p] = w;
        }
    }
    return true;
}

static void print_matrix(const std::vector<int>& dist, int V){
    std::ios::fmtflags f(std::cout.flags());
    for(int i=0;i<V;++i){
        for(int j=0;j<V;++j){
            if(j) std::cout << ' ';
            std::cout << dist[idx_rc(i,j,V)];
        }
        std::cout << '\n';
    }
    std::cout.flags(f);
}

int main(int argc, char* argv[]){
    if(argc < 2){
        std::fprintf(stderr, "Error: missing input file.\n");
        return 1;
    }
    int V = 0, E = 0;
    std::vector<int> h_dist;
    if(!read_graph(argv[1], V, E, h_dist)){
        std::fprintf(stderr, "Error: failed to read input file.\n");
        return 1;
    }
    const int B = BLOCK_SIZE;
    const int nTiles = (V + B - 1) / B;
    int* d_dist = nullptr;
    size_t bytes = static_cast<size_t>(V) * static_cast<size_t>(V) * sizeof(int);
    hipCheck(hipMalloc(&d_dist, bytes), "hipMalloc d_dist");
    hipCheck(hipMemcpy(d_dist, h_dist.data(), bytes, hipMemcpyHostToDevice), "hipMemcpy H2D");
    dim3 block(B,B,1);
    if(B*B > 1024){ block.x = 32; block.y = 32; }
    for(int k=0;k<nTiles;++k){
        {
            dim3 grid(1,1,1);
            size_t shmem = (size_t)B * (size_t)B * sizeof(int);
            hipLaunchKernelGGL(fw_phase1, grid, block, shmem, 0, d_dist, V, k, B);
            hipError_t err1 = hipGetLastError();
            if(err1 != hipSuccess){ std::fprintf(stderr, "Kernel fw_phase1 launch failed: %s\n", hipGetErrorString(err1)); return 1; }
            if(hipDeviceSynchronize() != hipSuccess){ std::fprintf(stderr, "Kernel fw_phase1 sync failed\n"); return 1; }
        }
        {
            dim3 grid(nTiles, 2, 1);
            size_t shmem = 2ull * B * B * sizeof(int);
            hipLaunchKernelGGL(fw_phase2, grid, block, shmem, 0, d_dist, V, k, B);
            hipError_t err2 = hipGetLastError();
            if(err2 != hipSuccess){ std::fprintf(stderr, "Kernel fw_phase2 launch failed: %s\n", hipGetErrorString(err2)); return 1; }
            if(hipDeviceSynchronize() != hipSuccess){ std::fprintf(stderr, "Kernel fw_phase2 sync failed\n"); return 1; }
        }
        {
            dim3 grid(nTiles, nTiles, 1);
            size_t shmem = 2ull * B * B * sizeof(int);
            hipLaunchKernelGGL(fw_phase3, grid, block, shmem, 0, d_dist, V, k, B);
            hipError_t err3 = hipGetLastError();
            if(err3 != hipSuccess){ std::fprintf(stderr, "Kernel fw_phase3 launch failed: %s\n", hipGetErrorString(err3)); return 1; }
            if(hipDeviceSynchronize() != hipSuccess){ std::fprintf(stderr, "Kernel fw_phase3 sync failed\n"); return 1; }
        }
    }
    hipCheck(hipMemcpy(h_dist.data(), d_dist, bytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");
    hipFree(d_dist);
    print_matrix(h_dist, V);
    return 0;
}