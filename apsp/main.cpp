#include "main.h"
#include "memory_pool.h"
#include <charconv>

__device__ __forceinline__ int min_plus(int a, int b){
    // Fast path for the most common case
    if(__builtin_expect(a < INF && b < INF, 1)) {
        long long s = static_cast<long long>(a) + static_cast<long long>(b);
        return __builtin_expect(s <= INF, 1) ? static_cast<int>(s) : INF;
    }
    return INF;
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
    #pragma unroll
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
        #pragma unroll
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
        #pragma unroll
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
        #pragma unroll
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

static bool read_graph_pinned(const char* path, int& V, int& E, int* h_dist_pinned){
    std::ifstream fin(path);
    if(!fin.is_open()) return false;
    if(!(fin >> V >> E)) return false;
    if(V <= 0){ return false; }
    
    size_t total_elements = static_cast<size_t>(V) * static_cast<size_t>(V);
    // Initialize with INF
    for(size_t i = 0; i < total_elements; ++i){
        h_dist_pinned[i] = INF;
    }
    // Set diagonal to 0
    for(int i=0;i<V;++i){ 
        h_dist_pinned[idx_rc(i,i,V)] = 0; 
    }
    // Read edges
    for(int e=0;e<E;++e){
        int s,t,w; fin >> s >> t >> w;
        if(s>=0 && s<V && t>=0 && t<V){
            size_t p = idx_rc(s,t,V);
            if(w < h_dist_pinned[p]) h_dist_pinned[p] = w;
        }
    }
    return true;
}

static void print_matrix(const std::vector<int>& dist, int V){
#if defined(FAST_OUTPUT) && FAST_OUTPUT
    const size_t BUF_SIZE = static_cast<size_t>(32) * 1024 * 1024; // 32MB
    std::vector<char> buffer(BUF_SIZE);
    char* const buf_begin = buffer.data();
    char* const buf_end = buffer.data() + buffer.size();
    char* p = buffer.data();

    for(int i=0;i<V;++i){
        for(int j=0;j<V;++j){
            if(j){
                if(p >= buf_end){
                    std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
                    p = const_cast<char*>(buf_begin);
                }
                *p++ = ' ';
            }
            // Reserve enough space for number (up to 10 digits) and possible newline later
            if(p + 16 > buf_end){
                std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
                p = const_cast<char*>(buf_begin);
            }
            int val = dist[idx_rc(i,j,V)];
            auto conv = std::to_chars(p, buf_end, val);
            // to_chars on base-10 for int should never fail given we ensured capacity
            p = conv.ptr;
        }
        if(p >= buf_end){
            std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
            p = const_cast<char*>(buf_begin);
        }
        *p++ = '\n';
        // Optional: flush per row only when buffer is large; we keep accumulating
    }
    if(p > buf_begin){
        std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
    }
    std::fflush(stdout);
#else
    std::ios::fmtflags f(std::cout.flags());
    for(int i=0;i<V;++i){
        for(int j=0;j<V;++j){
            if(j) std::cout << ' ';
            std::cout << dist[idx_rc(i,j,V)];
        }
        std::cout << '\n';
    }
    std::cout.flags(f);
#endif
}

static void print_matrix_pinned(const int* dist, int V){
#if defined(FAST_OUTPUT) && FAST_OUTPUT
    const size_t BUF_SIZE = static_cast<size_t>(32) * 1024 * 1024; // 32MB
    
    // Use stack allocator for small outputs, heap for large ones
    constexpr size_t STACK_BUF_SIZE = 2 * 1024 * 1024; // 2MB stack buffer
    static thread_local StackAllocator<STACK_BUF_SIZE> stack_alloc;
    
    char* buf_begin = nullptr;
    char* buf_end = nullptr;
    bool use_stack = (BUF_SIZE <= STACK_BUF_SIZE);
    
    std::vector<char> heap_buffer; // Only allocated if needed
    
    if(use_stack) {
        stack_alloc.reset(); // Reset stack allocator
        buf_begin = static_cast<char*>(stack_alloc.allocate(BUF_SIZE, 64));
        if(buf_begin) {
            buf_end = buf_begin + BUF_SIZE;
        } else {
            use_stack = false; // Fall back to heap if stack allocation fails
        }
    }
    
    if(!use_stack) {
        heap_buffer.resize(BUF_SIZE);
        buf_begin = heap_buffer.data();
        buf_end = heap_buffer.data() + heap_buffer.size();
    }
    
    char* p = buf_begin;

    for(int i=0;i<V;++i){
        for(int j=0;j<V;++j){
            if(j){
                if(p >= buf_end){
                    std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
                    p = const_cast<char*>(buf_begin);
                }
                *p++ = ' ';
            }
            // Reserve enough space for number (up to 10 digits) and possible newline later
            if(p + 16 > buf_end){
                std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
                p = const_cast<char*>(buf_begin);
            }
            int val = dist[idx_rc(i,j,V)];
            auto conv = std::to_chars(p, buf_end, val);
            // to_chars on base-10 for int should never fail given we ensured capacity
            p = conv.ptr;
        }
        if(p >= buf_end){
            std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
            p = const_cast<char*>(buf_begin);
        }
        *p++ = '\n';
        // Optional: flush per row only when buffer is large; we keep accumulating
    }
    if(p > buf_begin){
        std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
    }
    std::fflush(stdout);
#else
    std::ios::fmtflags f(std::cout.flags());
    for(int i=0;i<V;++i){
        for(int j=0;j<V;++j){
            if(j) std::cout << ' ';
            std::cout << dist[idx_rc(i,j,V)];
        }
        std::cout << '\n';
    }
    std::cout.flags(f);
#endif
}

int main(int argc, char* argv[]){
    if(argc < 2){
        std::fprintf(stderr, "Error: missing input file.\n");
        return 1;
    }
    
    // Parse command line arguments
    bool enable_timing = false;
    for(int i = 2; i < argc; ++i){
        if(std::string(argv[i]) == "--timing"){
            enable_timing = true;
            break;
        }
    }
#if defined(FAST_OUTPUT) && FAST_OUTPUT
    // Accelerate stdout I/O
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    // Use a large fully-buffered stdout to reduce syscalls when redirected to file
    setvbuf(stdout, nullptr, _IOFBF, 32 * 1024 * 1024);
#endif
    int V = 0, E = 0;
    
    // First pass: read graph dimensions only 
    {
        std::ifstream fin(argv[1]);
        if(!fin.is_open() || !(fin >> V >> E) || V <= 0){
            std::fprintf(stderr, "Error: failed to read input file dimensions.\n");
            return 1;
        }
    }
    
    // Allocate page-locked (pinned) host memory using memory pool
    size_t bytes = static_cast<size_t>(V) * static_cast<size_t>(V) * sizeof(int);
    
    auto start_host_alloc = std::chrono::high_resolution_clock::now();
    GlobalMemoryPool& pool = GlobalMemoryPool::getInstance();
    int* h_dist_pinned = static_cast<int*>(pool.allocateHostMemory(bytes));
    if(!h_dist_pinned){
        std::fprintf(stderr, "Error: failed to allocate pinned host memory from pool.\n");
        return 1;
    }
    auto end_host_alloc = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto host_alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_host_alloc - start_host_alloc);
        std::cerr << "[TIMER] Pinned host memory allocation (pool): " << host_alloc_duration.count() << " us" << std::endl;
    }
    
    // Timer for data loading to host
    auto start_load = std::chrono::high_resolution_clock::now();
    if(!read_graph_pinned(argv[1], V, E, h_dist_pinned)){
        std::fprintf(stderr, "Error: failed to read input file.\n");
        hipCheck(hipHostFree(h_dist_pinned), "hipHostFree on error");
        return 1;
    }
    auto end_load = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_load - start_load);
        std::cerr << "[TIMER] Data loading to pinned host memory: " << load_duration.count() << " us" << std::endl;
    }
    const int B = BLOCK_SIZE;
    const int nTiles = (V + B - 1) / B;
    
    // Get global HIP stream for async operations (reuse existing stream)
    auto start_stream = std::chrono::high_resolution_clock::now();
    hipStream_t stream = pool.getStream();
    auto end_stream = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto stream_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_stream - start_stream);
        std::cerr << "[TIMER] HIP stream acquisition (global): " << stream_duration.count() << " us" << std::endl;
    }
    
    // Timer for GPU memory allocation using memory pool
    auto start_alloc = std::chrono::high_resolution_clock::now();
    int* d_dist = static_cast<int*>(pool.allocateDeviceMemory(bytes));
    if(!d_dist){
        std::fprintf(stderr, "Error: failed to allocate GPU memory from pool.\n");
        pool.deallocateHostMemory(h_dist_pinned);
        return 1;
    }
    auto end_alloc = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_alloc - start_alloc);
        std::cerr << "[TIMER] GPU memory allocation (pool): " << alloc_duration.count() << " us" << std::endl;
    }
    
    // Timer for async data transfer to device (pinned memory H2D)
    auto start_h2d = std::chrono::high_resolution_clock::now();
    hipCheck(hipMemcpyAsync(d_dist, h_dist_pinned, bytes, hipMemcpyHostToDevice, stream), "hipMemcpyAsync H2D pinned");
    // Note: We don't synchronize here yet to allow potential overlap with other operations
    auto end_h2d_launch = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto h2d_launch_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_h2d_launch - start_h2d);
        std::cerr << "[TIMER] Async H2D transfer launch: " << h2d_launch_duration.count() << " us" << std::endl;
    }
    
    // Synchronize stream to ensure H2D transfer completes before GPU computation
    auto start_h2d_sync = std::chrono::high_resolution_clock::now();
    hipCheck(hipStreamSynchronize(stream), "hipStreamSynchronize H2D");
    auto end_h2d_sync = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto h2d_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_h2d_sync - start_h2d);
        auto h2d_sync_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_h2d_sync - start_h2d_sync);
        double transfer_speed_mbps = (bytes / 1024.0 / 1024.0) / (h2d_total_duration.count() / 1e6);
        std::cerr << "[TIMER] Async H2D transfer total: " << h2d_total_duration.count() << " us" 
                  << " (sync: " << h2d_sync_duration.count() << " us, " 
                  << std::fixed << std::setprecision(2) << transfer_speed_mbps << " MB/s)" << std::endl;
    }
    dim3 block(B,B,1);
    if(B*B > 1024){ block.x = 32; block.y = 32; }
    
    // Timer for GPU computation
    auto start_gpu = std::chrono::high_resolution_clock::now();
    
    // Create events for efficient synchronization
    hipEvent_t phase1_complete, phase2_complete;
    hipCheck(hipEventCreate(&phase1_complete), "hipEventCreate phase1");
    hipCheck(hipEventCreate(&phase2_complete), "hipEventCreate phase2");
    
    for(int k=0;k<nTiles;++k){
        // Phase 1: Process pivot block
        {
            dim3 grid(1,1,1);
            size_t shmem = (size_t)B * (size_t)B * sizeof(int);
            hipLaunchKernelGGL(fw_phase1, grid, block, shmem, stream, d_dist, V, k, B);
            hipEventRecord(phase1_complete, stream);
        }
        
        // Phase 2: Process row and column blocks (depends on phase1)
        {
            hipStreamWaitEvent(stream, phase1_complete, 0);
            dim3 grid(nTiles, 2, 1);
            size_t shmem = 2ull * B * B * sizeof(int);
            hipLaunchKernelGGL(fw_phase2, grid, block, shmem, stream, d_dist, V, k, B);
            hipEventRecord(phase2_complete, stream);
        }
        
        // Phase 3: Process remaining blocks (depends on phase2)
        {
            hipStreamWaitEvent(stream, phase2_complete, 0);
            dim3 grid(nTiles, nTiles, 1);
            size_t shmem = 2ull * B * B * sizeof(int);
            hipLaunchKernelGGL(fw_phase3, grid, block, shmem, stream, d_dist, V, k, B);
        }
    }
    
    // Only synchronize once at the end of all computation
    hipCheck(hipStreamSynchronize(stream), "hipStreamSynchronize GPU computation");
    
    // Clean up events
    hipEventDestroy(phase1_complete);
    hipEventDestroy(phase2_complete);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
        std::cerr << "[TIMER] GPU computation: " << gpu_duration.count() << " us" << std::endl;
    }
    
    // Timer for async data transfer from device (pinned memory D2H)
    auto start_d2h = std::chrono::high_resolution_clock::now();
    hipCheck(hipMemcpyAsync(h_dist_pinned, d_dist, bytes, hipMemcpyDeviceToHost, stream), "hipMemcpyAsync D2H pinned");
    auto end_d2h_launch = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto d2h_launch_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_d2h_launch - start_d2h);
        std::cerr << "[TIMER] Async D2H transfer launch: " << d2h_launch_duration.count() << " us" << std::endl;
    }
    
    // Synchronize stream to ensure D2H transfer completes 
    auto start_d2h_sync = std::chrono::high_resolution_clock::now();
    hipCheck(hipStreamSynchronize(stream), "hipStreamSynchronize D2H");
    auto end_d2h_sync = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto d2h_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_d2h_sync - start_d2h);
        auto d2h_sync_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_d2h_sync - start_d2h_sync);
        double transfer_speed_mbps = (bytes / 1024.0 / 1024.0) / (d2h_total_duration.count() / 1e6);
        std::cerr << "[TIMER] Async D2H transfer total: " << d2h_total_duration.count() << " us" 
                  << " (sync: " << d2h_sync_duration.count() << " us, " 
                  << std::fixed << std::setprecision(2) << transfer_speed_mbps << " MB/s)" << std::endl;
    }
    
    // Timer for GPU memory cleanup (return to pool)
    auto start_cleanup = std::chrono::high_resolution_clock::now();
    pool.deallocateDeviceMemory(d_dist);
    auto end_cleanup = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto cleanup_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cleanup - start_cleanup);
        std::cerr << "[TIMER] GPU memory cleanup (pool): " << cleanup_duration.count() << " us" << std::endl;
    }
    
    // Timer for result output
    auto start_output = std::chrono::high_resolution_clock::now();
    print_matrix_pinned(h_dist_pinned, V);
    auto end_output = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto output_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_output - start_output);
        std::cerr << "[TIMER] Result output: " << output_duration.count() << " us" << std::endl;
    }
    
    // Timer for pinned host memory cleanup (return to pool)
    auto start_host_cleanup = std::chrono::high_resolution_clock::now();
    pool.deallocateHostMemory(h_dist_pinned);
    auto end_host_cleanup = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto host_cleanup_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_host_cleanup - start_host_cleanup);
        std::cerr << "[TIMER] Pinned host memory cleanup (pool): " << host_cleanup_duration.count() << " us" << std::endl;
    }
    
    // HIP stream is global - no cleanup needed (reused for future calls)
    if(enable_timing){
        std::cerr << "[INFO] HIP stream is global and reused - no cleanup overhead" << std::endl;
    }
    
    return 0;
}