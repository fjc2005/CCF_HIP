#include "main.h"
#include <charconv>

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

// SSSP initialization kernel
__global__ void initialize_sssp_kernel(
    int* __restrict__ d_distances,     // Distance array to initialize (V elements)
    bool* __restrict__ d_frontier,     // Frontier mask to initialize (V elements)
    int source_vertex,                 // Source vertex for this SSSP instance
    int V)                            // Number of vertices
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    
    // Initialize distances: source = 0, others = INF
    d_distances[tid] = (tid == source_vertex) ? 0 : INF;
    
    // Initialize frontier: only source vertex is in initial frontier
    d_frontier[tid] = (tid == source_vertex);
}

// SSSP frontier-based relaxation kernel
__global__ void sssp_kernel(
    const int* __restrict__ d_csr_row_ptr, 
    const int* __restrict__ d_csr_col_idx, 
    const int* __restrict__ d_csr_weights,
    int* __restrict__ d_distances,       // Current SSSP distance array (V elements)
    const bool* __restrict__ d_frontier, // Current frontier boolean mask (V elements)
    bool* __restrict__ d_next_frontier,  // Next frontier to write to (V elements)
    bool* __restrict__ d_is_frontier_active, // Single boolean: is next frontier non-empty?
    int V)                               // Number of vertices
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V) return;

    // If current vertex u is not in frontier, do nothing
    if (!d_frontier[u]) {
        return;
    }

    // Otherwise, iterate through all neighbors v of u
    int start_edge = d_csr_row_ptr[u];
    int end_edge = d_csr_row_ptr[u + 1];

    for (int edge_idx = start_edge; edge_idx < end_edge; ++edge_idx) {
        int v = d_csr_col_idx[edge_idx];
        int weight = d_csr_weights[edge_idx];
        
        int new_dist = min_plus(d_distances[u], weight);
        
        // Use atomic operation to try relaxing edge (u,v)
        // atomicMin returns the OLD value
        int old_dist = atomicMin(&d_distances[v], new_dist);

        // If we successfully updated the distance, then v is in next frontier
        if (new_dist < old_dist) {
            d_next_frontier[v] = true;
            // Mark frontier as non-empty so host continues the loop
            *d_is_frontier_active = true;
        }
    }
}

#if BATCH_SSSP
// Batched SSSP initialization: initialize distances and frontier for a batch of sources
__global__ void initialize_batch_sssp_kernel(
    int* __restrict__ d_boundary_sssp_results, // num_total_boundary x V (row-major)
    uint8_t* __restrict__ d_frontier_batch,    // batch_size x V
    const int* __restrict__ d_sources_batch,   // batch_size
    int batch_offset,                          // global row offset in results
    int batch_size,                            // actual size of this batch
    int V)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if(s >= batch_size || v >= V) return;
    int global_row = batch_offset + s;
    int source_vertex = d_sources_batch[s];
    // set distance row
    d_boundary_sssp_results[idx_rc(global_row, v, V)] = (v == source_vertex) ? 0 : INF;
    // set frontier mask
    d_frontier_batch[s * V + v] = (v == source_vertex) ? 1 : 0;
}

// Batched frontier-based relaxation kernel
__global__ void sssp_batch_kernel(
    const int* __restrict__ d_csr_row_ptr,
    const int* __restrict__ d_csr_col_idx,
    const int* __restrict__ d_csr_weights,
    int* __restrict__ d_boundary_sssp_results, // num_total_boundary x V
    const uint8_t* __restrict__ d_frontier_batch, // batch_size x V
    uint8_t* __restrict__ d_next_frontier_batch,  // batch_size x V
    int* __restrict__ d_next_frontier_count,      // single int counter
    int batch_offset, int batch_size, int V)
{
    // 2D grid: x over vertices, y over sources in batch
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y; // one source per block row for simplicity
    if(s >= batch_size || u >= V) return;

    const uint8_t in_frontier = d_frontier_batch[s * V + u];
    if(!in_frontier) return;

    int global_row = batch_offset + s;
    int start_edge = d_csr_row_ptr[u];
    int end_edge = d_csr_row_ptr[u + 1];
    int du = d_boundary_sssp_results[idx_rc(global_row, u, V)];
    for(int edge_idx = start_edge; edge_idx < end_edge; ++edge_idx){
        int v = d_csr_col_idx[edge_idx];
        int w = d_csr_weights[edge_idx];
        int new_dist = min_plus(du, w);
        int* dv_ptr = &d_boundary_sssp_results[idx_rc(global_row, v, V)];
        int old = atomicMin(dv_ptr, new_dist);
        if(new_dist < old){
            d_next_frontier_batch[s * V + v] = 1;
            atomicAdd(d_next_frontier_count, 1);
        }
    }
}
#endif

// Gather kernel: collect component data from global distance matrix to local small matrix
__global__ void gather_kernel(
    const int* __restrict__ d_dist_global,  // Global V x V distance matrix
    int* __restrict__ d_comp_dist,          // Component's small dense matrix
    const int* __restrict__ d_comp_members, // Component member list (global IDs)
    int n_comp,                             // Size of component
    int V_global)                           // Global number of vertices
{
    int local_i = blockIdx.y * blockDim.y + threadIdx.y;
    int local_j = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_i < n_comp && local_j < n_comp) {
        int global_i = d_comp_members[local_i];
        int global_j = d_comp_members[local_j];
        
        size_t global_idx = idx_rc(global_i, global_j, V_global);
        size_t local_idx = idx_rc(local_i, local_j, n_comp);

        d_comp_dist[local_idx] = d_dist_global[global_idx];
    }
}

// Scatter kernel: write local APSP results back to global distance matrix
__global__ void scatter_kernel(
    int* __restrict__ d_dist_global,        // Global V x V distance matrix
    const int* __restrict__ d_comp_dist,    // Component's computed small dense matrix
    const int* __restrict__ d_comp_members, // Component member list (global IDs)
    int n_comp,                             // Size of component
    int V_global)                           // Global number of vertices
{
    int local_i = blockIdx.y * blockDim.y + threadIdx.y;
    int local_j = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_i < n_comp && local_j < n_comp) {
        int global_i = d_comp_members[local_i];
        int global_j = d_comp_members[local_j];
        
        size_t global_idx = idx_rc(global_i, global_j, V_global);
        size_t local_idx = idx_rc(local_i, local_j, n_comp);

        d_dist_global[global_idx] = d_comp_dist[local_idx];
    }
}

// MIN-PLUS finalize kernel: combine SSSP and Local APSP results for cross-component paths
__global__ void min_plus_finalize_kernel(
    int* __restrict__ d_dist,                          // Global V x V matrix for reading and updating
    const int* __restrict__ d_boundary_sssp_results,   // SSSP results matrix (num_boundary x V)
    const int* __restrict__ d_comp_vertices,           // All vertices of current component (interior + boundary)
    const int* __restrict__ d_comp_boundary_vertices,  // Boundary vertices of current component
    const int* __restrict__ d_boundary_vertex_to_sssp_row, // Mapping from boundary vertex ID to SSSP row
    const bool* __restrict__ d_vertex_in_component,    // Boolean mask: is vertex in current component?
    int n_comp_vertices,                               // Number of vertices in component (interior + boundary)
    int n_boundary,                                    // Number of boundary vertices in component
    int V)                                             // Total number of vertices
{
    // Grid layout: (V, n_comp_vertices)
    // Each thread computes dist(u, v) for one component vertex u and one global vertex v
    int u_local_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int v_global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (u_local_idx >= n_comp_vertices || v_global_idx >= V) return;

    // Get component vertex's global ID
    int u_global_idx = d_comp_vertices[u_local_idx];
    
    // Skip if v is in the same component as u (already computed in Local APSP) - THIS IS WRONG
    // The check is removed because Local APSP cannot find paths that detour outside the component.
    // The min_plus_finalize_kernel is the correct place to discover these detour paths.
    // if (d_vertex_in_component[v_global_idx]) return;
    
    int min_dist = d_dist[idx_rc(u_global_idx, v_global_idx, V)];;

    // Iterate through all boundary vertices b of current component
    for (int i = 0; i < n_boundary; ++i) {
        int b_global_idx = d_comp_boundary_vertices[i];
        
        // 1. Get dist(u, b) from Local APSP results
        int dist_u_b = d_dist[idx_rc(u_global_idx, b_global_idx, V)];
        
        // 2. Get dist(b, v) from SSSP results
        int sssp_row = d_boundary_vertex_to_sssp_row[b_global_idx];
        if (sssp_row >= 0) {  // Check if b is indeed a boundary vertex
            int dist_b_v = d_boundary_sssp_results[idx_rc(sssp_row, v_global_idx, V)];
            
            // 3. Compute path distance through boundary vertex b
            int current_path_dist = min_plus(dist_u_b, dist_b_v);
            if (current_path_dist < min_dist) {
                min_dist = current_path_dist;
            }
        }
    }
    
    // Update global distance matrix with best cross-component path
    if (min_dist < d_dist[idx_rc(u_global_idx, v_global_idx, V)]) {
        atomicMin(&d_dist[idx_rc(u_global_idx, v_global_idx, V)], min_dist);
    }
}

// Convert edge list to CSR format
static void build_csr_graph(const std::vector<Edge>& edges, int V, CSRGraph& csr_graph){
    csr_graph = CSRGraph(V, static_cast<int>(edges.size()));
    
    // Count outgoing edges for each vertex
    std::vector<int> out_degree(V, 0);
    for(const auto& edge : edges){
        if(edge.src >= 0 && edge.src < V){
            out_degree[edge.src]++;
        }
    }
    
    // Build row_ptr array (prefix sum)
    csr_graph.row_ptr[0] = 0;
    for(int i = 0; i < V; ++i){
        csr_graph.row_ptr[i + 1] = csr_graph.row_ptr[i] + out_degree[i];
    }
    
    // Reset out_degree to use as current position counter
    std::fill(out_degree.begin(), out_degree.end(), 0);
    
    // Fill col_idx and weights arrays
    for(const auto& edge : edges){
        if(edge.src >= 0 && edge.src < V && edge.dst >= 0 && edge.dst < V){
            int pos = csr_graph.row_ptr[edge.src] + out_degree[edge.src];
            csr_graph.col_idx[pos] = edge.dst;
            csr_graph.weights[pos] = edge.weight;
            out_degree[edge.src]++;
        }
    }
}

static bool read_graph(const char* path, int& V, int& E, std::vector<int>& dist, std::vector<Edge>& edges){
    std::ifstream fin(path);
    if(!fin.is_open()) return false;
    if(!(fin >> V >> E)) return false;
    if(V <= 0){ return false; }
    
    // Initialize distance matrix
    dist.assign(static_cast<size_t>(V) * static_cast<size_t>(V), INF);
    for(int i=0;i<V;++i){ dist[idx_rc(i,i,V)] = 0; }
    
    // Reserve space for edges
    edges.clear();
    edges.reserve(E);
    
    // Read edges and populate both distance matrix and edge list
    for(int e=0;e<E;++e){
        int s,t,w; fin >> s >> t >> w;
        if(s>=0 && s<V && t>=0 && t<V){
            // Add to edge list
            edges.emplace_back(s, t, w);
            
            // Update distance matrix (keep minimum weight for duplicate edges)
            size_t p = idx_rc(s,t,V);
            if(w < dist[p]) dist[p] = w;
        }
    }
    return true;
}

static void print_matrix(const std::vector<int>& dist, int V){
#if FAST_OUTPUT
    // 使用大块缓冲与 to_chars 加速输出，确保与原格式一致
    // 选择块大小为多行聚合，目标每次 fwrite 至少 ~256KB
    constexpr size_t kTargetChunkBytes = 1u << 20; // 1 MB
    const size_t estimated_avg_digits = 11; // 最多 10 位 + 分隔符
    const size_t max_line_bytes = static_cast<size_t>(V) * (estimated_avg_digits + 1) + 1;
    const size_t chunk_lines = std::max<size_t>(1, kTargetChunkBytes / std::max<size_t>(1, max_line_bytes));
    const size_t buffer_bytes = std::max<size_t>(kTargetChunkBytes, max_line_bytes * std::min<size_t>(static_cast<size_t>(V), chunk_lines));
    std::vector<char> buffer(buffer_bytes);
    size_t write_pos = 0;

    auto flush_buffer = [&](bool final_flush){
        if(write_pos > 0){
            (void)std::fwrite(buffer.data(), 1, write_pos, stdout);
            write_pos = 0;
        }
        if(final_flush){ std::fflush(stdout); }
    };

    for(int i=0;i<V;++i){
        for(int j=0;j<V;++j){
            if(j){
                if(write_pos >= buffer.size()) flush_buffer(false);
                buffer[write_pos++] = ' ';
            }
            // 预留最多 11 字节（含可能的负数，不过此处非负）
            if(buffer.size() - write_pos < 16){ flush_buffer(false); }
            char* ptr = buffer.data() + write_pos;
            char* end = buffer.data() + buffer.size();
            auto r = std::to_chars(ptr, end, dist[idx_rc(i,j,V)]);
            if(r.ec == std::errc()){
                write_pos = static_cast<size_t>(r.ptr - buffer.data());
            }else{
                // 极端情况下不足，再次 flush 后重试
                flush_buffer(false);
                ptr = buffer.data() + write_pos;
                end = buffer.data() + buffer.size();
                r = std::to_chars(ptr, end, dist[idx_rc(i,j,V)]);
                write_pos = static_cast<size_t>(r.ptr - buffer.data());
            }
        }
        if(write_pos >= buffer.size()) flush_buffer(false);
        buffer[write_pos++] = '\n';

        // 行结束后，若缓冲已接近目标大小，则写出
        if(write_pos >= kTargetChunkBytes) flush_buffer(false);
    }
    flush_buffer(true);
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
#if FAST_OUTPUT
    // I/O 加速：关闭 iostream 与 stdio 同步，并扩大 stdout 缓冲
    std::ios_base::sync_with_stdio(false);
    std::cout.tie(nullptr);
    // 将 stdout 设为全缓冲，8MB 缓冲区（若为 TTY 仍可能被行缓冲覆盖，但评测一般重定向到文件）
    setvbuf(stdout, nullptr, _IOFBF, 8u << 20);
#endif
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
    int V = 0, E = 0;
    std::vector<int> h_dist;
    std::vector<Edge> edges;
    
    // Timer for data loading to host
    auto start_load = std::chrono::high_resolution_clock::now();
    if(!read_graph(argv[1], V, E, h_dist, edges)){
        std::fprintf(stderr, "Error: failed to read input file.\n");
        return 1;
    }
    auto end_load = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_load - start_load);
        std::cerr << "[TIMER] Data loading to host: " << load_duration.count() << " us" << std::endl;
    }
    
    // Step 1: Graph Partitioning and Boundary Identification
    auto start_partition = std::chrono::high_resolution_clock::now();
    
    // Vertex interval partitioning with optional adaptivity
    int k = NUM_COMPONENTS;  // Number of components
    if(k > V) k = V;
#if ADAPTIVE_K
    // Initial guess by target component size
    const int target_comp_size = 1024;
    k = std::min(NUM_COMPONENTS, std::max(1, V / std::max(1, target_comp_size)));
    if(k > V) k = V;
    if(V <= 8) k = 2; else if(V <= 64) k = std::min(k, std::max(1, V / 2));
#endif
    std::vector<int> vertex_to_component(V);
rebalance_components:
    std::vector<ComponentInfo> components(k);
    for(int i = 0; i < V; ++i){
        int comp_id = std::min(k - 1, i / ((V + k - 1) / k));
        vertex_to_component[i] = comp_id;
        components[comp_id].all_vertices.push_back(i);
    }
    std::vector<bool> is_boundary(V, false);
    for(const auto& edge : edges){
        int u = edge.src;
        int v = edge.dst;
        if(vertex_to_component[u] != vertex_to_component[v]){
            is_boundary[u] = true;
            is_boundary[v] = true;
        }
    }
    std::vector<int> h_all_boundary_vertices;
    for(int i = 0; i < V; ++i){ if(is_boundary[i]) h_all_boundary_vertices.push_back(i); }
#if ADAPTIVE_K
    // Adaptive reduction of k if boundary ratio too high
    const double max_boundary_ratio = 0.15; // 15%
    double boundary_ratio = (V > 0) ? (static_cast<double>(h_all_boundary_vertices.size()) / static_cast<double>(V)) : 0.0;
    if(boundary_ratio > max_boundary_ratio && k > 1){
        int new_k = std::max(1, k / 2);
        if(new_k != k){ k = new_k; goto rebalance_components; }
    }
#endif
    
    // Create mapping from boundary vertex ID to SSSP result row index
    std::vector<int> h_boundary_vertex_to_sssp_row(V, -1);  // -1 means not a boundary vertex
    for(int i = 0; i < static_cast<int>(h_all_boundary_vertices.size()); ++i){
        int boundary_vertex_id = h_all_boundary_vertices[i];
        h_boundary_vertex_to_sssp_row[boundary_vertex_id] = i;
    }
    
    // Build interior and boundary vertex lists for each component
    for(int comp_id = 0; comp_id < k; ++comp_id){
        for(int vertex : components[comp_id].all_vertices){
            if(is_boundary[vertex]){
                components[comp_id].boundary_vertices.push_back(vertex);
            } else {
                components[comp_id].interior_vertices.push_back(vertex);
            }
        }
    }
    
    auto end_partition = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto partition_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_partition - start_partition);
        std::cerr << "[TIMER] Graph partitioning and boundary identification: " << partition_duration.count() << " us" << std::endl;
        std::cerr << "[INFO] Total boundary vertices: " << h_all_boundary_vertices.size() << std::endl;
        std::cerr << "[INFO] Number of components: " << k << std::endl;
        if(V > 0){
            double ratio = static_cast<double>(h_all_boundary_vertices.size()) / static_cast<double>(V);
            std::cerr << "[INFO] Boundary ratio: " << std::fixed << std::setprecision(4) << ratio << std::endl;
        }
    }
    
    // Build flattened component data structures for device
    std::vector<int> h_component_members;   // Flattened array of all component members
    std::vector<int> h_component_offsets(k + 1, 0);  // Offset array for each component
    
    // Build flattened interior and boundary vertex data structures
    std::vector<int> h_component_interior_members;   // Flattened array of interior vertices
    std::vector<int> h_component_interior_offsets(k + 1, 0);  // Offset array for interior vertices
    std::vector<int> h_component_boundary_members;   // Flattened array of boundary vertices
    std::vector<int> h_component_boundary_offsets(k + 1, 0);  // Offset array for boundary vertices
    
    for(int comp_id = 0; comp_id < k; ++comp_id) {
        // All vertices
        h_component_offsets[comp_id] = static_cast<int>(h_component_members.size());
        for(int vertex : components[comp_id].all_vertices) {
            h_component_members.push_back(vertex);
        }
        
        // Interior vertices
        h_component_interior_offsets[comp_id] = static_cast<int>(h_component_interior_members.size());
        for(int vertex : components[comp_id].interior_vertices) {
            h_component_interior_members.push_back(vertex);
        }
        
        // Boundary vertices
        h_component_boundary_offsets[comp_id] = static_cast<int>(h_component_boundary_members.size());
        for(int vertex : components[comp_id].boundary_vertices) {
            h_component_boundary_members.push_back(vertex);
        }
    }
    
    // End markers
    h_component_offsets[k] = static_cast<int>(h_component_members.size());
    h_component_interior_offsets[k] = static_cast<int>(h_component_interior_members.size());
    h_component_boundary_offsets[k] = static_cast<int>(h_component_boundary_members.size());
    
    // Build CSR representation for SSSP
    auto start_csr = std::chrono::high_resolution_clock::now();
    CSRGraph h_csr_graph;
    build_csr_graph(edges, V, h_csr_graph);
    auto end_csr = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto csr_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_csr - start_csr);
        std::cerr << "[TIMER] CSR graph construction: " << csr_duration.count() << " us" << std::endl;
    }
    const int B = BLOCK_SIZE;
    const int nTiles = (V + B - 1) / B;
    
    // Device memory pointers
    int* d_dist = nullptr;
    int* d_csr_row_ptr = nullptr;
    int* d_csr_col_idx = nullptr;
    int* d_csr_weights = nullptr;
    int* d_all_boundary_vertices = nullptr;
    int* d_boundary_sssp_results = nullptr;
    int* d_boundary_vertex_to_sssp_row = nullptr;  // Mapping array
    
    // Component information device memory
    int* d_component_members = nullptr;  // Flattened array of all component members
    int* d_component_offsets = nullptr;  // Offset array for each component
    
    // Component interior/boundary vertex device memory
    int* d_component_interior_members = nullptr;  // Flattened array of interior vertices
    int* d_component_interior_offsets = nullptr;  // Offset array for interior vertices
    int* d_component_boundary_members = nullptr;  // Flattened array of boundary vertices
    int* d_component_boundary_offsets = nullptr;  // Offset array for boundary vertices
    
    // SSSP working memory pointers
#if BATCH_SSSP
    uint8_t* d_frontier_batch = nullptr;
    uint8_t* d_next_frontier_batch = nullptr;
    int* d_next_frontier_count = nullptr;
    int* d_sources_batch = nullptr;
#else
    bool* d_frontier = nullptr;
    bool* d_next_frontier = nullptr;
    bool* d_is_frontier_active = nullptr;
#endif
    
    // Memory sizes calculation
    size_t dist_bytes = static_cast<size_t>(V) * static_cast<size_t>(V) * sizeof(int);
    size_t csr_row_ptr_bytes = (V + 1) * sizeof(int);
    size_t csr_col_idx_bytes = h_csr_graph.num_edges * sizeof(int);
    size_t csr_weights_bytes = h_csr_graph.num_edges * sizeof(int);
    size_t boundary_vertices_bytes = h_all_boundary_vertices.size() * sizeof(int);
    size_t sssp_results_bytes = h_all_boundary_vertices.size() * static_cast<size_t>(V) * sizeof(int);
    size_t boundary_mapping_bytes = V * sizeof(int);  // Mapping array
    
    // Component information memory sizes
    size_t component_members_bytes = h_component_members.size() * sizeof(int);
    size_t component_offsets_bytes = (k + 1) * sizeof(int);
    
    // Component interior/boundary vertex memory sizes
    size_t component_interior_members_bytes = h_component_interior_members.size() * sizeof(int);
    size_t component_interior_offsets_bytes = (k + 1) * sizeof(int);
    size_t component_boundary_members_bytes = h_component_boundary_members.size() * sizeof(int);
    size_t component_boundary_offsets_bytes = (k + 1) * sizeof(int);
    
    // SSSP working memory sizes
#if BATCH_SSSP
    const int batch_cap = BATCH_SIZE;
    size_t frontier_batch_bytes = static_cast<size_t>(batch_cap) * static_cast<size_t>(V) * sizeof(uint8_t);
#else
    size_t frontier_bytes = V * sizeof(bool);
    size_t next_frontier_bytes = V * sizeof(bool);
    size_t is_frontier_active_bytes = sizeof(bool);
#endif
    
    // Timer for GPU memory allocation
    auto start_alloc = std::chrono::high_resolution_clock::now();
    
    // Allocate device memory
    hipCheck(hipMalloc(&d_dist, dist_bytes), "hipMalloc d_dist");
    hipCheck(hipMalloc(&d_csr_row_ptr, csr_row_ptr_bytes), "hipMalloc d_csr_row_ptr");
    hipCheck(hipMalloc(&d_csr_col_idx, csr_col_idx_bytes), "hipMalloc d_csr_col_idx");
    hipCheck(hipMalloc(&d_csr_weights, csr_weights_bytes), "hipMalloc d_csr_weights");
    hipCheck(hipMalloc(&d_all_boundary_vertices, boundary_vertices_bytes), "hipMalloc d_all_boundary_vertices");
    hipCheck(hipMalloc(&d_boundary_sssp_results, sssp_results_bytes), "hipMalloc d_boundary_sssp_results");
    hipCheck(hipMalloc(&d_boundary_vertex_to_sssp_row, boundary_mapping_bytes), "hipMalloc d_boundary_vertex_to_sssp_row");
    
    // Allocate component information memory
    hipCheck(hipMalloc(&d_component_members, component_members_bytes), "hipMalloc d_component_members");
    hipCheck(hipMalloc(&d_component_offsets, component_offsets_bytes), "hipMalloc d_component_offsets");
    
    // Allocate component interior/boundary vertex memory
    hipCheck(hipMalloc(&d_component_interior_members, component_interior_members_bytes), "hipMalloc d_component_interior_members");
    hipCheck(hipMalloc(&d_component_interior_offsets, component_interior_offsets_bytes), "hipMalloc d_component_interior_offsets");
    hipCheck(hipMalloc(&d_component_boundary_members, component_boundary_members_bytes), "hipMalloc d_component_boundary_members");
    hipCheck(hipMalloc(&d_component_boundary_offsets, component_boundary_offsets_bytes), "hipMalloc d_component_boundary_offsets");
    
    // Allocate SSSP working memory
#if BATCH_SSSP
    hipCheck(hipMalloc(&d_frontier_batch, frontier_batch_bytes), "hipMalloc d_frontier_batch");
    hipCheck(hipMalloc(&d_next_frontier_batch, frontier_batch_bytes), "hipMalloc d_next_frontier_batch");
    hipCheck(hipMalloc(&d_next_frontier_count, sizeof(int)), "hipMalloc d_next_frontier_count");
    hipCheck(hipMalloc(&d_sources_batch, static_cast<size_t>(batch_cap) * sizeof(int)), "hipMalloc d_sources_batch");
#else
    hipCheck(hipMalloc(&d_frontier, frontier_bytes), "hipMalloc d_frontier");
    hipCheck(hipMalloc(&d_next_frontier, next_frontier_bytes), "hipMalloc d_next_frontier");
    hipCheck(hipMalloc(&d_is_frontier_active, is_frontier_active_bytes), "hipMalloc d_is_frontier_active");
#endif
    
    auto end_alloc = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_alloc - start_alloc);
        std::cerr << "[TIMER] GPU memory allocation: " << alloc_duration.count() << " us" << std::endl;
    }
    
    // Timer for data transfer to device
    auto start_h2d = std::chrono::high_resolution_clock::now();
    
    // Copy distance matrix
    hipCheck(hipMemcpy(d_dist, h_dist.data(), dist_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_dist");
    
    // Copy CSR graph data
    hipCheck(hipMemcpy(d_csr_row_ptr, h_csr_graph.row_ptr.data(), csr_row_ptr_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_csr_row_ptr");
    hipCheck(hipMemcpy(d_csr_col_idx, h_csr_graph.col_idx.data(), csr_col_idx_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_csr_col_idx");
    hipCheck(hipMemcpy(d_csr_weights, h_csr_graph.weights.data(), csr_weights_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_csr_weights");
    
    // Copy boundary vertices
    if(!h_all_boundary_vertices.empty()){
        hipCheck(hipMemcpy(d_all_boundary_vertices, h_all_boundary_vertices.data(), boundary_vertices_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_all_boundary_vertices");
    }
    
    // Copy boundary vertex to SSSP row mapping
    hipCheck(hipMemcpy(d_boundary_vertex_to_sssp_row, h_boundary_vertex_to_sssp_row.data(), boundary_mapping_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_boundary_vertex_to_sssp_row");
    
    // Copy component information
    if(!h_component_members.empty()) {
        hipCheck(hipMemcpy(d_component_members, h_component_members.data(), component_members_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_component_members");
    }
    hipCheck(hipMemcpy(d_component_offsets, h_component_offsets.data(), component_offsets_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_component_offsets");
    
    // Copy component interior/boundary vertex information
    if(!h_component_interior_members.empty()) {
        hipCheck(hipMemcpy(d_component_interior_members, h_component_interior_members.data(), component_interior_members_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_component_interior_members");
    }
    hipCheck(hipMemcpy(d_component_interior_offsets, h_component_interior_offsets.data(), component_interior_offsets_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_component_interior_offsets");
    
    if(!h_component_boundary_members.empty()) {
        hipCheck(hipMemcpy(d_component_boundary_members, h_component_boundary_members.data(), component_boundary_members_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_component_boundary_members");
    }
    hipCheck(hipMemcpy(d_component_boundary_offsets, h_component_boundary_offsets.data(), component_boundary_offsets_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_component_boundary_offsets");
    
    // Initialize SSSP results matrix to INF (properly)
    size_t sssp_matrix_size = h_all_boundary_vertices.size() * V;
    if(sssp_matrix_size > 0){
        std::vector<int> h_sssp_init(sssp_matrix_size, INF);
        hipCheck(hipMemcpy(d_boundary_sssp_results, h_sssp_init.data(), sssp_results_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_boundary_sssp_results init");
    }
    
    auto end_h2d = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto h2d_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_h2d - start_h2d);
        std::cerr << "[TIMER] Data transfer to device: " << h2d_duration.count() << " us" << std::endl;
    }
    // --- GPU 计算阶段 ---
    // Timer for total GPU computation
    auto start_gpu = std::chrono::high_resolution_clock::now();
    
    // === Step 2: Parallel SSSP from Boundary Vertices ===
    auto start_sssp = std::chrono::high_resolution_clock::now();
    
    if(enable_timing){
        std::cerr << "[INFO] Step 2: Starting SSSP computation for " << h_all_boundary_vertices.size() << " boundary vertices" << std::endl;
    }
    
    // Only proceed if there are boundary vertices to process
    if(!h_all_boundary_vertices.empty()) {
        int num_total_boundary_vertices = static_cast<int>(h_all_boundary_vertices.size());
#if BATCH_SSSP
        const int batch_cap = BATCH_SIZE;
        dim3 init_block(32, 8, 1); // 256 threads
        dim3 relax_block(256, 1, 1);
        for(int batch_start = 0; batch_start < num_total_boundary_vertices; batch_start += batch_cap){
            int batch_size = std::min(batch_cap, num_total_boundary_vertices - batch_start);
            // Prepare sources batch on host
            std::vector<int> h_sources_batch(batch_size);
            for(int s = 0; s < batch_size; ++s){ h_sources_batch[s] = h_all_boundary_vertices[batch_start + s]; }
            hipCheck(hipMemcpy(d_sources_batch, h_sources_batch.data(), static_cast<size_t>(batch_size) * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy H2D d_sources_batch");
            // Initialize distances and frontier for the whole batch
            dim3 init_grid((V + init_block.x - 1) / init_block.x, (batch_size + init_block.y - 1) / init_block.y, 1);
            hipLaunchKernelGGL(initialize_batch_sssp_kernel, init_grid, init_block, 0, 0,
                d_boundary_sssp_results, d_frontier_batch, d_sources_batch,
                batch_start, batch_size, V);
            hipError_t err_initb = hipGetLastError();
            if(err_initb != hipSuccess){
                std::fprintf(stderr, "Kernel initialize_batch_sssp_kernel launch failed: %s\n", hipGetErrorString(err_initb));
                return 1;
            }
            int iter = 0; const int max_iterations = V;
            while(iter < max_iterations){
                // Clear next frontier and counter
                hipCheck(hipMemset(d_next_frontier_batch, 0, static_cast<size_t>(batch_size) * static_cast<size_t>(V) * sizeof(uint8_t)), "hipMemset d_next_frontier_batch");
                hipCheck(hipMemset(d_next_frontier_count, 0, sizeof(int)), "hipMemset d_next_frontier_count");
                // Launch relaxation across vertices (x) and sources (y)
                dim3 relax_grid((V + relax_block.x - 1) / relax_block.x, batch_size, 1);
                hipLaunchKernelGGL(sssp_batch_kernel, relax_grid, relax_block, 0, 0,
                    d_csr_row_ptr, d_csr_col_idx, d_csr_weights,
                    d_boundary_sssp_results, d_frontier_batch, d_next_frontier_batch,
                    d_next_frontier_count, batch_start, batch_size, V);
                hipError_t err_relax = hipGetLastError();
                if(err_relax != hipSuccess){
                    std::fprintf(stderr, "Kernel sssp_batch_kernel launch failed: %s\n", hipGetErrorString(err_relax));
                    return 1;
                }
                int h_active_count = 0;
                hipCheck(hipMemcpy(&h_active_count, d_next_frontier_count, sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy D2H d_next_frontier_count");
                if(h_active_count == 0) break;
                // Swap batches
                uint8_t* tmp = d_frontier_batch; d_frontier_batch = d_next_frontier_batch; d_next_frontier_batch = tmp;
                iter++;
            }
            if(iter >= max_iterations && enable_timing){
                std::cerr << "[WARNING] Batched SSSP reached max iterations at batch start index " << batch_start << std::endl;
            }
        }
#else
        dim3 sssp_grid((V + 255) / 256, 1, 1);
        dim3 sssp_block(256, 1, 1);
        bool h_is_frontier_active;
        for(int i = 0; i < num_total_boundary_vertices; ++i) {
            int source_vertex = h_all_boundary_vertices[i];
            int* d_current_sssp_dist = d_boundary_sssp_results + static_cast<size_t>(i) * V;
            hipLaunchKernelGGL(initialize_sssp_kernel, sssp_grid, sssp_block, 0, 0,
                d_current_sssp_dist, d_frontier, source_vertex, V);
            hipError_t err_init = hipGetLastError();
            if(err_init != hipSuccess) { std::fprintf(stderr, "Kernel initialize_sssp_kernel launch failed: %s\n", hipGetErrorString(err_init)); return 1; }
            int iter = 0; const int max_iterations = V;
            while(iter < max_iterations) {
                hipCheck(hipMemset(d_next_frontier, 0, V * sizeof(bool)), "hipMemset d_next_frontier");
                hipCheck(hipMemset(d_is_frontier_active, 0, sizeof(bool)), "hipMemset d_is_frontier_active");
                hipLaunchKernelGGL(sssp_kernel, sssp_grid, sssp_block, 0, 0,
                    d_csr_row_ptr, d_csr_col_idx, d_csr_weights,
                    d_current_sssp_dist, d_frontier, d_next_frontier,
                    d_is_frontier_active, V);
                hipError_t err_sssp = hipGetLastError();
                if(err_sssp != hipSuccess) { std::fprintf(stderr, "Kernel sssp_kernel launch failed: %s\n", hipGetErrorString(err_sssp)); return 1; }
                hipCheck(hipMemcpy(&h_is_frontier_active, d_is_frontier_active, sizeof(bool), hipMemcpyDeviceToHost), "hipMemcpy frontier active check");
                if(!h_is_frontier_active) break;
                bool* temp = d_frontier; d_frontier = d_next_frontier; d_next_frontier = temp; iter++;
            }
            if(iter >= max_iterations && enable_timing) { std::cerr << "[WARNING] SSSP for boundary vertex " << source_vertex << " reached max iterations" << std::endl; }
        }
#endif
        // Ensure all SSSP computations are complete
        hipSyncCheck("sync after Step 2");
    }
    
    auto end_sssp = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto sssp_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_sssp - start_sssp);
        std::cerr << "[TIMER] Step 2 - SSSP computation: " << sssp_duration.count() << " us" << std::endl;
    }
    
    // === Step 3: Local APSP using Blocked FW ===
    auto start_local_apsp = std::chrono::high_resolution_clock::now();
    
    if(enable_timing){
        std::cerr << "[INFO] Step 3: Starting Local APSP computation for " << k << " components" << std::endl;
    }
    
    // Preallocate a reusable temporary matrix for the largest component
    int max_comp_size = 0;
    for(int comp_id = 0; comp_id < k; ++comp_id){
        int comp_start_i = h_component_offsets[comp_id];
        int comp_end_i = h_component_offsets[comp_id + 1];
        int n_comp_i = comp_end_i - comp_start_i;
        if(n_comp_i > max_comp_size) max_comp_size = n_comp_i;
    }
    int* d_comp_dist = nullptr;
    size_t comp_dist_max_bytes = static_cast<size_t>(max_comp_size) * static_cast<size_t>(max_comp_size) * sizeof(int);
    if(max_comp_size > 0){ hipCheck(hipMalloc(&d_comp_dist, comp_dist_max_bytes), "hipMalloc d_comp_dist (prealloc)"); }

    // Execute Local APSP for each component
    for(int comp_id = 0; comp_id < k; ++comp_id) {
        int comp_start = h_component_offsets[comp_id];
        int comp_end = h_component_offsets[comp_id + 1];
        int n_comp = comp_end - comp_start;
        
        if(n_comp <= 0) continue;  // Skip empty components
        
        // Get component members pointer on device
        int* d_comp_members_ptr = d_component_members + comp_start;
        
        // 1. Gather: collect component data from global matrix to local matrix
        dim3 gather_block(16, 16, 1);  // 16x16 = 256 threads per block
        dim3 gather_grid((n_comp + 15) / 16, (n_comp + 15) / 16, 1);
        
        hipLaunchKernelGGL(gather_kernel, gather_grid, gather_block, 0, 0,
            d_dist, d_comp_dist, d_comp_members_ptr, n_comp, V);
        hipError_t err_gather = hipGetLastError();
        if(err_gather != hipSuccess) {
            std::fprintf(stderr, "Kernel gather_kernel launch failed: %s\n", hipGetErrorString(err_gather));
            return 1;
        }
        
        // 2. Execute Blocked FW on component's small matrix
        const int comp_nTiles = (n_comp + B - 1) / B;
        dim3 fw_block(B, B, 1);
        if(B*B > 1024){ fw_block.x = 32; fw_block.y = 32; }
        
        for(int fw_k = 0; fw_k < comp_nTiles; ++fw_k) {
            // Phase 1: Update pivot block
            {
                dim3 fw_grid(1, 1, 1);
                size_t shmem = (size_t)B * (size_t)B * sizeof(int);
                hipLaunchKernelGGL(fw_phase1, fw_grid, fw_block, shmem, 0, 
                    d_comp_dist, n_comp, fw_k, B);
                hipError_t err1 = hipGetLastError();
                if(err1 != hipSuccess){ 
                    std::fprintf(stderr, "Kernel fw_phase1 launch failed: %s\n", hipGetErrorString(err1)); 
                    return 1; 
                }
            }
            
            // Phase 2: Update pivot row and column blocks
            {
                dim3 fw_grid(comp_nTiles, 2, 1);
                size_t shmem = 2ull * B * B * sizeof(int);
                hipLaunchKernelGGL(fw_phase2, fw_grid, fw_block, shmem, 0, 
                    d_comp_dist, n_comp, fw_k, B);
                hipError_t err2 = hipGetLastError();
                if(err2 != hipSuccess){ 
                    std::fprintf(stderr, "Kernel fw_phase2 launch failed: %s\n", hipGetErrorString(err2)); 
                    return 1; 
                }
            }
            
            // Phase 3: Update remaining blocks
            {
                dim3 fw_grid(comp_nTiles, comp_nTiles, 1);
                size_t shmem = 2ull * B * B * sizeof(int);
                hipLaunchKernelGGL(fw_phase3, fw_grid, fw_block, shmem, 0, 
                    d_comp_dist, n_comp, fw_k, B);
                hipError_t err3 = hipGetLastError();
                if(err3 != hipSuccess){ 
                    std::fprintf(stderr, "Kernel fw_phase3 launch failed: %s\n", hipGetErrorString(err3)); 
                    return 1; 
                }
            }
        }
        
        // 3. Scatter: write local APSP results back to global matrix
        hipLaunchKernelGGL(scatter_kernel, gather_grid, gather_block, 0, 0,
            d_dist, d_comp_dist, d_comp_members_ptr, n_comp, V);
        hipError_t err_scatter = hipGetLastError();
        if(err_scatter != hipSuccess) {
            std::fprintf(stderr, "Kernel scatter_kernel launch failed: %s\n", hipGetErrorString(err_scatter));
            return 1;
        }
    }
    
    // Ensure all Local APSP computations are complete
    hipSyncCheck("sync after Step 3");
    // Free preallocated component matrix
    hipFreeCheck(d_comp_dist, "d_comp_dist(prealloc)");
    
    auto end_local_apsp = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto local_apsp_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_local_apsp - start_local_apsp);
        std::cerr << "[TIMER] Step 3 - Local APSP computation: " << local_apsp_duration.count() << " us" << std::endl;
    }
    
    // === Step 4: Final MIN-PLUS Computation ===
    auto start_min_plus = std::chrono::high_resolution_clock::now();
    
    if(enable_timing){
        std::cerr << "[INFO] Step 4: Starting MIN-PLUS computation for " << k << " components" << std::endl;
    }
    
    // Allocate temporary device memory for component membership mask
    bool* d_vertex_in_component = nullptr;
    size_t vertex_mask_bytes = V * sizeof(bool);
    hipCheck(hipMalloc(&d_vertex_in_component, vertex_mask_bytes), "hipMalloc d_vertex_in_component");
    
    // Execute MIN-PLUS computation for each component
    for(int comp_id = 0; comp_id < k; ++comp_id) {
        int interior_start = h_component_interior_offsets[comp_id];
        int interior_end = h_component_interior_offsets[comp_id + 1];
        int n_interior = interior_end - interior_start;
        
        int boundary_start = h_component_boundary_offsets[comp_id];
        int boundary_end = h_component_boundary_offsets[comp_id + 1];
        int n_boundary = boundary_end - boundary_start;
        
        int comp_members_start = h_component_offsets[comp_id];
        int comp_members_end = h_component_offsets[comp_id + 1];
        int n_all_vertices = comp_members_end - comp_members_start;
        
        if(n_all_vertices <= 0 || n_boundary <= 0) continue;  // Skip if no vertices or no boundary vertices
        
        // 1. Set up component membership mask
        // Clear the mask first
        hipCheck(hipMemset(d_vertex_in_component, 0, vertex_mask_bytes), "hipMemset d_vertex_in_component");
        
        // Set mask for vertices in current component
        int n_comp_members = comp_members_end - comp_members_start;
        
        if(n_comp_members > 0) {
            // Create host mask for this component using char array (bool has no data() method)
            std::vector<char> h_vertex_mask(V, 0);
            for(int i = comp_members_start; i < comp_members_end; ++i) {
                int vertex_id = h_component_members[i];
                h_vertex_mask[vertex_id] = 1;
            }
            
            // Copy to device
            hipCheck(hipMemcpy(d_vertex_in_component, h_vertex_mask.data(), vertex_mask_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D d_vertex_in_component");
        }
        
        // 2. Get device pointers for this component's vertices
        int* d_comp_all_vertices_ptr = d_component_members + comp_members_start;
        int* d_comp_boundary_ptr = d_component_boundary_members + boundary_start;
        
        // 3. Launch MIN-PLUS finalize kernel for ALL vertices in component (interior + boundary)
        // Grid layout: (V, n_all_vertices) - each thread handles one (u, v) pair
        dim3 minplus_block(16, 16, 1);  // 16x16 = 256 threads per block
        dim3 minplus_grid((V + 15) / 16, (n_all_vertices + 15) / 16, 1);
        
        hipLaunchKernelGGL(min_plus_finalize_kernel, minplus_grid, minplus_block, 0, 0,
            d_dist, d_boundary_sssp_results,
            d_comp_all_vertices_ptr, d_comp_boundary_ptr,
            d_boundary_vertex_to_sssp_row, d_vertex_in_component,
            n_all_vertices, n_boundary, V);
        
        hipError_t err_minplus = hipGetLastError();
        if(err_minplus != hipSuccess) {
            std::fprintf(stderr, "Kernel min_plus_finalize_kernel launch failed: %s\n", hipGetErrorString(err_minplus));
            return 1;
        }
    }
    
    // Ensure all MIN-PLUS computations are complete
    hipSyncCheck("sync after Step 4");
    
    // Free temporary component mask memory
    hipFreeCheck(d_vertex_in_component, "d_vertex_in_component");
    
    auto end_min_plus = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto min_plus_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_min_plus - start_min_plus);
        std::cerr << "[TIMER] Step 4 - MIN-PLUS computation: " << min_plus_duration.count() << " us" << std::endl;
    }
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
        std::cerr << "[TIMER] GPU computation: " << gpu_duration.count() << " us" << std::endl;
    }
    
    // Timer for data transfer from device
    auto start_d2h = std::chrono::high_resolution_clock::now();
    hipCheck(hipMemcpy(h_dist.data(), d_dist, dist_bytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");
    auto end_d2h = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto d2h_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_d2h - start_d2h);
        std::cerr << "[TIMER] Data transfer from device: " << d2h_duration.count() << " us" << std::endl;
    }
    
    // Timer for GPU memory cleanup
    auto start_cleanup = std::chrono::high_resolution_clock::now();
    hipFreeCheck(d_dist, "d_dist");
    hipFreeCheck(d_csr_row_ptr, "d_csr_row_ptr");
    hipFreeCheck(d_csr_col_idx, "d_csr_col_idx");
    hipFreeCheck(d_csr_weights, "d_csr_weights");
    hipFreeCheck(d_all_boundary_vertices, "d_all_boundary_vertices");
    hipFreeCheck(d_boundary_sssp_results, "d_boundary_sssp_results");
    hipFreeCheck(d_boundary_vertex_to_sssp_row, "d_boundary_vertex_to_sssp_row");
    
    // Free component information memory
    hipFreeCheck(d_component_members, "d_component_members");
    hipFreeCheck(d_component_offsets, "d_component_offsets");
    
    // Free component interior/boundary vertex memory
    hipFreeCheck(d_component_interior_members, "d_component_interior_members");
    hipFreeCheck(d_component_interior_offsets, "d_component_interior_offsets");
    hipFreeCheck(d_component_boundary_members, "d_component_boundary_members");
    hipFreeCheck(d_component_boundary_offsets, "d_component_boundary_offsets");
    
    // Free SSSP working memory
    // Free SSSP working memory
#if BATCH_SSSP
    hipFreeCheck(d_frontier_batch, "d_frontier_batch");
    hipFreeCheck(d_next_frontier_batch, "d_next_frontier_batch");
    hipFreeCheck(d_next_frontier_count, "d_next_frontier_count");
    hipFreeCheck(d_sources_batch, "d_sources_batch");
#else
    hipFreeCheck(d_frontier, "d_frontier");
    hipFreeCheck(d_next_frontier, "d_next_frontier");
    hipFreeCheck(d_is_frontier_active, "d_is_frontier_active");
#endif
    auto end_cleanup = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto cleanup_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cleanup - start_cleanup);
        std::cerr << "[TIMER] GPU memory cleanup: " << cleanup_duration.count() << " us" << std::endl;
    }
    
    // Timer for result output
    auto start_output = std::chrono::high_resolution_clock::now();
    print_matrix(h_dist, V);
    auto end_output = std::chrono::high_resolution_clock::now();
    if(enable_timing){
        auto output_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_output - start_output);
        std::cerr << "[TIMER] Result output: " << output_duration.count() << " us" << std::endl;
    }
    return 0;
}