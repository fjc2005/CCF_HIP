#include "main.h"
#include <chrono>
#include <string>

// 与 prefix_sum/main.h 对应的全局符号
bool g_enable_timing = false;
double g_print_overhead_ms = 0.0;
void print_timing(const char* /*label*/, double /*ms*/) {}
void flush_timings() {}

static bool read_input_file(const char* path, std::vector<int>& input, int& N){
	std::ifstream fin(path);
	if(!fin.is_open()) return false;
	if(!(fin >> N)) return false;
	if(N < 0) return false;
	input.resize(static_cast<size_t>(N));
	for(int i=0;i<N;++i){
		if(!(fin >> input[static_cast<size_t>(i)])) return false;
	}
	return true;
}

int main(int argc, char* argv[]){
	if(argc < 2){
		std::fprintf(stderr, "Error: missing input file.\n");
		return 1;
	}

	int impl_select = 1; // 1=blelloch, 0=hs
	for(int i=2;i<argc;++i){
		std::string arg(argv[i]);
		if(arg == "--timing"){
			g_enable_timing = true;
		}else if(arg == "--impl" && i+1 < argc){
			std::string v(argv[i+1]);
			if(v == "hs") impl_select = 0; else if(v == "blelloch") impl_select = 1;
			++i;
		}
	}
	set_scan_impl(impl_select);

	std::vector<int> h_in;
	int N = 0;
	if(!read_input_file(argv[1], h_in, N)){
		std::fprintf(stderr, "Error: failed to read input file.\n");
		return 1;
	}
	std::vector<int> h_out(static_cast<size_t>(N));

	// 求解（GPU 内部计时见 kernel.hip 的 solve() 实现）
	solve(h_in.data(), h_out.data(), N);

	// 结果输出计时（与 apsp 一致）
	auto start_output = std::chrono::high_resolution_clock::now();
	for(int i=0;i<N;++i){
		if(i) std::cout << ' ';
		std::cout << h_out[static_cast<size_t>(i)];
	}
	std::cout << '\n';
	auto end_output = std::chrono::high_resolution_clock::now();
	if(g_enable_timing){
		auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end_output - start_output);
		std::cerr << "[TIMER] Result output: " << dur.count() << " us" << std::endl;
	}
	return 0;
}

