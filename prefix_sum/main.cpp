#include "main.h"
#include <chrono>
#include <string>

// 启用快速输出优化
#ifndef FAST_OUTPUT
#define FAST_OUTPUT 1
#endif

// 与 prefix_sum/main.h 对应的全局符号
bool g_enable_timing = false;
double g_print_overhead_ms = 0.0;
void print_timing(const char* /*label*/, double /*ms*/) {}
void flush_timings() {}

// 高性能数组输出函数（基于apsp项目的fast_output技术）
static void fast_output_array(const int* data, int N){
#if defined(FAST_OUTPUT) && FAST_OUTPUT
    const size_t BUF_SIZE = static_cast<size_t>(32) * 1024 * 1024; // 32MB
    std::vector<char> buffer(BUF_SIZE);
    char* const buf_begin = buffer.data();
    char* const buf_end = buffer.data() + buffer.size();
    char* p = buffer.data();

    for(int i = 0; i < N; ++i){
        if(i){
            if(p >= buf_end){
                std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
                p = const_cast<char*>(buf_begin);
            }
            *p++ = ' ';
        }
        // 为数字预留足够空间（最多10位数字）
        if(p + 16 > buf_end){
            std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
            p = const_cast<char*>(buf_begin);
        }
        int val = data[static_cast<size_t>(i)];
        auto conv = std::to_chars(p, buf_end, val);
        // to_chars 对于 int 基本不会失败（只要有足够空间）
        p = conv.ptr;
    }
    if(p >= buf_end){
        std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
        p = const_cast<char*>(buf_begin);
    }
    *p++ = '\n';
    
    if(p > buf_begin){
        std::fwrite(buf_begin, 1, static_cast<size_t>(p - buf_begin), stdout);
    }
    std::fflush(stdout);
#else
    // 回退到标准输出
    for(int i = 0; i < N; ++i){
        if(i) std::cout << ' ';
        std::cout << data[static_cast<size_t>(i)];
    }
    std::cout << '\n';
#endif
}

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

#if defined(FAST_OUTPUT) && FAST_OUTPUT
	// 加速 stdout I/O（基于apsp项目的优化）
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);
	std::cout.tie(nullptr);
	// 使用大缓冲区减少系统调用
	setvbuf(stdout, nullptr, _IOFBF, 32 * 1024 * 1024);
#endif

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

	// 结果输出计时（使用优化的 fast_output_array）
	auto start_output = std::chrono::high_resolution_clock::now();
	fast_output_array(h_out.data(), N);
	auto end_output = std::chrono::high_resolution_clock::now();
	if(g_enable_timing){
		auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end_output - start_output);
		std::cerr << "[TIMER] Result output: " << dur.count() << " us" << std::endl;
	}
	return 0;
}

