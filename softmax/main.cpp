#include "main.h"
#include <chrono>
#include <cstdio>

// 高速输出：分块缓冲 + snprintf
static inline void print_output_fast(const float* data, int N){
    // 目标块大小（至少 1MB）
    constexpr size_t kTargetChunkBytes = 1u << 20;
    std::vector<char> buffer(kTargetChunkBytes);
    size_t write_pos = 0;

    auto flush_buffer = [&](bool final_flush){
        if(write_pos > 0){
            (void)std::fwrite(buffer.data(), 1, write_pos, stdout);
            write_pos = 0;
        }
        if(final_flush){ std::fflush(stdout); }
    };

    for(int i = 0; i < N; ++i){
        // 预留足够空间，若不足则先刷新
        if(buffer.size() - write_pos < 32){ flush_buffer(false); }
        int avail = static_cast<int>(buffer.size() - write_pos);
        int wrote = std::snprintf(buffer.data() + write_pos, static_cast<size_t>(avail), "%.6g", static_cast<double>(data[i]));
        if(wrote < 0){
            // 极端错误情况下跳过（不应发生）
            continue;
        }
        if(wrote >= avail){
            // 空间不够，刷新后重试一次
            flush_buffer(false);
            avail = static_cast<int>(buffer.size() - write_pos);
            wrote = std::snprintf(buffer.data() + write_pos, static_cast<size_t>(avail), "%.6g", static_cast<double>(data[i]));
        }
        write_pos += static_cast<size_t>(wrote);
        // 追加一个空格，保持与原实现一致（每个数字后都有空格）
        if(write_pos >= buffer.size()) flush_buffer(false);
        buffer[write_pos++] = ' ';
        if(write_pos >= kTargetChunkBytes) flush_buffer(false);
    }
    if(write_pos >= buffer.size()) flush_buffer(false);
    buffer[write_pos++] = '\n';
    flush_buffer(true);
}

bool g_enable_timing = false;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <input_file> [--timing]" << std::endl;
        return 1;
    }

    // 解析参数：argv[1] 为输入文件，其余检查 --timing
    std::string filename = argv[1];
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--timing") {
            g_enable_timing = true;
        }
    }

    std::ifstream input_file;
    input_file.open(filename);
    if (!input_file.is_open()) {
        std::cerr << "fileopen error" << filename << std::endl;
        return 1;
    }
    int N;
    input_file >> N;

    std::vector<float> input(N), output(N);

    for(int i = 0; i < N; ++i) input_file >> input[i];

    input_file.close();

    solve(input.data(), output.data(), N);

    auto start_output = std::chrono::high_resolution_clock::now();
#if FAST_OUTPUT
    // I/O 加速：关闭 iostream 同步并设置 stdout 大缓冲
    std::ios_base::sync_with_stdio(false);
    std::cout.tie(nullptr);
    setvbuf(stdout, nullptr, _IOFBF, 8u << 20);
    print_output_fast(output.data(), N);
#else
    for(int i = 0; i < N; ++i) std::cout << output[i] << " ";
    std::cout << std::endl;
#endif
    auto end_output = std::chrono::high_resolution_clock::now();
    if (g_enable_timing) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_output - start_output);
        std::cerr << "[TIMER] Result output: " << duration.count() << " us" << std::endl;
    }
}

