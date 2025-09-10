#include "main.h"
#include <chrono>
#include <sstream>
#include <vector>

bool g_enable_timing = false;
static std::vector<std::string> g_timing_lines;

static inline double elapsed_ms(const std::chrono::steady_clock::time_point& s,
                                const std::chrono::steady_clock::time_point& e) {
    return std::chrono::duration<double, std::milli>(e - s).count();
}

void print_timing(const char* label, double ms) {
    if (!g_enable_timing) return;
    std::ostringstream oss;
    oss << "[TIMER] " << label << ": " << std::fixed << std::setprecision(3) << ms << " ms";
    g_timing_lines.push_back(oss.str());
}

void flush_timings() {
    if (!g_enable_timing) return;
    for (const auto& line : g_timing_lines) {
        std::cout << line << std::endl;
    }
    std::cout.flush();
}

static void cpu_inclusive_scan(const std::vector<int>& in, std::vector<int>& out) {
    long long acc = 0;
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        acc += in[i];
        out[i] = (int)acc;
    }
}

int main(int argc, char* argv[]) {
    // pre-scan for --time to avoid affecting existing behavior when not enabled
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--time" || std::string(argv[i]) == "--profile") { g_enable_timing = true; break; }
    }

    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <input_file> [--verify] [--repeat K] [--impl hs|blelloch]" << std::endl;
        return 1;
    }

    std::string filename;
    bool do_verify = false;
    int repeat = 1;
    int impl = 1; // 1 = blelloch, 0 = hs

    std::chrono::steady_clock::time_point t_total_start;
    if (g_enable_timing) t_total_start = std::chrono::steady_clock::now();

    filename = argv[1];
    std::chrono::steady_clock::time_point t_parse_start, t_parse_end;
    if (g_enable_timing) t_parse_start = std::chrono::steady_clock::now();
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--time" || arg == "--profile") { /* handled in pre-scan; ignore here */ }
        else if (arg == "--verify") do_verify = true;
        else if (arg == "--repeat" && i + 1 < argc) { repeat = std::max(1, atoi(argv[++i])); }
        else if (arg == "--impl" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "hs") impl = 0; else impl = 1;
        }
    }
    if (g_enable_timing) { t_parse_end = std::chrono::steady_clock::now(); print_timing("arg parse", elapsed_ms(t_parse_start, t_parse_end)); }

    std::ifstream input_file;
    input_file.open(filename);
    if (!input_file.is_open()) {
        std::cerr << "fileopen error " << filename << std::endl;
        return 1;
    }

    std::chrono::steady_clock::time_point t_read_start, t_read_end;
    if (g_enable_timing) t_read_start = std::chrono::steady_clock::now();

    int N;
    input_file >> N;

    std::vector<int> input(N), output(N);
    for(int i = 0; i < N; ++i)
        input_file >> input[i];
    input_file.close();
    if (g_enable_timing) { t_read_end = std::chrono::steady_clock::now(); print_timing("read input", elapsed_ms(t_read_start, t_read_end)); }

    set_scan_impl(impl);

    std::chrono::steady_clock::time_point t_compute_start, t_compute_end;
    if (g_enable_timing) t_compute_start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) {
        solve(input.data(), output.data(), N);
    }
    if (g_enable_timing) { t_compute_end = std::chrono::steady_clock::now(); print_timing("compute (solve)", elapsed_ms(t_compute_start, t_compute_end)); }

    if (do_verify) {
        std::chrono::steady_clock::time_point t_verify_start, t_verify_end;
        if (g_enable_timing) t_verify_start = std::chrono::steady_clock::now();
        std::vector<int> ref;
        cpu_inclusive_scan(input, ref);
        bool ok = true;
        for (int i = 0; i < N; ++i) if (ref[i] != output[i]) { ok = false; break; }
        std::cerr << (ok ? "verify: OK" : "verify: MISMATCH") << std::endl;
        if (g_enable_timing) { t_verify_end = std::chrono::steady_clock::now(); print_timing("verify", elapsed_ms(t_verify_start, t_verify_end)); }
    }

    // 在计时模式下不打印前缀和结果，只输出计时与校验信息
    if (!g_enable_timing) {
        for (int i = 0; i < N; ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;
    }

    if (g_enable_timing) {
        auto t_total_end = std::chrono::steady_clock::now();
        print_timing("total", elapsed_ms(t_total_start, t_total_end));
        flush_timings();
    }
}
