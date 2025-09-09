#include "main.h"

static void cpu_inclusive_scan(const std::vector<int>& in, std::vector<int>& out) {
    long long acc = 0;
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        acc += in[i];
        out[i] = (int)acc;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <input_file> [--verify] [--repeat K] [--impl hs|blelloch]" << std::endl;
        return 1;
    }

    std::string filename;
    bool do_verify = false;
    int repeat = 1;
    int impl = 1; // 1 = blelloch, 0 = hs

    filename = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verify") do_verify = true;
        else if (arg == "--repeat" && i + 1 < argc) { repeat = std::max(1, atoi(argv[++i])); }
        else if (arg == "--impl" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "hs") impl = 0; else impl = 1;
        }
    }

    std::ifstream input_file;
    input_file.open(filename);
    if (!input_file.is_open()) {
        std::cerr << "fileopen error " << filename << std::endl;
        return 1;
    }

    int N;
    input_file >> N;

    std::vector<int> input(N), output(N);
    for(int i = 0; i < N; ++i)
        input_file >> input[i];
    input_file.close();

    set_scan_impl(impl);

    for (int r = 0; r < repeat; ++r) {
        solve(input.data(), output.data(), N);
    }

    if (do_verify) {
        std::vector<int> ref;
        cpu_inclusive_scan(input, ref);
        bool ok = true;
        for (int i = 0; i < N; ++i) if (ref[i] != output[i]) { ok = false; break; }
        std::cerr << (ok ? "verify: OK" : "verify: MISMATCH") << std::endl;
    }

    for(int i = 0; i < N; ++i)
        std::cout << output[i] << " ";
    std::cout << std::endl;
}
