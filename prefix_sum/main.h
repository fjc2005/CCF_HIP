#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <fstream>
#include <charconv>
#include <cstdio>
#include <cstdlib>

extern "C" void solve(const int* input, int* output, int N);
extern "C" void set_scan_impl(int impl);

// Timing controls shared between translation units
extern bool g_enable_timing;
void print_timing(const char* label, double ms);
void flush_timings();
extern double g_print_overhead_ms;

#endif 