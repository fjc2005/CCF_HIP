#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <hip/hip_runtime.h>
#include <float.h>
#include <fstream>

extern "C" void solve(const float* input, float* output, int N);

// 全局计时开关，由 main.cpp 定义；在 kernel.hip 中读取
extern bool g_enable_timing;

#endif 