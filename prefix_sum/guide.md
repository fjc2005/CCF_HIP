好的，作为一名精通从NVIDIA CUDA到AMD ROCm/HIP平台代码迁移与优化的GPU并行计算专家，我将为您把这份宝贵的CUDA优化经验，转换为一份专业、严谨、可直接用于项目的\*\*《基于AMD MI100的GPU并行前缀和(Scan)计算优化方案指导手册》\*\*。

本手册将严格遵循您的指令，保持原始优化脉络，同时将所有关键概念、术语和代码示例无缝映射到AMD ROCm/HIP生态及MI100的Wave64架构特性上。

-----

# 基于AMD MI100的GPU并行前缀和(Scan)计算优化方案指导手册

## 引言：AMD MI100架构与ROCm/HIP生态简介

在开始我们的优化之旅前，了解目标平台AMD MI100 GPU的架构特性至关重要。MI100基于CDNA架构，其核心并行执行单元是**Wavefront**（简称Wave），由**64个**工作项（线程）组成。这与NVIDIA架构中的Warp（32个线程）是关键区别。理解**Wave64**执行模型是编写高效HIP代码的基础。

MI100拥有极高的内存带宽和性能优异的**LDS (Local Data Share)**，即片上共享内存。为了最大化硬件利用率，推荐在MI100上设置的线程块大小（`blockDim`）通常是**256的倍数**（如256, 512, 1024）。

本手册将引导您完成一个并行前缀和（Scan）算法在ROCm/HIP平台上的完整优化流程，所有示例均已针对MI100的Wave64特性进行适配。在性能对比和最终方案推荐中，我们将使用AMD官方提供的高性能计算库 **`rocPRIM`** 作为黄金标准。

## 1\. 问题定义

首先我们不严谨地定义这个问题，输入一个数组 `input[n]`, 计算新数组 `output[n]`, 使得对于任意元素 `output[i]` 都满足：
`output[i] = input[0] + input[1] + ... input[i]`

一个示例如下：
**输入:** `0 1 2 ... 9`
**输出:** `0 1 3 ... 45`

在CPU上，这是一个简单的顺序累加过程：

```cpp
void PrefixSum(const int32_t* input, size_t n, int32_t* output) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += input[i];
    output[i] = sum;
  }
}
```

问题的挑战在于，每个元素的计算都依赖于前一个元素的结果，这存在明显的依赖性。如何在拥有数千个并行线程的GPU上高效地解决这个问题，正是本手册要探讨的核心。

## 2\. 两段式扫描 (Two-Pass Scan) 算法思想

我们将采用一种分而治之的经典并行算法，其核心思想分为两个主要阶段（Pass），非常适合GPU的大规模并行架构。

1.  **分块处理与局部求和 (Reduce/Scan Pass)**：将巨大的输入数组切分为多个数据块（Part）。每个线程块（Thread Block）负责处理一个或多个数据块。在块内，所有线程协同进行并行的前缀和计算，并计算出该块内所有元素的总和，我们称之为`PartSum`。
2.  **块间求和与全局更新 (Scan/Update Pass)**：对上一步产生的所有`PartSum`组成的新数组，再次执行一个前缀和计算，得到每个块的累加基准值`BaseSum`。最后，将这些`BaseSum`加回到对应数据块的每个元素上，得到最终的全局前缀和结果。

我们将从一个基础实现开始，逐步迭代优化。

### 2.1 Baseline：基础框架实现

首先，我们搭建一个基础的两段式扫描框架。在这个版本中，为了聚焦于整体流程，线程块内部的扫描暂时由单个线程串行完成。

```cpp
#include <hip/hip_runtime.h>

// 第一阶段：分块进行局部扫描，并写回每个块的总和
__global__ void ScanAndWritePartSumKernel(const int32_t* input, int32_t* part,
                                          int32_t* output, size_t n,
                                          size_t part_num) {
  for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
    // 每个块处理 input[part_begin:part_end]
    // 结果存入 output, 块总和存入 part[part_i]
    size_t part_begin = part_i * blockDim.x;
    size_t part_end = min((part_i + 1) * blockDim.x, n);
    if (threadIdx.x == 0) {  // 朴素实现：由单个线程串行处理
      int32_t acc = 0;
      for (size_t i = part_begin; i < part_end; ++i) {
        acc += input[i];
        output[i] = acc;
      }
      part[part_i] = acc;
    }
  }
}

// 对各块的总和数组(part)进行前缀和计算
__global__ void ScanPartSumKernel(int32_t* part, size_t part_num) {
  // 同样由单个Grid中的单个线程串行处理，作为性能瓶颈点
  int32_t acc = 0;
  for (size_t i = 0; i < part_num; ++i) {
    acc += part[i];
    part[i] = acc;
  }
}

// 第二阶段：将计算好的基准值加回到每个块的元素上
__global__ void AddBaseSumKernel(int32_t* part, int32_t* output, size_t n,
                                 size_t part_num) {
  for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
    if (part_i == 0) {
      continue;
    }
    int32_t index = part_i * blockDim.x + threadIdx.x;
    if (index < n) {
      // 加上前一个块的累加和
      output[index] += part[part_i - 1];
    }
  }
}

// 主机端调用逻辑
void TwoPassScan(const int32_t* input, int32_t* buffer, int32_t* output,
                 size_t n) {
  // 在MI100上，推荐blockDim为256的倍数
  size_t part_size = 1024;
  size_t part_num = (n + part_size - 1) / part_size;
  size_t block_num = std::min<size_t>(part_num, 256); // 设定一个合理的Grid大小
  // 使用buffer来存储中间结果 part
  int32_t* part = buffer;

  // Pass 1: 启动Kernel进行局部扫描
  hipLaunchKernelGGL(ScanAndWritePartSumKernel, dim3(block_num), dim3(part_size), 0, 0,
                     input, part, output, n, part_num);

  // 对PartSum数组进行扫描
  hipLaunchKernelGGL(ScanPartSumKernel, dim3(1), dim3(1), 0, 0, part, part_num);
  
  // Pass 2: 将BaseSum加回
  hipLaunchKernelGGL(AddBaseSumKernel, dim3(block_num), dim3(part_size), 0, 0,
                     part, output, n, part_num);
}
```

这个未经任何优化的Baseline版本是我们后续性能比较的起点。

### 2.2 利用LDS (Local Data Share) 进行优化

第一个明显的瓶颈是`ScanAndWritePartSumKernel`中对全局内存（Global Memory）的重复读写。我们可以利用GPU上高速的片上内存——在AMD平台称为**LDS (Local Data Share)**——来显著提升性能。

我们将每个数据块首先加载到LDS中，然后在LDS上进行计算，最后将结果写回全局内存。

```cpp
// 块内扫描函数，操作LDS
__device__ void ScanBlock(int32_t* lds_data) {
  if (threadIdx.x == 0) {  // 仍然是朴素实现
    int32_t acc = 0;
    for (size_t i = 0; i < blockDim.x; ++i) {
      acc += lds_data[i];
      lds_data[i] = acc;
    }
  }
  __syncthreads(); // 确保所有线程在继续前完成LDS操作
}

__global__ void ScanAndWritePartSumKernel(const int32_t* input, int32_t* part,
                                          int32_t* output, size_t n,
                                          size_t part_num) {
  // HIP中使用 extern __shared__ 声明动态LDS
  extern __shared__ int32_t lds[];
  for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
    // 1. 将数据从全局内存加载到LDS
    size_t index = part_i * blockDim.x + threadIdx.x;
    lds[threadIdx.x] = index < n ? input[index] : 0;
    __syncthreads();

    // 2. 在LDS上进行扫描
    ScanBlock(lds);

    // 3. 将结果从LDS写回全局内存
    if (index < n) {
      output[index] = lds[threadIdx.x];
    }
    if (threadIdx.x == blockDim.x - 1) {
      part[part_i] = lds[threadIdx.x];
    }
  }
}
```

通过将主要计算转移到LDS上，我们避免了大量的全局内存访问延迟，性能得到显著提升。

### 2.3 块内并行扫描 `ScanBlock`

现在，我们来解决`ScanBlock`内部的串行计算问题。一个线程块内的扫描可以进一步分解为多个Wavefront的扫描。

1.  **Wavefront内扫描**: 每个Wavefront（64个线程）独立地对其负责的64个元素进行前缀和。
2.  **Wavefront间扫描**: 每个Wavefront将其内部的总和写入LDS的一个`wave_sum`数组。
3.  启动**一个**Wavefront对`wave_sum`数组进行前缀和计算。
4.  **结果合并**: 每个Wavefront的线程将其计算的局部前缀和，加上前一个Wavefront的累加和。

<!-- end list -->

```cpp
// Wavefront内部扫描，暂时仍为朴素实现
__device__ void ScanWave(int32_t* lds_data, int32_t lane) {
  if (lane == 0) {
    int32_t acc = 0;
    for (int32_t i = 0; i < 64; ++i) {
      acc += lds_data[i];
      lds_data[i] = acc;
    }
  }
}

__device__ void ScanBlock(int32_t* lds_data) {
  // 关键区别：Wavefront是64线程
  int32_t wave_id = threadIdx.x >> 6; // 除以64
  int32_t lane = threadIdx.x & 63;  // 对64取模

  // 假设 blockDim.x = 1024, 则有 1024/64 = 16 个Wavefront
  __shared__ int32_t wave_sum[16];

  // 1. 每个Wavefront内部做扫描
  ScanWave(lds_data, lane);
  __syncthreads();

  // 2. 每个Wavefront的最后一个线程，将其总和写入wave_sum
  if (lane == 63) {
    wave_sum[wave_id] = *lds_data;
  }
  __syncthreads();

  // 3. 使用第一个Wavefront扫描wave_sum数组
  if (wave_id == 0) {
    // 注意这里传递的是wave_sum数组中对应当前lane的元素地址
    ScanWave(wave_sum + lane, lane);
  }
  __syncthreads();

  // 4. 每个Wavefront加上基准值
  if (wave_id > 0) {
    *lds_data += wave_sum[wave_id - 1];
  }
  __syncthreads();
}
```

这个改动将块内扫描的复杂度从 $O(N)$ 降低到了 $O(log N)$（N为块大小），带来了巨大的性能飞跃。

### 2.4 Wavefront内高效扫描 `ScanWave` (基于LDS)

接下来，我们并行化`ScanWave`。我们可以采用一种无冲突（bank-conflict-free）的并行加法树算法。对于64个元素，需要6个步骤（$2^5 \< 64 \<= 2^6$）。

```cpp
__device__ void ScanWave(int32_t* lds_data) {
  int32_t lane = threadIdx.x & 63;
  // 使用volatile防止编译器过度优化，确保LDS读写顺序
  volatile int32_t* vlds_data = lds_data;

  // offset = 1
  if (lane >= 1) vlds_data[0] += vlds_data[-1];
  // 在Wavefront内部，对于这种依赖关系，AMD GCN/RDNA架构
  // 能保证执行顺序，但为保险和跨代兼容性，显式同步或使用volatile是好习惯
  
  // offset = 2
  if (lane >= 2) vlds_data[0] += vlds_data[-2];
  
  // offset = 4
  if (lane >= 4) vlds_data[0] += vlds_data[-4];
  
  // offset = 8
  if (lane >= 8) vlds_data[0] += vlds_data[-8];

  // offset = 16
  if (lane >= 16) vlds_data[0] += vlds_data[-16];

  // 关键区别：Wave64需要额外一步
  // offset = 32
  if (lane >= 32) vlds_data[0] += vlds_data[-32];
}
```

### 2.5 消除分支：零填充 (Zero Padding)

`ScanWave`中的`if (lane >= ...)`分支会导致Wavefront内的线程执行路径分化（divergence），影响性能。我们可以通过在LDS中预留空间并填充0（Zero Padding）来消除这些分支，让所有线程执行完全相同的指令。

对于64个元素的扫描，最大偏移量是32，因此我们需要为每个待扫描的数据段预留32个元素的填充空间。

```cpp
// 主机端启动时需要分配更大的LDS
void TwoPassScan(const int32_t* input, int32_t* buffer, int32_t* output,
                 size_t n) {
  size_t part_size = 1024;
  size_t part_num = (n + part_size - 1) / part_size;
  size_t block_num = std::min<size_t>(part_num, 256);
  int32_t* part = buffer;

  // 重新计算LDS大小
  size_t wave_num = part_size / 64;
  size_t padding_size = 32; // 最大偏移量
  // 为 wave_sum 和每个wave的数据段都添加padding
  size_t shm_size = (padding_size + wave_num + wave_num * (padding_size + 64)) * sizeof(int32_t);

  hipLaunchKernelGGL(ScanAndWritePartSumKernel, dim3(block_num), dim3(part_size), shm_size, 0,
                     input, part, output, n, part_num);

  // ... 后续Kernel
}

// 无分支的ScanWave
__device__ void ScanWave_NoBranch(int32_t* data) {
    volatile int32_t* vdata = data;
    vdata[0] += vdata[-1];
    vdata[0] += vdata[-2];
    vdata[0] += vdata[-4];
    vdata[0] += vdata[-8];
    vdata[0] += vdata[-16];
    vdata[0] += vdata[-32]; // Wave64
}

// ScanBlock 和 Kernel也需要相应修改以使用padding后的地址
// ... (此处省略详细的索引计算修改，其逻辑与CUDA版本类似，但所有数值需适配Wave64)
```

虽然索引计算变得复杂，但消除分支通常能带来微小但稳定的性能提升。

### 2.6 递归优化

目前，我们对`PartSum`数组的扫描`ScanPartSumKernel`是串行执行的，这是一个巨大的瓶颈。我们可以递归地调用我们的并行扫描函数来处理这个`PartSum`数组。

```cpp
void TwoPassScanRecursive(const int32_t* input, int32_t* buffer, int32_t* output,
                        size_t n) {
  size_t part_size = 1024;
  size_t part_num = (n + part_size - 1) / part_size;
  size_t block_num = std::min<size_t>(part_num, 256);
  int32_t* part = buffer;

  // 计算并启动主Kernel (ScanAndWritePartSumKernel)...
  // ...

  if (part_num >= 2) {
    // 递归调用，对part_sum数组进行扫描
    TwoPassScanRecursive(part, buffer + part_num, part, part_num);
    // 启动AddBaseSumKernel将结果加回...
    // ...
  }
}
```

通过递归，我们将整个计算流程都并行化了，性能得到大幅改善。

### 2.7 使用Wavefront Shuffle指令优化

现代GPU提供了Wavefront内部线程间直接交换数据的指令，即Shuffle指令。这可以完全避免使用LDS来进行Wavefront内部的扫描，从而减少LDS的占用和访存延迟。

HIP中对应的指令是`__shfl_up()`。它不需要`_sync`后缀，因为Wavefront内部的执行模型已保证同步。

```cpp
__device__ int32_t ScanWaveShuffle(int32_t val) {
  int32_t lane = threadIdx.x & 63;
  int32_t tmp;

  // offset = 1
  tmp = __shfl_up(val, 1);
  if (lane >= 1) val += tmp;
  
  // offset = 2
  tmp = __shfl_up(val, 2);
  if (lane >= 2) val += tmp;

  // offset = 4
  tmp = __shfl_up(val, 4);
  if (lane >= 4) val += tmp;

  // offset = 8
  tmp = __shfl_up(val, 8);
  if (lane >= 8) val += tmp;

  // offset = 16
  tmp = __shfl_up(val, 16);
  if (lane >= 16) val += tmp;

  // offset = 32 (Wave64)
  tmp = __shfl_up(val, 32);
  if (lane >= 32) val += tmp;
  
  return val;
}
```

使用Shuffle后，`ScanBlock`函数不再需要为每个Wavefront的数据在LDS中分配空间，只需为`wave_sum`数组分配即可，大大节省了LDS资源。

### 2.8 终极优化：内联GCN汇编

与NVIDIA PTX类似，AMD GPU也有其指令集架构（ISA），如GCN或RDNA。直接将CUDA的PTX内联汇编翻译到GCN ISA是**不可行的**，因为两者指令集、寄存器模型完全不同。

然而，我们可以采用相同的**思想**：利用汇编指令来消除Shuffle版本中仍然存在的`if`分支。在GCN汇编中，可以通过条件执行（`s_movk_i32` 配合执行掩码 `exec`）等技术来实现。

在HIP中，我们可以使用`asm volatile`内联GCN汇编。但这需要深入的GCN ISA知识，调试困难，且可移植性差。

**指导性建议**：
不推荐工程师手写复杂的GCN汇编。这里提供一个框架示例，说明其可能性。在实践中，**优先使用 `rocPRIM` 库**，它已经包含了由AMD专家编写的高度优化的汇编级别原语。

```cpp
__device__ __forceinline__ int32_t ScanWaveAsm(int32_t val) {
  int32_t result;
  // 警告：以下为伪代码/概念示例，并非可直接编译的GCN汇编
  // 实际GCN汇编会更复杂，需要处理VGPR和SGPR
  asm volatile(
      "// GCN汇编指令将在这里实现无分支的Wavefront Scan\n"
      "// 例如，使用 v_add_u32, s_and_saveexec_b64 等指令\n"
      "// ... 复杂的GCN汇编序列 ...\n"
      "v_mov_b32 %0, %1\n" // 示例：将结果移动到输出寄存器
      : "=v"(result) // output: v代表VGPR
      : "v"(val)   // input: v代表VGPR
      : /* clobbers */);
  return result;
}

// ScanBlock中使用Shuffle版本，并大幅减少LDS占用
__device__ __forceinline__ int32_t ScanBlockShuffle(int32_t val) {
    int32_t wave_id = threadIdx.x >> 6;
    int32_t lane = threadIdx.x & 63;
    // LDS只需存储wave_sum
    extern __shared__ int32_t wave_sum[]; 

    // 1. 各Wavefront内部扫描
    val = ScanWaveShuffle(val); // 使用高效的Shuffle版本

    // 2. 存储Wavefront总和
    if (lane == 63) {
        wave_sum[wave_id] = val;
    }
    __syncthreads();

    // 3. 第一个Wavefront扫描wave_sum
    if (wave_id == 0) {
        // 在寄存器中对加载到LDS的wave_sum进行扫描
        wave_sum[lane] = ScanWaveShuffle(wave_sum[lane]);
    }
    __syncthreads();

    // 4. 添加基准值
    if (wave_id > 0) {
        val += wave_sum[wave_id - 1];
    }
    return val;
}

// 主机端调用时，LDS大小仅为 wave_num
void TwoPassScanFinal(const int32_t* input, int32_t* buffer, int32_t* output,
                      size_t n) {
  size_t part_size = 1024;
  // ...
  size_t wave_num = part_size / 64;
  size_t shm_size = wave_num * sizeof(int32_t); // LDS只需16个int
  // 启动ScanAndWritePartSumKernel，其内部调用ScanBlockShuffle...
}
```

## 3\. 算法变种：Reduce-Then-Scan

`ScanThenFan`（我们之前的算法名）的一个变种是`ReduceThenScan`。其核心区别在于：

1.  **第一遍 (Reduce)**: 每个线程块只计算其数据块的**总和**（Summation/Reduction），而不进行块内扫描。这通常比扫描更快。
2.  对所有块的总和数组`part_sum`进行一次全局扫描（可递归调用自身）。
3.  **第二遍 (Scan)**: 再次遍历原始数据。每个线程块利用上一步计算出的`base_sum`，对其负责的数据块执行一次**局部的、从0开始的扫描**，并将`base_sum`加到每个结果上。

这里我们可以直接使用\*\*`rocPRIM`\*\*库中的高度优化原语`rocprim::block_reduce`来简化实现。

```cpp
#include <rocprim/rocprim.hpp>

__global__ void ReducePartSumKernel(const int32_t* input, int32_t* part_sum,
                                    size_t n, size_t part_num) {
  // 使用 rocprim::block_reduce
  using BlockReduce = rocprim::block_reduce<int32_t, 1024>;
  __shared__ typename BlockReduce::storage_type temp_storage;

  for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
    size_t index = part_i * blockDim.x + threadIdx.x;
    int32_t val = index < n ? input[index] : 0;
    
    int32_t sum;
    BlockReduce().reduce(val, sum, temp_storage); // 高效的块内规约
    
    if (threadIdx.x == 0) {
      part_sum[part_i] = sum;
    }
    __syncthreads(); // 确保下一轮循环前sum已写入
  }
}
// ... 后续的ScanWithBaseSum Kernel...
```

## 4\. 终极方案：消除递归的Two-Pass实现

递归虽然优雅，但会带来多次Kernel启动的开销。我们可以设计一个严格的Two-Pass（两遍）算法来消除递归。

1.  **Pass 1 (Reduce)**: 启动一个Grid，其中每个Block负责一大块数据，并计算出总和。这个Grid的大小与数据块数相同。
2.  **Device-Wide Scan**: 启动一个专用的、高度优化的Kernel（或者直接调用`rocprim::inclusive_scan`）来扫描上一步生成的`part_sum`数组。
3.  **Pass 2 (Scan with Base)**: 再次启动一个Grid，每个Block读取对应的`base_sum`，然后对其负责的原始数据块进行局部扫描并加上基准值。

<!-- end list -->

```cpp
void ReduceThenScanTwoPass(const int32_t* d_input, int32_t* d_buffer,
                           int32_t* d_output, size_t n) {
  size_t block_size = 1024;
  size_t items_per_block = 4096; // 每个块处理更多数据
  size_t part_num = (n + items_per_block - 1) / items_per_block;
  int32_t* d_part_sum = d_buffer;

  // Pass 1: 每个Block计算一个大块的总和
  hipLaunchKernelGGL(ReduceLargePartKernel, dim3(part_num), dim3(block_size), 0, 0,
                     d_input, d_part_sum, n, items_per_block);

  // 中间步骤：对d_part_sum进行全局扫描
  // 强烈推荐使用rocPRIM库
  size_t temp_storage_bytes = 0;
  rocprim::inclusive_scan(nullptr, temp_storage_bytes, d_part_sum, d_part_sum, part_num);
  hipMalloc(&d_temp_storage, temp_storage_bytes);
  rocprim::inclusive_scan(d_temp_storage, temp_storage_bytes, d_part_sum, d_part_sum, part_num);

  // Pass 2: 结合base_sum进行最终扫描
  hipLaunchKernelGGL(ScanWithBaseSumFinalKernel, dim3(part_num), dim3(block_size), 0, 0,
                     d_input, d_part_sum, d_output, n, items_per_block);
  
  hipFree(d_temp_storage);
}
```

这种非递归的两遍式方法，通过精确控制Kernel启动和利用`rocPRIM`库，通常能达到接近硬件极限的性能。

## 5\. 结语与建议

我们从一个简单的Baseline出发，通过利用LDS、实现块内和Wavefront内并行、消除分支、使用Shuffle指令，逐步将并行前缀和算法在AMD MI100平台上优化到了相当高的水平。

尽管我们手动实现了诸多优化，但与AMD官方的 **`rocPRIM`** 库相比，通常仍有差距。`rocPRIM` 是一个开源的、模板化的HIP C++并行算法库，其内部实现包含了针对AMD各代硬件（包括MI100）的深度优化，涵盖了汇编级别的调优。

**最终指导建议：**

1.  **理解原理**：学习并理解本手册中揭示的优化思路和平台差异（特别是Wave32 vs Wave64）至关重要。
2.  **原型开发**：在项目初期或性能要求不极致的场景，可以基于本手册中的`Shuffle`版本进行快速开发和验证。
3.  **生产环境**：对于追求极致性能的生产环境代码，**强烈推荐直接使用 `rocPRIM` 库**。它提供了健壮、高效且经过充分测试的`rocprim::inclusive_scan`和`rocprim::exclusive_scan`等原语。秉承“打不过就加入”的原则，拥抱官方库是最高效的选择。

通过本手册的学习，您现在应具备在AMD ROCm/HIP平台上进行高性能并行算法设计与优化的坚实基础。