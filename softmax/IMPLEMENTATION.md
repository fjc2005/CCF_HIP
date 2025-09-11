# IMPLEMENTATION NOTES

## 目标
实现 1-D 浮点数组的数值稳定 softmax（单 GPU、保持 `solve` 签名不变）。

## 代码结构与改动

- `kernel.hip`
  - 新增 4 个 GPU kernel：
    - `kernel_block_max`: 分块并行求最大值
    - `kernel_reduce_max_final`: 单块归约得到全局最大值
    - `kernel_compute_exp_and_block_sum`: 计算 `t_i = exp(x_i - m)` 并做分块求和
    - `kernel_reduce_sum_final`: 单块归约得到 `S = sum(t_i)`
    - `kernel_normalize`: `y_i = t_i / S`
  - 在 `solve` 中：
    1) 从 Host 将输入拷贝到 GPU；
    2) 两阶段归约得到 `m`；
    3) 计算 `t_i` 与分块和并二次归约得到 `S`；
    4) `S` 加下限 `1e-12`；
    5) 归一化得到输出并拷回 Host；
    6) 释放资源。

- `Makefile`
  - 保持不变：`hipcc -O2` 编译 `main.cpp kernel.hip` 为 `softmax`。

## 数值稳定性

- 使用 `m = max(x)`，计算 `exp(x_i - m)`，避免溢出/下溢。
- 对分母 `S` 施加最小值 `1e-12`，满足 README 容忍度要求。

## 运行

```bash
make
./softmax testcases/1.in
```

## Slurm 提交

- 使用 `softmax.sbatch`（见项目根目录）提交到集群。依赖 `verify.py` 校验输出。

```bash
sbatch /home/user095/hip_programming_contest/softmax/softmax.sbatch
```

日志在 `softmax_<jobid>.log`。


## 输出优化（FAST_OUTPUT）

### 背景
- 在大规模 N 时，主机侧将结果写到 stdout 成为显著瓶颈。参考 `apsp` 项目的 `FAST_OUTPUT` 方案（分块缓冲 + 高效数字格式化 + 关闭 iostream 同步），我们在本项目中复用同一思想优化结果输出。

### 优化前实现
- 逐元素通过 iostream 输出：
  - `for (i) std::cout << output[i] << " ";`
  - 行末输出 `std::endl`。
- 特点：简单但频繁的格式化与 flush，导致系统调用次数多、吞吐低。

### 优化后实现
- 新增 `print_output_fast(const float* data, int N)`（见 `main.cpp`）：
  - 使用约 1MB 的 `std::vector<char>` 作为输出缓冲；
  - 使用 `std::snprintf("%.6g")` 将浮点写入缓冲（与默认 iostream 精度 6 位有效数字保持一致）；
  - 每个数字后追加一个空格，末尾追加一个换行，严格保持与原有输出字节级一致；
  - 缓冲写满时通过 `std::fwrite` 批量写出；
  - 程序开始输出前执行：`std::ios_base::sync_with_stdio(false); std::cout.tie(nullptr); setvbuf(stdout, nullptr, _IOFBF, 8<<20);` 将 stdout 设为全缓冲并关闭 iostream 同步。
- 通过编译开关控制：`Makefile` 默认添加 `-DFAST_OUTPUT=1`，可通过 `make CXXFLAGS='-O2 -DFAST_OUTPUT=0'` 回退旧实现。

### 核心原理（为何更快）
- 批量缓冲：显著减少 `write`/`fwrite` 调用次数；
- 轻量格式化：`snprintf`/`to_chars` 相比 iostream 的类型擦除与本地化机制更轻；
- 避免 `std::endl` 的隐式刷新，减少不必要的 `fflush`；
- 关闭 iostream/stdio 同步，避免重复缓冲与锁。

### 结果一致性
- 与原实现保持相同的数值打印风格与分隔符：每个数后有一个空格，末尾换行；默认 6 位有效数字（`%.6g`）。
- 通过 `verify.py` 校验绝对/相对误差容忍度不受影响。

### 预期性能收益
- 在 N 很大（例如 1e8 元素以极端测试）时，输出阶段吞吐可从 <200MB/s 提升至数百 MB/s（依赖磁盘与系统 I/O）。
- 在本地/集群评测中，输出阶段用时通常可获得 3×–6× 的加速（参考 `apsp` 的日志数据），端到端总时间受输出占比影响而定。


