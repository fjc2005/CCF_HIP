## 实现说明

### 概述
实现了基于 HIP 的整型数组前缀和（inclusive scan）单 GPU 版本，支持大规模数据：
- 块内扫描（Hillis-Steele）得到每个元素的前缀和与每块块和
- 对块和数组进行递归扫描
- 将块偏移加回至对应块的每个元素

### 关键文件
- `kernel.hip`: 包含 GPU 内核与 `solve()` 的完整实现
- `main.cpp`: 读取输入、调用 `solve()`、打印输出（原样保留）
- `main.h`: 头文件与声明（原样保留）

### 实现细节
1. 常量与块大小
   - 使用 `BLOCK_SIZE = 256`，适配常见 AMD GPU SM 容量。

2. 块内扫描 `block_scan_inclusive_kernel`
   - 使用共享内存 `extern __shared__ int sdata[]` 存储当前块数据
   - 采用 Hillis-Steele 方式进行 in-place 包含式扫描
   - 写回每个元素的扫描结果到 `out`
   - 由块内最后一个有效线程写出该块块和到 `block_sums[blockIdx.x]`

3. 偏移累加 `add_block_offsets_kernel`
   - 对于第 `b` 个块，为该块所有元素加上 `scanned_block_sums[b-1]`（块扫描结果的前一项），`b=0` 时偏移为 0

4. 递归扫描 `device_scan_inclusive`
   - 先对输入按块扫描，产出 `out` 与 `block_sums`
   - 若 `numBlocks > 1`，递归调用自身对 `block_sums` 扫描得到 `scanned_block_sums`
   - 调用 `add_block_offsets_kernel` 将块偏移加回 `out`

5. `solve()`
   - 在 GPU 上分配 `d_in`、`d_out`，拷入输入后调用 `device_scan_inclusive`
   - 将结果拷回 `output`，释放 GPU 资源

### 构建与运行
```
make
./prefix_sum testcases/1.in
```

### 测试
- 批量运行 `testcases/` 目录下 1-10 的样例，实际输出与期望无差异

### 注意与后续优化
- 目前未对 HIP API 的返回值进行错误处理（编译器会产生告警），在评测环境如需严格处理，可检查返回码并在失败时早退
- Hillis-Steele 共 O(BLOCK_SIZE log BLOCK_SIZE) 操作量，可进一步改为 Blelloch 扫描或使用 warp-level primitives 优化
- 对于超大 N，递归层数为 O(log_{BLOCK_SIZE} N)，内存开销为 O(块数)


## Improvements based on README hints — 2025-09-09

### 动机与 hint 摘要
- 基于 README 的分块扫描三阶段提示（块内扫描、块和扫描、加偏移），实现层级式前缀和。
- 采用工作高效的 Blelloch 扫描以降低块内工作量，同时保留原 Hillis-Steele 作为对照。
- 支持非 2 的幂长度：块内对不足 `BLOCK_SIZE` 的尾块做 `local_n` 限定，并用最近的 2 次幂 `nPow2` 做 upsweep/downsweep，越界元素以 0 填充。
- 保持全局内存合并访问：按块连续读写；共享内存用于块内扫描。

### 算法与实现细节
- 层级结构：
  - 第 1 阶段：`block_scan_inclusive_kernel_{hs, blelloch}` 在共享内存完成块内 inclusive 扫描，并写出 `block_sums[b]`。
  - 第 2 阶段：递归调用 `device_scan_inclusive` 对 `block_sums` 做扫描得到 `scanned_block_sums`。
  - 第 3 阶段：`add_block_offsets_kernel` 将 `scanned_block_sums[b-1]` 加回块 `b` 的所有元素。
- 共享内存布局：每块 `BLOCK_SIZE` 个 `int`，尾块使用 `local_n` 标识有效元素个数，超出部分置 0。
- Blelloch 内核：
  - upsweep 在 `[0, nPow2)` 范围做树形加法，`nPow2` 为不小于 `local_n` 的最小 2 次幂。
  - 将末位置 0 做 exclusive 化，再经 downsweep 恢复为 exclusive 结果；最终 `inclusive = exclusive + 原值` 写回。
- 接口与开关：新增 `set_scan_impl(int)` 选择实现（0=Hillis-Steele，1=Blelloch），便于 A/B 对比。

### 复杂度与访存分析
- 块内：
  - Hillis-Steele：O(B log B) 操作，O(B) 共享内存。
  - Blelloch：O(B) 操作级别（树形），两段对数轮次同步。
- 全局：每层执行一次读写并写出块和；层数约为 O(log_{B} N)。
- 访存：全局读写合并；共享内存连续访问。未加显式 padding，因索引模式对 bank 冲突敏感度较低，且 `BLOCK_SIZE=256` 在实测无明显冲突热点。

### 正确性要点
- 块间偏移使用扫描后的前一项（b=0 时为 0），与 README 的“块间 exclusive 扫描”一致。
- 尾块 `local_n` 与 `nPow2` 处理确保非 2 的幂长度正确；超范围元素视为 0。
- 程序提供 CPU 参考 `cpu_inclusive_scan` 并可 `--verify` 逐元素校验。

### 参数选择
- `BLOCK_SIZE=256`：在 MI100 上是较稳妥选择，能保持较高占用与足够并行度。
- `--impl hs|blelloch`：默认 `blelloch`，可切换到 `hs` 做对照。
- `--repeat K`：重复次数用于稳定计时。

### 测试与基准
- 正确性：对 `testcases/1..10.in` 批量运行并 `--verify`，全部 `verify: OK`。
- 基准（MI100，本机）：`testcases/10.in`，`--repeat 10`：
  - Blelloch：约 0.80s 实时；HS：约 0.80s（样例较小，差异不显著）。
  - 更大规模下 Blelloch 预计更具优势（工作更高效）。

### 局限与后续工作
- 未显式处理共享内存 bank padding；如出现冲突，可加入 `+ (tid >> 5)` 等轻量 padding。
- 未使用 warp/shuffle 原语做更细粒度优化；可在块内替换部分共享内存同步。
- 未对 HIP API 返回码逐项检查；评测环境若需严格容错可补充错误处理。

### 复现步骤
```
cd $HOME/hip_programming_contest/prefix_sum
make clean && make -j
./prefix_sum testcases/1.in --verify
for f in testcases/*.in; do ./prefix_sum "$f" --verify > my_outputs/$(basename "$f" .in).actual; done
/usr/bin/time -f "%E real, %M KB" ./prefix_sum testcases/10.in --repeat 10 --impl blelloch > /dev/null
/usr/bin/time -f "%E real, %M KB" ./prefix_sum testcases/10.in --repeat 10 --impl hs > /dev/null
```

## 高性能输出优化 — 2025-09-11

### 动机
基于 `apsp` 项目中的高效输出实现，将 `prefix_sum` 项目原本使用的标准 `std::cout` 输出方式替换为高性能的 `fast_output` 方法，以显著减少输出文件所需的时间。

### 优化技术
- **大缓冲区**：使用 32MB 的缓冲区减少系统调用频率
- **手动整数转换**：使用 `std::to_chars` 替代格式化输出，避免格式化开销
- **批量写入**：使用 `std::fwrite` 进行块写入，比逐个字符输出更高效
- **stdio 加速**：
  - `std::ios_base::sync_with_stdio(false)` 解除 C++ 流与 C stdio 的同步
  - `setvbuf(stdout, nullptr, _IOFBF, 32 * 1024 * 1024)` 设置全缓冲模式

### 实现细节
1. **新增 `fast_output_array` 函数**：
   - 基于 `apsp/main.cpp` 中的 `print_matrix_pinned` 函数设计
   - 专门优化一维数组的输出（相比二维矩阵更简化）
   - 保持与原输出完全相同的格式（数字间空格分隔，末尾换行）

2. **编译时开关**：
   - 定义 `FAST_OUTPUT` 宏，默认启用
   - 当禁用时自动回退到标准 `std::cout` 输出方式

3. **缓冲区管理**：
   - 智能缓冲区溢出检测和刷新
   - 为数字输出预留足够空间（最多16字节）
   - 最终确保所有数据完全输出并刷新

### 代码修改
1. **头文件更新**（`main.h`）：
   - 添加 `#include <charconv>` 支持 `std::to_chars`
   - 添加 `#include <cstdio>` 和 `#include <cstdlib>` 支持 C 标准库函数

2. **主程序更新**（`main.cpp`）：
   - 添加 `fast_output_array` 函数实现
   - 在 `main` 函数开始处添加 stdio 加速设置
   - 将原来的循环输出替换为单次 `fast_output_array` 调用

### 兼容性保证
- 输出格式完全兼容：数字间用单个空格分隔，末尾添加换行符
- 保持原有的输出计时功能不变
- 支持编译时禁用优化，回退到原始输出方式

### 性能收益
- **减少系统调用**：从 O(N) 次 `operator<<` 调用减少到 O(1) 次 `fwrite` 调用
- **避免格式化开销**：`std::to_chars` 比流操作符更高效
- **大缓冲区**：减少内核态切换频率
- **预期改进**：输出时间应有显著降低，特别是对于大型数据集

### 测试验证
- 输出结果与原版本完全一致
- 保持所有现有测试用例的兼容性
- 性能提升在使用 `--timing` 选项时可观察到输出时间的减少

