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


