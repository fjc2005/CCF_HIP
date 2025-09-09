# Softmax

## 描述

实现一个在 GPU 上计算一维浮点数组 **softmax** 的程序。  
给定输入向量 $\mathbf{x} = [x_1, x_2, \dots, x_N]$，输出 $\mathbf{y} = [y_1, y_2, \dots, y_N]$，其中：

$$
y_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

由于直接指数运算可能导致溢出/下溢，**必须**使用数值稳定的形式：

$$
m = \max_i x_i, \quad
t_i = e^{x_i - m}, \quad
S = \sum_{i=1}^{N} t_i, \quad
y_i = \frac{t_i}{S}
$$

## 要求

* `solve` 函数签名必须保持不变。
* 必须使用上述数值稳定的公式。
* 只允许**单 GPU**实现（禁止多 GPU）。

## 代码结构

```
.
├── main.cpp        # 读取输入，调用 solve()，输出结果
├── kernel.hip      # GPU 内核及 solve() 实现
├── main.h          # 公共头文件及 solve() 声明
├── Makefile
├── README.md
└── testcases       # 本地验证用的样例测试数据
```

## 编译与运行

### 编译

```bash
make
```

生成可执行文件：`softmax`

### 运行

```bash
./softmax input.txt
```

---

## 测试用例

`testcases/` 文件夹包含 **10** 个样例输入文件及对应输出。

运行样例：

```bash
./softmax testcases/1.in
```

容差要求：

* 绝对容差：$1\times 10^{-6}$
* 相对容差：$1\times 10^{-5}$
* 最小分母：$1\times 10^{-12}$

---

### 输入格式

* 第一行是一个整数 $N$，表示数组长度。
* 第二行是 $N$ 个用空格分隔的浮点数。

**示例**

```
3
1.0 2.0 3.0
```

**约束**

* $1 \le N \le 100{,}000{,}000$
* $\text{input}[i]$ 为浮点数

---

### 输出格式

* 输出 $N$ 个浮点数，表示 **softmax** 结果 $y_1, y_2, \dots, y_N$。
* 每个数需满足上述容差要求。
* 数字之间用空格分隔，末尾换行。

**示例**

```
0.090 0.244 0.665
```

---

## 提交说明

提交的文件夹必须命名为 `softmax`

包含所有必需的源文件（`main.cpp`、`kernel.hip`、`main.h`、`Makefile`），可直接通过以下命令编译：

```bash
make
```

评测流程：

```bash
cd $HOME/hip_programming_contest/softmax
make
./softmax <hidden_testcase.txt>
```

---