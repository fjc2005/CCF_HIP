HIPCC = hipcc

TARGET = apsp

SRCS = main.cpp 

HEADERS = main.h

# 默认优化开关（可通过 make CXXFLAGS+="..." 覆盖）
# 针对 MI100 (gfx908) 指定目标架构，并关闭 GPU RDC 以利于跨函数内联
CXXFLAGS = -O3 -std=c++17 -DBATCH_SSSP=1 -DADAPTIVE_K=1 -DFAST_OUTPUT=1 --offload-arch=gfx908 -fno-gpu-rdc -ffast-math

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o