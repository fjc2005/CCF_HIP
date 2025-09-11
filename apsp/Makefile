HIPCC = hipcc

TARGET = apsp

SRCS = main.cpp 

HEADERS = main.h

CXXFLAGS = -O3 -std=c++17 -DBATCH_SSSP=1 -DADAPTIVE_K=1 -DFAST_OUTPUT=1

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o