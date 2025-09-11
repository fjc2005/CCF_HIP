HIPCC = hipcc

TARGET = softmax

SRCS = main.cpp kernel.hip

HEADERS = main.h

CXXFLAGS = -O3 -DFAST_OUTPUT=1

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o
