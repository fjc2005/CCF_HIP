HIPCC = hipcc

TARGET = prefix_sum

SRCS = main.cpp kernel.hip

HEADERS = main.h

CXXFLAGS = -O3 $(EXTRA_FLAGS)

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o
