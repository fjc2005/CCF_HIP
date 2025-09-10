HIPCC = hipcc

TARGET = apsp

SRCS = main.cpp 

HEADERS = main.h

CXXFLAGS = -O3 -std=c++17

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o