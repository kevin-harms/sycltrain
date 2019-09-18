CC=clang++
#Yes. module show dpcc to know why
CCFLAGS=-std=c++17 -fsycl -lOpenCL

SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))

all: $(PRGS) 
%: %.cpp
	$(CC) -o $@ $^ $(CCFLAGS)

clean:
	rm -f $(PRGS)
