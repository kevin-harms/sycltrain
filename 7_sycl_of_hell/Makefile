CC=clang++
#Yes. module show dpcc to know why
CCFLAGS=-nostdinc++ -I/soft/compilers/gcc/7.4.0/linux-rhel7-x86_64/include/c++/7.4.0 -I/soft/compilers/gcc/7.4.0/linux-rhel7-x86_64/include/c++/7.4.0/x86_64-pc-linux-gnu -L/soft/compilers/gcc/7.4.0/linux-rhel7-x86_64/lib64 -fsycl -lOpenCL

SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))

all: $(PRGS) 
%: %.cpp
	$(CC) -o $@ $^ $(CCFLAGS)

clean:
	rm -f $(PRGS)
