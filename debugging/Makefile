CXX=clang++
CXXFLAGS=-g -fsycl --gcc-toolchain=/soft/compilers/gcc/7.4.0/linux-rhel7-x86_64 

all: test

test: main.o diag.o
	$(CXX) -o $@ $(CXXFLAGS) $^ -lsycl -lOpenCL

clean::
	rm -f *.o test
