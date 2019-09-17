CXX=clang++
CXXFLAGS=-fsycl --gcc-toolchain=/soft/compilers/gcc/8.2.0/linux-rhel7-x86_64

all: main

main: main.o
	$(CXX) -o $@ $(CXXFLAGS) $^ -lsycl -lOpenCL

clean::
	rm -f *.o main
