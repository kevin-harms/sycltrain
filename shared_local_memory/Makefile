CXX=clang++
CXXFLAGS=-fsycl --gcc-toolchain=/soft/compilers/gcc/7.4.0/linux-rhel7-x86_64 

all: test

test: main.o
	$(CXX) -o $@ $(CXXFLAGS) $(<F) -lsycl -lOpenCL

clean::
	rm -f *.o test
