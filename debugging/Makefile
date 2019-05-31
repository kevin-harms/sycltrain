CXX=clang++
CXXFLAGS=-g -fsycl

all: test

test: main.o diag.o
	$(CXX) -o $@ $(CXXFLAGS) $(<F) -lsycl -lOpenCL

clean::
	rm -f *.o test
