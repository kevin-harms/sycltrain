CXX=clang++
CXXFLAGS=-fsycl

all: test

test: main.o
	$(CXX) -o $@ $(CXXFLAGS) $(<F) -lsycl -lOpenCL

clean::
	rm -f *.o test
