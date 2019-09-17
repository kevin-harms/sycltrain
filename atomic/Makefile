CXX=clang++
CXXFLAGS=-fsycl

all: main

main: main.o
	$(CXX) -o $@ $(CXXFLAGS) $^ -lsycl -lOpenCL

clean::
	rm -f *.o main

