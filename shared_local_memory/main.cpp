#include <stdio.h>
#include <CL/sycl.hpp>
#include <iostream>

#define SIZE 256


int main (int argc, char **argv)
{
    const cl::sycl::device_selector &cpu_dev = cl::sycl::cpu_selector();
    const cl::sycl::device_selector &gpu_dev = cl::sycl::gpu_selector();
    cl::sycl::queue q(gpu_dev);

    q.submit([&](cl::sycl::handler &h)
        {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> acc = cl::sycl::accessor<int, 1,  cl::sycl::access::mode::read_write, cl::sycl::access::target::local>(cl::sycl::range<1>(SIZE), h);

            h.parallel_for<class kernel1>(cl::sycl::nd_range<1>(cl::sycl::range<1>(SIZE), cl::sycl::range<1>(SIZE)),
                [=](cl::sycl::nd_item<1> i)
                {
                    int x = i.get_global_linear_id();
                    int y;
                    acc[x] = x;
                    if ((x < (SIZE-1)) &&
                        (acc[x+1] != (x+1)))
                    {
                        printf("unexpected value: %d %d\n", acc[x+1], x+1);
                    }
                }
            );
        }
    );
    q.wait();

    return 0;
}
