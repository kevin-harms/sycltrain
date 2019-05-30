#include <stdio.h>
#include <CL/sycl.hpp>
#include <iostream>

#define SIZE 16

void print_buffer (cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer> &acc)
{
    for (int i = 0; i < SIZE; i++)
    {
        std::cout << acc[i] << std::endl;
    }
}

int main (int argc, char **argv)
{
    const cl::sycl::device_selector &cpu_dev = cl::sycl::cpu_selector();
    cl::sycl::queue q(cpu_dev);
    cl::sycl::buffer<int, 1> b(cl::sycl::range<1>(SIZE));
    cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer> ha_b = b.get_access<cl::sycl::access::mode::write>();

    for (int i = 0; i < SIZE; i++)
    {
        ha_b[i] = 0;
    }
    
    print_buffer(ha_b);

    q.submit([&](cl::sycl::handler &h)
        {
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> acc = cl::sycl::accessor<int, 1,  cl::sycl::access::mode::read_write, cl::sycl::access::target::local>(cl::sycl::range<1>(SIZE), h);
            cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> da_b = b.get_access<cl::sycl::access::mode::write>(h);

            h.parallel_for<class kernel1>(cl::sycl::range<1>(SIZE),
                [=](cl::sycl::id<1> i)
                {
                    int x = i[0];
                    int y;
                    acc[x] = x;
                    for (int j = 0, y = 0; j < 1000000; j++)
                    {
                        y = y + j;
                    }
                    if (acc[x] != x)
                    {
                        da_b[x] = 1;
                    }
                }
            );
/*
            h.parallel_for_work_group<class kernel1>(
                range<1>(4), 
                [=](cl::sycl::group<1> g)
                {
                    g.parallel_for_work_item(
                        [&](cl::sycl::h_item<1> wi) 
                        {
                            int i = wi.get_global_id();
                            acc[i] = 7;
                        }
                    );
                }
            );
*/
        }
    );
    q.wait();

    print_buffer(ha_b);

    return 0;
}
