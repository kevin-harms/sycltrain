#include <stdio.h>
#include <CL/sycl.hpp>
#include <iostream>

void display_device(const cl::sycl::device_selector &selector);

int main (int argc, char **argv)
{

    const cl::sycl::device_selector &cpu_dev = cl::sycl::cpu_selector();
    const cl::sycl::device_selector &gpu_dev = cl::sycl::gpu_selector();

    cl::sycl::async_handler ah = \
        [](cl::sycl::exception_list elist) { for( auto &e : elist) { std::cout << "Async Exception: " << e.what() << std::endl; } };

    display_device(cpu_dev);
    cl::sycl::queue q(cpu_dev, ah);

    cl::sycl::buffer<int, 1> b(cl::sycl::range<1>(SIZE));
    cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer> ha_b = b.get_access<cl::sycl::access::mode::write>();

    for (int i = 0; i < SIZE; i++)
    {
        ha_b[i] = 0;
    }

    print_buffer(ha_b);

    try
    {
        q.submit([&](cl::sycl::handler &h)
        {
             cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, cl::sycl::access::target::local> acc = cl::sycl::accessor<int, 1,  cl::sycl::access::mode::write, cl::sycl::access::target::local>(cl::sycl::range<1>(SIZE), h);
            auto da_b = b.get_access<cl::sycl::access::mode::write>(h);

            h.parallel_for<class kernel1>(cl::sycl::nd_range<1>(cl::sycl::range<1>(100000), cl::sycl::range<1>(10)),
                [=](cl::sycl::nd_item<1> i)
                {
                    size_t idx = i.get_global_linear_id();
                    da_b[idx] = idx;
                }
            );
        }
        );
        q.wait();
        // q.wait_and_throw();
    }
    catch (cl::sycl::exception &e)
    {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    print_buffer(ha_b);

    return 0;
}
