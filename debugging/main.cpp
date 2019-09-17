#include <stdio.h>
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl;

void display_device(const sycl::device_selector &selector);

#define SIZE 16

void print_buffer (sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::host_buffer> &acc)
{
    for (int i = 0; i < SIZE; i++)
    {
        std::cout << acc[i] << " ";
    }
    std::cout << std::endl;

    return;
}

int main (int argc, char **argv)
{

    const sycl::device_selector &dev = sycl::gpu_selector();

    sycl::async_handler ah = \
        [](sycl::exception_list elist)
        {
            for( auto e : elist)
            {
                std::rethrow_exception(e);
            }
        };

    std::vector<int> v(SIZE);

    sycl::queue q(dev, ah);
    display_device(dev);

    sycl::buffer<int, 1> b { sycl::range<1>(SIZE) };
    auto ha_b = b.get_access<sycl::access::mode::read_write>();

    for (int i = 0; i < SIZE; i++)
    {
        ha_b[i] = 1;
    }

    print_buffer(ha_b);
    std::cout << "run kernel" << std::endl;

    try
    {

    q.submit([&](sycl::handler &cgh)
    {
        auto da_b = b.get_access<sycl::access::mode::write>(cgh);
        int *ptr = 0;

        cgh.parallel_for<class kernel1>(sycl::nd_range<1>(sycl::range<1>(16),
                                                          sycl::range<1>(1)),
        [=](sycl::nd_item<1> i)
        {
            size_t idx = i.get_global_linear_id();
            *ptr = 0;
            da_b[idx] = idx;
        });
    });

    // Observe difference in behavior between wait and wait_and_throw
    // q.wait();
    q.wait_and_throw();

    auto ha2_b = b.get_access<cl::sycl::access::mode::read_write>();
    print_buffer(ha2_b);

    }
    catch (sycl::exception &e)
    {
        std::cout << "Async Exception: " << e.what() << std::endl;
    }

    return 0;
}
