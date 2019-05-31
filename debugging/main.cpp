#include <stdio.h>
#include <CL/sycl.hpp>
#include <iostream>

void display_device(const cl::sycl::device_selector &selector);

#define SIZE 16

void print_buffer (cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer> &acc)
{
    for (int i = 0; i < SIZE; i++)
    {
        std::cout << acc[i] << std::endl;
    }
}

int main (int argc, char **argv)
{

    const cl::sycl::device_selector &dev = cl::sycl::gpu_selector();

    cl::sycl::async_handler ah = \
        [](cl::sycl::exception_list elist)
        {
            for( auto &e : elist)
            {
                try { std::rethrow_exception(e); }
                catch (cl::sycl::exception &ce) { std::cout << "Async Exception: " << ce.what() << std::endl; }
                catch (std::exception &se) { std::cout << "Async Exception: " << se.what() << std::endl; }
            }
            exit(1);
        };

    std::vector<int> v(SIZE);

    try
    {

        cl::sycl::queue q(dev, ah);
        display_device(dev);

        cl::sycl::buffer<int, 1> b { cl::sycl::range<1>(SIZE) };
        cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer> ha_b = b.get_access<cl::sycl::access::mode::read_write>();

        for (int i = 0; i < SIZE; i++)
        {
             ha_b[i] = 1;
        }

        print_buffer(ha_b);
        std::cout << "run kernel" << std::endl;

        q.submit([&](cl::sycl::handler &h)
        {
            auto da_b = b.get_access<cl::sycl::access::mode::write>(h);

            h.parallel_for<class kernel1>(cl::sycl::nd_range<1>(cl::sycl::range<1>(16), cl::sycl::range<1>(1)),
            //h.parallel_for<class kernel1>(cl::sycl::nd_range<1>(cl::sycl::range<1>(5), cl::sycl::range<1>(2)),
                [=](cl::sycl::nd_item<1> i)
                {
                    size_t idx = i.get_global_linear_id();
                    da_b[idx] = idx;
                }
            );
        }
        );
        q.wait_and_throw();
        //q.wait();

        cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer> ha2_b = b.get_access<cl::sycl::access::mode::read_write>();
        print_buffer(ha2_b);
    }
    catch (cl::sycl::exception &e)
    {
        std::cout << "Exception: " << e.what() << std::endl;
    }


    return 0;
}
