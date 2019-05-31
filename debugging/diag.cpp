#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <assert.h>

void display_device(const cl::sycl::device_selector &selector)
{
    cl::sycl::platform platform = cl::sycl::platform(selector);
    std::vector<cl::sycl::device> devices = platform.get_devices();

    std::cout << platform.get_info<cl::sycl::info::platform::name>() << std::endl;
    std::cout << "\t" << platform.get_info<cl::sycl::info::platform::version>() << std::endl;
    for (auto const device : devices)
    {
        if (device.is_gpu())
        {
            std::cout << "\t" << "name: " << device.get_info<cl::sycl::info::device::name>() << std::endl;
            std::cout << "\t" << "eu: " << device.get_info<cl::sycl::info::device::max_compute_units>() << std::endl;
            std::cout << "\t" << "vector: " << device.get_info<cl::sycl::info::device::preferred_vector_width_char>() << std::endl;
            std::cout << "\t" << "local mem: " << device.get_info<cl::sycl::info::device::local_mem_size>() << std::endl;
        }
        else
        {
            std::cout << "\t" << "name: " << device.get_info<cl::sycl::info::device::name>() << std::endl;
            std::cout << "\t" << "cores: " << device.get_info<cl::sycl::info::device::max_compute_units>() << std::endl;
        }
    }

    return;
}
