#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <assert.h>

using namespace cl;

void display_device(const sycl::device_selector &selector)
{
    sycl::platform platform = sycl::platform(selector);
    std::vector<sycl::device> devices = platform.get_devices();

    std::cout << platform.get_info<sycl::info::platform::name>() << std::endl;
    std::cout << "\t" << platform.get_info<sycl::info::platform::version>() << std::endl;
    for (auto const device : devices)
    {
        if (device.is_gpu())
        {
            std::cout << "\t" << "name: " << device.get_info<sycl::info::device::name>() << std::endl;
            std::cout << "\t" << "eu: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
            std::cout << "\t" << "vector: " << device.get_info<sycl::info::device::preferred_vector_width_char>() << std::endl;
            std::cout << "\t" << "local mem: " << device.get_info<sycl::info::device::local_mem_size>() << std::endl;
        }
        else
        {
            std::cout << "\t" << "name: " << device.get_info<sycl::info::device::name>() << std::endl;
            std::cout << "\t" << "cores: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
        }
    }

    return;
}
