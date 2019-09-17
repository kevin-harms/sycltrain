#include <CL/sycl.hpp>

using namespace cl;

#define SIZE 4096

int main (int argc, char **argv)
{
    sycl::queue q {sycl::default_selector() };
    sycl::buffer<int, 1> Ba(sycl::range<1>(SIZE)); 
    sycl::buffer<int, 1> Bsum(sycl::range<1>(1));
    auto Ha   = Ba.get_access<sycl::access::mode::write>();
    auto Hsum = Bsum.get_access<sycl::access::mode::read_write>();
    int expected_sum = 0;

    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    for (int i = 0; i < SIZE; i++)
    {
        Ha[i] = i;
    }
 
    Hsum[0] = 0;

    for (int i = 0; i < SIZE; i++)
    {
        expected_sum += i;
    }

    q.submit([&](sycl::handler &cgh)
    {
        auto Aa = Ba.get_access<sycl::access::mode::read_write>(cgh);
        auto Asum = Bsum.get_access<sycl::access::mode::atomic>(cgh);

        cgh.parallel_for<class kernel_atomic>(sycl::range<1>(SIZE),
          [=](sycl::id<1> i) 
          {
              Asum[0].fetch_add(Aa[i[0]]);
          });
    });
    q.wait();

    auto Hsum2 = Bsum.get_access<sycl::access::mode::read>();

    if (Hsum2[0] != expected_sum)
    {
        std::cout << "Unexecpted sum: " << Hsum2[0] << " " << expected_sum << std::endl;
    }
    else
    {
        std::cout << "Results correct" << std::endl;
    }
  
    return 0;
}
