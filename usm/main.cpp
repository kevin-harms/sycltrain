#include <CL/sycl.hpp>

using namespace cl;

#define SIZE 10

int main (int argc, char **argv)
{
    sycl::queue q{sycl::cpu_selector()};
    sycl::device dev = q.get_device();
    sycl::context ctex = q.get_context();

    double* a = static_cast<double*>(sycl::malloc_shared(SIZE*sizeof(double), dev, ctex));
    double* b = static_cast<double*>(sycl::malloc_shared(SIZE*sizeof(double), dev, ctex));
    double* c = static_cast<double*>(sycl::malloc_shared(SIZE*sizeof(double), dev, ctex));

    for (int i = 0; i < SIZE; i++)
    {
        a[i] = i * 0.5;
        b[i] = i * 0.5;
    }

    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;

    // Advise runtime how memory will be used
    // auto e = q.mem_advise(a, SIZE*sizeof(double), 0);
    // e.wait();

    q.submit ([&](sycl::handler &cgh) {
        cgh.parallel_for<class kernel_1138>(sycl::range<1>(SIZE),
        [=](sycl::id<1> i) {
            size_t ii = i[0];
            c[ii] = a[ii] + b[ii];
        });
    });

    q.wait();

    for (int i = 0; i < SIZE; i++)
    {
        if (c[i] != ((i*0.5)+(i*0.5)))
        {
            std::cout << "value incorrect: " << c[i] << " " << i << std::endl;
        }
    }

    sycl::free(a, ctex);
    sycl::free(b, ctex);
    sycl::free(c, ctex);

    std::cout << "Run complete" << std::endl;

    return 0;
}
