#include <iostream>
#include <vector>
#include <CL/sycl.hpp>

void force_allocate(cl::sycl::handler& cgh, int global_range, cl::sycl::buffer<cl::sycl::cl_int, 1> buffer) {
    // Create a device accesors and use it. This will force the allocation
    cl::sycl::accessor<cl::sycl::cl_int, 1, cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer> accessorA(buffer, cgh, global_range);
    cgh.single_task<class allocate>([=]() { accessorA[0]; });
}



int main(int argc, char** argv) {

   const auto global_range =  (size_t) atoi(argv[1]);
   // Create array
   std::vector<int> A(global_range, 1);

  // Selectors determine which device kernels will be dispatched to.
  cl::sycl::default_selector selector;
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
  cl::sycl::queue myQueue(selector);

  // Create buffer.
  cl::sycl::buffer<cl::sycl::cl_int, 1> bufferA(A.data(), global_range);

  std::cout << "Running on "
            << myQueue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";

  //Force the allocation of the buffer.
  myQueue.submit(std::bind(force_allocate, std::placeholders::_1, global_range, bufferA));

  }  // End of scope, wait for the queued work to stop.
  return 0;
}
