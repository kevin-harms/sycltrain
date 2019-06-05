#include <CL/sycl.hpp>
template <typename dataT>
class vector_init_kernel{
 public:
   vector_init_kernel(dataT accessorA_): accessorA{accessorA_} {}
   void operator()(cl::sycl::item<1> idx) { accessorA[idx.get_id()] = idx.get_id(0); }
 private:
   dataT accessorA;
};

void f_write(cl::sycl::handler& cgh, int global_range, cl::sycl::buffer<cl::sycl::cl_int, 1> bufferA) {
    auto accessorA = bufferA.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.parallel_for(cl::sycl::range<1>(global_range), vector_init_kernel(accessorA));
}

void f_discard_write(cl::sycl::handler& cgh, int global_range, cl::sycl::buffer<cl::sycl::cl_int, 1> bufferA) {
    auto accessorA = bufferA.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(cl::sycl::range<1>(global_range), vector_init_kernel(accessorA));
}


using namespace cl::sycl;

int main(int argc, char** argv) {

  const auto global_range =  (size_t) atoi(argv[1]);
  const auto write = (bool) atoi(argv[2]);

   // Crrate array
   int A[global_range];

  // Selectors determine which device kernels will be dispatched to.
  cl::sycl::gpu_selector selector;
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
  cl::sycl::queue myQueue(selector);
  std::cout << "Running on "
            << myQueue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";
//  _
// /   _  ._   _| o _|_ o  _  ._   _. |
// \_ (_) | | (_| |  |_ | (_) | | (_| |
//
  cl::sycl::buffer<cl::sycl::cl_int, 1> bufferA(A, global_range);
  myQueue.submit(std::bind( write ? f_write : f_discard_write, 
                            std::placeholders::_1, global_range, bufferA));
  }  // End of scope, wait for the queued work to stop.
 
  return 0;
}

