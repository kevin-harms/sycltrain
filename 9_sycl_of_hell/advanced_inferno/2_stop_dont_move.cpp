#include <iostream>
#include <vector>
#include <CL/sycl.hpp>

template <typename TAccessorRW, typename TAccessorR>
class memcopy_kernel{
 public:
   memcopy_kernel(TAccessorRW accessorRW_, TAccessorR accessorR_): accessorRW{accessorRW_},  accessorR{accessorR_}  {}
   void operator()(cl::sycl::item<1> idx) { accessorRW[idx.get_id()] += accessorR[idx.get_id()]; }
 private:
   TAccessorRW accessorRW;
   TAccessorR accessorR;
};

void f_copy(cl::sycl::handler& cgh, int global_range, cl::sycl::buffer<cl::sycl::cl_int, 1> bufferRW, 
                                                      cl::sycl::buffer<cl::sycl::cl_int, 1> bufferR) {
    auto accessorRW = bufferRW.get_access<cl::sycl::access::mode::read_write>(cgh);
    auto accessorR = bufferR.get_access<cl::sycl::access::mode::read>(cgh);
    cgh.parallel_for(cl::sycl::range<1>(global_range), memcopy_kernel(accessorRW, accessorR));
}



int main(int argc, char** argv) {

   // B += A
   const auto global_range =  (size_t) atoi(argv[1]);
   const auto num_iter = (size_t) atoi(argv[2]);
   // Create array
   std::vector<int> A(global_range, 1);
   std::vector<int> B(global_range, 0);
   
  // Selectors determine which device kernels will be dispatched to.
  cl::sycl::default_selector selector;
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
  cl::sycl::queue myQueue(selector);

  // Create buffer. 
  cl::sycl::buffer<cl::sycl::cl_int, 1> bufferA(A.data(), global_range);
  cl::sycl::buffer<cl::sycl::cl_int, 1> bufferB(B.data(), global_range);

  std::cout << "Running on "
            << myQueue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";

  // We should expect only one copy of B. 
  for (size_t i = 0; i < num_iter ; i++) {
    myQueue.submit(std::bind(f_copy, std::placeholders::_1, global_range, bufferB, bufferA));
  }

  }  // End of scope, wait for the queued work to stop.
 
 for (size_t i = 0; i < global_range; i++)
        std::cout<< "B[ " << i <<" ] = " << B[i] << std::endl;
  return 0;
}

