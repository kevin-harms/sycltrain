#include <CL/sycl.hpp>

// Inspired by Codeplay compute cpp hello-world
int main(int argc, char** argv) {

   const auto global_range =  (size_t) atoi(argv[1]);
   const auto local_range =  (size_t) atoi(argv[2]);

   // Crrate array
  // Selectors determine which device kernels will be dispatched to.
  cl::sycl::default_selector selector; 
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
  cl::sycl::queue myQueue(selector);
  std::cout << "Running on "
            << myQueue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";

  //Create a command_group to issue command to the group
  myQueue.submit([&](cl::sycl::handler& cgh) {

     //Create an accesor for the sycl buffer. Trust me, use auto.
    // auto accessorA = bufferA.get_access<cl::sycl::access::mode::discard_write>(cl::sycl::range<1>(global_range),cgh);
    auto slm = cl::sycl::accessor<int, 1,  cl::sycl::access::mode::read_write, cl::sycl::access::target::local>(cl::sycl::range<1>(global_range), cgh);

    // Nd range allow use to access information
    cgh.parallel_for<class hello_world>(cl::sycl::nd_range<1>{cl::sycl::range<1>(global_range), 
                                                             cl::sycl::range<1>(local_range) }, 
                                        [=](cl::sycl::nd_item<1> idx) {
      const int world_rank = idx.get_global_id(0);
      slm[world_rank] = world_rank;
      //So compiler doesn't optimize away
      printf( "slm rank: %d \n", slm[world_rank % INT_MAX ]);
    }); // End of the kernel function
  }); // End of the queue commands
  }  // End of scope, wait for the queued work to stop. 
     // Can also use  myQueue.wait_and_throw();

  return 0;
}
