#include <CL/sycl.hpp>

// Inspired by Codeplay compute cpp hello-world
int main(int argc, char** argv) {
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

    /* - - - -
    Range
    - - - - */
    // Range configuration...
    const auto global_range =  (size_t) atoi(argv[1]);
    // Nd range allow use to access information
    cgh.parallel_for<class hello_world>(cl::sycl::range<1>(global_range), 
                                        [=](cl::sycl::id<1> idx) {
       printf("Hello world: World rank %zu \n", idx[0]);

    }); // End of the kernel function
  }); // End of the queue commands 
  }  // End of scope, wait for the queued work to stop. 
     // Can also use  myQueue.wait_and_throw();
  return 0;
}
