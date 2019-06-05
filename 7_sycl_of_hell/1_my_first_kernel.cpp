#include <CL/sycl.hpp>

// Inspired by Codeplay compute cpp hello-world
int main() {
  // Selectors determine which device kernels will be dispatched to.
  cl::sycl::default_selector selector; 
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
  
  cl::sycl::queue myQueue(selector);
  std::cout << "Running on "
            << myQueue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";
//         __                     ___          
//  /\    (_  o ._ _  ._  |  _     |  _.  _ |  
// /--\   __) | | | | |_) | (/_    | (_| _> |< 
//                    |                        
                                    
  //Create a command_group to issue command to the group
  myQueue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task<class hello_world>([=]() {
        printf("Hello, World!\n");
    }); // End of the kernel function
  }); // End of the queue commands 
  printf("Before or After HW \n");
  }  // End of scope, wait for the queued work to stop. 
     // Can also use  myQueue.wait_and_throw();
  printf("For sure adter the Hello, World\n");
  return 0;
}
