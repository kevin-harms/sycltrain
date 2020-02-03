#include <CL/sycl.hpp>

// Inspired by Codeplay compute cpp hello-world
int main() {
  // Selectors determine which device kernels will be dispatched to.
 // Create your own or use `{cpu,gpu,accelerator}_selector`
  cl::sycl::default_selector selector; 
  
  // SYCL rely heavily on constructor / destructor semantics
  // e.g the destructor of the queue, will wait for all the kernel 
  // submited to this queue to terminate
  {
  
  cl::sycl::queue myQueue(selector);
  std::cout << "Running on "
            << myQueue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";
//         __                     ___          
//  /\    (_  o ._ _  ._  |  _     |  _.  _ |  
// /--\   __) | | | | |_) | (/_    | (_| _> |< 
//                    |                        
                                    
  //Create a command_group to issue command to the group.
  // Use A lambda to generate the control group handler
  myQueue.submit([&](cl::sycl::handler& cgh) {
    // Create a output stream
    cl::sycl::stream cout(1024, 256, cgh);

    // Submit a unique task, using a lambda
    cgh.single_task<class hello_world>([=]() {
    cout << "Hello, World!" << cl::sycl::endl;
    }); // End of the kernel function
  }); // End of the queue commands. The kernel is now submited 
  }  // End of scopes, the queue will be destroyed, trigering a synchronization
  return 0;
}
