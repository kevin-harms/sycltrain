#include <CL/sycl.hpp>

class hello_world_kernel {
 public:
  hello_world_kernel(int uuid_): uuid{uuid_} {}
 
  void operator()(cl::sycl::item<1> idx) {
    printf("Hello world form kernel %d: World rank %zu \n", uuid, idx.get_id(0));
  }
  
  private:
    int uuid;
};

void hello_world(cl::sycl::handler& cgh, int uuid, int global_range) {
    auto k = hello_world_kernel(uuid);
    cgh.parallel_for(cl::sycl::range<1>(global_range),k);
}

int main(int argc, char** argv) {

  const auto global_range1 =  (size_t) atoi(argv[1]);
  const auto global_range2 =  (size_t) atoi(argv[2]);

  // Selectors determine which device kernels will be dispatched to.
  cl::sycl::gpu_selector selector;
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
  cl::sycl::queue myQueue(selector);
  std::cout << "Running on "
            << myQueue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";

//  __                     ___             
// (_  o ._ _  ._  |  _     |  _.  _ |   _ 
// __) | | | | |_) | (/_    | (_| _> |< _> 
//             |                           

  myQueue.submit(std::bind(hello_world,std::placeholders::_1, 0, global_range1));
  myQueue.submit(std::bind(hello_world,std::placeholders::_1, 1, global_range2));
  }  // End of scope, wait for the queued work to stop.
  return 0;
}

