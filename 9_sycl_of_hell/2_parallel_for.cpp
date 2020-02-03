#include "cxxopts.hpp"
#include <CL/sycl.hpp>

int main(int argc, char** argv) {

//  _                ___                
// |_) _. ._ _  _     |  ._  ._     _|_ 
// |  (_| | _> (/_   _|_ | | |_) |_| |_ 
//                           |          

  cxxopts::Options options("2_parallel_for", " How to use 'parallel_for' and 'range' ");

  options.add_options()
   ("h,help", "Print help")
   ("g,grange", "Global range", cxxopts::value<int>() ->default_value("1"))
  ;

 auto result = options.parse(argc, argv);

 if (result.count("help"))
    {
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }

  const auto global_range= result["grange"].as<int>();

//  _                             _
// |_) _. ._ ._ _. | | |  _  |   |_ _  ._
// |  (_| |  | (_| | | | (/_ |   | (_) |


  // Selectors determine which device kernels will be dispatched to.
  cl::sycl::default_selector selector; 
  {
  
  cl::sycl::queue myQueue(selector);
  std::cout << "Running on "
            << myQueue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";
  
  //Create a command_group to issue command to the group
  myQueue.submit([&](cl::sycl::handler& cgh) {

    cl::sycl::stream cout(1024, 256, cgh);
    //
    // #pragma omp parallel for
    // for(int idx[0]=0; idx[0]++; idx[0]< global_range)
    cgh.parallel_for<class hello_world>(cl::sycl::range<1>(global_range), 
                                        [=](cl::sycl::id<1> idx) {
       cout << "Hello, World: World rank " << idx << cl::sycl::endl;
    }); // End of the kernel function
  }); // End of the queue commands 
  }  // End of scope, wait for the queued work to stop. 
  return 0;
}
