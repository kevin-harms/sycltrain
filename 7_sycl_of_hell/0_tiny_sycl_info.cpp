#include <CL/sycl.hpp>
#include <vector>

int main()
{

   //  _              _                      _
   // |_) |  _. _|_ _|_ _  ._ ._ _    ()    | \  _     o  _  _
   // |   | (_|  |_  | (_) |  | | |   (_X   |_/ (/_ \/ | (_ (/_
   //
   printf(">>> List Platform and device\n");

   std::vector<cl::sycl::platform> platforms = cl::sycl::platform::get_platforms();
   for(const auto& plat : platforms){
        // get_info is a template. So we pass the type as an `arguments`.
        std::cout << "Platform: " ;

        std::cout << plat.get_info<cl::sycl::info::platform::name>() << " ";
        std::cout << plat.get_info<cl::sycl::info::platform::vendor>() << " ";
        std::cout << plat.get_info<cl::sycl::info::platform::version>() << std::endl;
        // Trivia: how can we loop over argument?

        std::vector<cl::sycl::device> devices = plat.get_devices();
        for(const auto& dev : devices){
                    std::cout << "-- Device: " ; 
                    std::cout << dev.get_info<cl::sycl::info::device::name>() << " ";
                    std::cout << ( dev.is_gpu() ? "is a gpu" : " is not a gpu" ) << std::endl;
                   // cl::sycl::info::device::device_type exist, but do not overloead the << operator
        }
   }
} 
