# How to compile
```
export MODULEPATH=/soft/restricted/intel_dga/modulefiles_sdk
module load dpcpp
make -j 8 
```

# List of program

- `tiny_sycl_info.cpp` How to get information on platform and devices ( `./0_tiny_sycl_info`)
- `my_first_kernel.cpp`  How to create queues and command groups ( `./1_my_first_kernel`)
- `parallel_for.cpp` How to use a parralle\_for and range (`./2_parallel_for 8`)
- `nd_range.cpp`   how to use a nd\_rang. (`./3_nd_range 8 1`)
- `buffer.cpp`  How to data-transfer (`./4_buffer 8 1`)
- `buffer_slm.cpp`  How to *not* data-transfer (`./5_buffer_slm 8 1`)
- `order.cpp`     How to handle dependences (`./6_order 8 0`)
