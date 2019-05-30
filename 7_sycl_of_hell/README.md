# How to compile
```
export MODULEPATH=/soft/restricted/intel_dga/modulefiles_sdk
module load dpcpp
make -j 8 
```

# List of program

- `tiny_sycl_info.cpp` show how to get information on platform and devices ( `./0_tiny_sycl_info`)
- `my_first_kernel.cpp`  show how to create a queue, a command group ( `./1_my_first_kernel`)
- `parallel_for.cpp` How to use a parralle\_for and a range (`./2_parallel_for 8`)
- `nd_range.cpp`  show how to use a nd\_range. (`./3_nd_range 8 1`)
- `buffer.cpp`  How to do data-transfer (`./4_buffer 8 1`)
- `buffer_slm.cpp`  How to *not* do data-transfer (`./5_buffer_slm 8 1`)
- `order.cpp`     How to handle dependency between kernel  (`./6_order 8 0`)
