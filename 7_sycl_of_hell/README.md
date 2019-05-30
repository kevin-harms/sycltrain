# How to compile
```
export MODULEPATH=/soft/restricted/intel_dga/modulefiles_sdk
module load dpcpp
make -j 8 
```

# List of program

- `tiny_sycl_info.cpp` show how to get information on platform and devices
- `my_first_kernel.cpp`  show how to create a queue, a command group 
- `parallel_for.cpp` How to use a parralle\_for and a range
- `nd_range.cpp`  show how to use a nd\_range. 
- `buffer.cpp`  How to do data-transfer
- `buffer_slm.cpp`  How to *not* do data-transfer
- `order.cpp`     How to handle dependency between kernel 
