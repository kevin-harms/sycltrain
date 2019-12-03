# How to compile

```
make -j 7
```

# List of program

- `tiny_sycl_info.cpp` How to get information on platform and devices ( `./0_tiny_sycl_info`)
- `my_first_kernel.cpp`  How to create queues and command groups ( `./1_my_first_kernel`)
- `parallel_for.cpp` How to use a parralle\_for and range (`./2_parallel_for 8`)
- `buffer.cpp`  How to data-transfer (`./4_buffer 8 1`)
- `explicit_data_mouvement` How to use `copy`
- `error_handling` How to raise Error
- `buffer_usm` How to use one flavor of Unified shared memory 
- `simple dag` Demonstration of a DAG. Warning lot of C++
