The module you need:

```
module load papi
module load intel-compiler
```

There are 5 versions of codes in the project

- kernel_main
- kernel_opt
- kernel_sse
- kernel_vect_omp
- kernel_omp
- kernel_block

## Verify the correctness

If you want to test whether the result produced by the code is correct, you have to

```
python3 auto_test.py
```

Notice we didn't provide a direct way to test kernel_block, however, it can be done by edit little in CMakeLists.txt:

1. Comment the codes below(line 87-91)

   ```
   add_executable(
     kernel_omp
     kernel_omp.cpp
     cloth_code_omp.cpp
     simple_papi.cpp)
   ```

2. modify the line 93-97 to:

   ```
   add_executable(
     kernel_omp
     kernel_omp.cpp
     cloth_code_omp_block.cpp
     simple_papi.cpp)
   ```

3. Comment the following code

   ```
   target_link_libraries(kernel_block m OpenMP::OpenMP_CXX)
   ```

4. command: python3 auto_test.py

## Collect execution time

To execute them, you have to(take kernel_block as example)

```
cd build
cmake ..
make
time ./kernel_block -p xx
```

It is noticed that only kernel_block and kernel_omp has the flag of the number of thread -p. 



