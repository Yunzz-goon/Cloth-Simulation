# CMake build system for cloth simulation
cmake_minimum_required(VERSION 3.12)
project(cloth VERSION 1.0.0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# ##############################################################################
# enable C++  ##
# ##############################################################################
enable_language(CXX)

# ##############################################################################
# setup general C++ compilation flags#
# ##############################################################################

# add explictly debug symbols 
set(CMAKE_CXX_FLAGS " -g -msse -mavx -mavx512f")
# add g++ specific flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS}  -fpic -Wall -Wextra -Wno-deprecated-declarations -Wpedantic"
  )
endif()

# add icc specific flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout -march=native"
  )
endif()

find_program(PAPI papi_avail)
if(PAPI)
    message("-- Found PAPI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpapi -DPAPI")
else()
    message("-- PAPI not found.")
endif()

# OpenGL
find_package(OpenGL REQUIRED)
message(STATUS "OpenGL included at ${OPENGL_INCLUDE_DIR}")

# GLUT
find_package(GLUT REQUIRED)
message(STATUS "GLUT included at ${GLUT_INCLUDE_DIR}")

# OpenMP
find_package(OpenMP)
message(STATUS "OpenMP included at ${OPENMP_INCLUDE_DIR}")

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

add_executable(
  opengl_main
  opengl_main.cpp
  cloth_code.cpp
  simple_papi.cpp)

add_executable(
  kernel_main
  kernel_main.cpp
  cloth_code.cpp
  simple_papi.cpp)

add_executable(
  kernel_opt
  kernel_main.cpp
  cloth_code_opt.cpp
  simple_papi.cpp)

add_executable(
  kernel_sse
  kernel_main.cpp
  cloth_code_sse.cpp
  simple_papi.cpp)

add_executable(
  kernel_vect_omp
  kernel_main.cpp
  cloth_code_vect_omp.cpp
  simple_papi.cpp)

add_executable(
  kernel_omp
  kernel_omp.cpp
  cloth_code_omp.cpp
  simple_papi.cpp)

add_executable(
  kernel_block
  kernel_omp.cpp
  cloth_code_omp_block.cpp
  simple_papi.cpp)

if(OpenMP_FOUND)
  target_link_libraries(kernel_main m OpenMP::OpenMP_CXX)
  target_link_libraries(kernel_opt m OpenMP::OpenMP_CXX)
  target_link_libraries(kernel_sse m OpenMP::OpenMP_CXX)
  target_link_libraries(kernel_vect_omp m OpenMP::OpenMP_CXX)
  target_link_libraries(kernel_omp m OpenMP::OpenMP_CXX)
  target_link_libraries(kernel_block m OpenMP::OpenMP_CXX)
  target_link_libraries(opengl_main m ${OPENGL_LIBRARIES} ${GLUT_LIBRARY}
                        OpenMP::OpenMP_CXX)
else()
  target_link_libraries(kernel_main m)
  target_link_libraries(opengl_main m ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
endif()
