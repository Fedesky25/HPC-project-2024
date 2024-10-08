cmake_minimum_required(VERSION 3.14)
project(HPC_project_2024 LANGUAGES CXX CUDA C)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    # https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
    find_package(CUDAToolkit)
    link_libraries(CUDA::curand)
else ()
    # deprecated after 3.17 but required to use cuda libraries in older version
    # https://cmake.org/cmake/help/latest/module/FindCUDA.html
    find_package(CUDA REQUIRED)
    link_libraries(${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
    message("-- Found CUDA v${CUDA_VERSION} (compiler v${CMAKE_CUDA_COMPILER_VERSION})")
endif ()

find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=native -Xcompiler -openmp")

include_directories(include)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
#message("CUDA TOOLKIT = ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

add_executable(HPC_project_2024
        main.cu
        include/utils.cuh src/utils.cu
        include/cli.hpp src/cli.cpp
        include/lower_bound.cuh
        include/particle_generator.cuh src/particle_generator.cu
        include/tiles.cuh src/tiles.cu
        src/canvas.cu include/canvas.cuh
        include/evolution.cuh src/evolution.cu
        include/complex_functions.cuh src/complex_functions.cu
        include/frames.cuh src/frames.cu
        include/color.cuh src/color.cu
        include/video.cuh src/video.cu
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(HPC_project_2024 PRIVATE DEBUG)
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    message("Windows detected -> injecting getopt port")
    target_sources(HPC_project_2024 PRIVATE getopt-win/getopt.h getopt-win/getopt.c)
    target_include_directories(HPC_project_2024 PRIVATE getopt-win)
endif()

set_target_properties(HPC_project_2024 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# available only after 3.23
# set_target_properties(HPC_project_2024 PROPERTIES CUDA_ARCHITECTURES all-major)

#target_compile_options(HPC_project_2024 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)