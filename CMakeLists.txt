cmake_minimum_required(VERSION 3.25)
project(HPC_project_2024 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

#find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -openmp")

include_directories(include)

add_executable(HPC_project_2024
        main.cu
        include/utils.cuh src/utils.cu
        include/cli.hpp src/cli.cpp
        include/particle_generator.cuh src/particle_generator.cu
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(HPC_project_2024 PRIVATE DEBUG)
endif()

set_target_properties(HPC_project_2024 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

#target_compile_options(HPC_project_2024 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)