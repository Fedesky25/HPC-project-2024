cmake_minimum_required(VERSION 3.25)
project(HPC_project_2024 CUDA)

set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

add_executable(HPC_project_2024 main.cu include/utils.cuh src/utils.cu include/cli.cuh src/cli.cu
        include/particle_generator.cuh
        src/particle_generator.cu
)

set_target_properties(HPC_project_2024 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
