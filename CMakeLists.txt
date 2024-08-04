cmake_minimum_required(VERSION 3.25)
project(HPC_project_2024 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(HPC_project_2024 main.cu)

set_target_properties(HPC_project_2024 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
