cmake_minimum_required(VERSION 3.14)
project(HPC_project_2024 LANGUAGES CXX CUDA C)

message("-- CMake version ${CMAKE_VERSION}")

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


# cuda standard 17 support introduced in 3.18
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_STANDARD.html
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#std-c-03-c-11-c-14-c-17-c-20-std

set(CMAKE_CXX_STANDARD 17)
if (CMAKE_VERSION VERSION_LESS 3.18)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")
else ()
    set(CMAKE_CUDA_STANDARD 17)
endif ()

# 'native' option for cuda architectures support introduced in 3.24
# however, it does not work
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_STANDARD.html
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=native -Xcompiler -openmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=native")

# https://stackoverflow.com/questions/9313585/specifying-openmp-cuda-flag-in-a-cmake-project
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_executable(HPC_project_2024
        main.cu
        include/utils.cuh src/utils.cu
        include/cli.cuh src/cli.cu
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

include_directories(include)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    message("Windows detected -> injecting getopt port")
    target_sources(HPC_project_2024 PRIVATE getopt-win/getopt.h getopt-win/getopt.c)
    target_include_directories(HPC_project_2024 PRIVATE getopt-win)
endif()

set_target_properties(HPC_project_2024 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#target_compile_options(HPC_project_2024 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)