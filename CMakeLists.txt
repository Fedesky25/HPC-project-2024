cmake_minimum_required(VERSION 3.14)
project(HPC_project_2024 LANGUAGES CUDA CXX C)

message("-- CMake version ${CMAKE_VERSION}")

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    # https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
    find_package(CUDAToolkit)
    link_libraries(CUDA::curand)
    message("-- CUDA toolkit ${CUDAToolkit_VERSION}")
else ()
    # deprecated after 3.17 but required to use cuda libraries in older version
    # https://cmake.org/cmake/help/latest/module/FindCUDA.html
    find_package(CUDA REQUIRED)
    message("-- CUDA toolkit ${CUDA_VERSION}")
    # link_libraries(${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
endif ()
message("-- CUDA compiler ${CMAKE_CUDA_COMPILER_VERSION}")
link_libraries(-lcurand_static -lculibos)


# cuda standard 17 support introduced in cmake 3.18 & cuda 11
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_STANDARD.html
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#std-c-03-c-11-c-14-c-17-c-20-std

set(CMAKE_CXX_STANDARD 17)
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
    set(CMAKE_CXX_STANDARD 14)
endif ()
if (CMAKE_VERSION VERSION_LESS 3.18)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CMAKE_CXX_STANDARD}")
else ()
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
endif ()
message("-- CXX/CUDA standard set to ${CMAKE_CXX_STANDARD}")


# retrieve base path of ffmpeg installation from env variable
if (DEFINED ENV{FFMPEG_PATH})
    set(FFMPEG_INCLUDE_DIR "$ENV{FFMPEG_PATH}/include/")
    set(FFMPEG_LIBRARY_DIR "$ENV{FFMPEG_PATH}/lib/")
else ()
    message(FATAL_ERROR "FFMPEG_PATH environment variable not specified!")
endif ()


# Include libraries and header files for libavcodec, libavformat, libavutils
# https://stackoverflow.com/questions/50760024/cmake-configuration-for-ffmpeg-in-c-project
find_library(AVFORMAT_LIBRARY avformat PATHS ${FFMPEG_LIBRARY_DIR} REQUIRED)
find_library(AVCODEC_LIBRARY  avcodec  PATHS ${FFMPEG_LIBRARY_DIR} REQUIRED)
find_library(AVUTIL_LIBRARY   avutil   PATHS ${FFMPEG_LIBRARY_DIR} REQUIRED)

# https://stackoverflow.com/questions/9313585/specifying-openmp-cuda-flag-in-a-cmake-project
# If found, OpenMP_CXX_FLAGS is usually "-fopenmp" -> pass this value to -Xcompiler
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


add_executable(HPC_project_2024
        main.cu
        include/utils.cuh src/utils.cu
        include/config.cuh src/config.cu
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
        include/sorter.cuh
)

target_include_directories(HPC_project_2024 PRIVATE include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${FFMPEG_INCLUDE_DIR})
target_link_libraries(HPC_project_2024 PRIVATE OpenMP::OpenMP_CXX ${AVFORMAT_LIBRARY} ${AVCODEC_LIBRARY} ${AVUTIL_LIBRARY})

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    message("-- Windows detected -> injecting getopt port")
    target_sources(HPC_project_2024 PRIVATE getopt-win/getopt.h getopt-win/getopt.c)
    target_include_directories(HPC_project_2024 PRIVATE getopt-win)
endif()

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_STANDARD.html
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation
if (DEFINED ENV{GPU_ARCHITECTURE})
    set(SELECTED_GPU_ARCH "$ENV{GPU_ARCHITECTURE}")
    message("-- GPU architecture variable detected: ${SELECTED_GPU_ARCH}")
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        set_property(TARGET HPC_project_2024 PROPERTY CUDA_ARCHITECTURES  "${SELECTED_GPU_ARCH}")
        target_compile_options(HPC_project_2024 PRIVATE -Wno-deprecated-gpu-targets)
    else ()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${SELECTED_GPU_ARCH} -Wno-deprecated-gpu-targets")
    endif ()
else ()
    message("-- GPU architecture not specified -> try to use native one")
    # 'native' option for cuda architectures support introduced in 3.24 which however does not work
    # to disable warnings, we must disable CUDA_ARCHITECTURES as it cannot be empty
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        set_property(TARGET HPC_project_2024 PROPERTY CUDA_ARCHITECTURES OFF)
    endif ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=native")
endif ()

set_target_properties(HPC_project_2024 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(HPC_project_2024 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
#	--verbose
#        --use_fast_math
#        --relocatable-device-code=true
	-Xcompiler "${OpenMP_CXX_FLAGS}"
	-Xptxas -v
	-Xptxas -maxrregcount=64
>)

