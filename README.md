# HPC-project-2024

This is the final project for the 2024 course of "High Performance Computing" of the Master degree "Quantum Engineering" at Politecnico di Torino.

Written in CUDA, the end result should be a CLI tool. Given a complex function (among a relative wide range of possibilities) and, optionally, the values of some options (width, height,fps, center, function parameters, etc.), such tool produces a video of the stream plot of the chosen function.

The general program flow is the following
1. Initial particle positions are generated using [Lloyd's algorithm](https://en.m.wikipedia.org/wiki/Lloyd's_algorithm)
2. The evolution of the particles is written on multiple canvases in order to parallelize writing on different OpenMP threads or CUDA thread blocks
3. For each frame, the same pixel of each canvas are combined into a single pixel which is drawn into the frame buffer
4. Frames are encoded into a mp4 file using ffmpeg

## Building from source

1. Set the env variable `FFMPEG_PATH` to the path to the root library of ffmpeg (without trailing slash)
2. Be sure to have `CUDA` and `CMake` available on your machine (e.g. by running `module load cmake nvidia/cudasdk`)
3. On Windows, you must use the MSVC compiler
4. Create the build folder
5. Run the following
    ```bash
    cmake -DCMAKE_BUILD_TYPE=Release -S <source-directory> -B <build-directory>
    cmake --build <build-directory> --target HPC_project_2024 -j
    ```
6. In case `nvcc` does not recognize the option `-arch=native` or does not detect a CUDA capable device, please specify the environment variable `GPU_ARCHITECTURE` with the name of the [GPU architecture](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-architecture-arch) you want to compile for (e.g. `export GPU_ARCHITECTURE=sm_75` for [compute capability](https://developer.nvidia.com/cuda-gpus) 7.5)

## Usage 

Full synopsis of the command, as well as all the available flags, can be seen by running the command without arguments.

### Basic usage
```bash
HPC_project_2024 -R <resolution> -D <video-duration-seconds> -p <parallelization> -d <particle-distance> -L <lloyd-iterations>  <function-name>
```

### Particle load/save

You can avoid recomputing the particle position through Lloyd algorithm by saving and then loading the particles to and from a file.
```bash
# save particles into particles.txt after 5 iterations of Lloyd algorithm
HPC_project_2024 -L 5:particles.txt <other-flags> <function-name>

# save particles into other_particles.txt after 8 iterations of Lloyd algorithm, and produce no video
HPC_project_2024 -D 0 -L 8:other_particles.txt <other-flags>

# load particles later
HPC_project_2024 -L particle.txt <other-flags> <function-name>
```
To get good results you should make sure that the screen resolution (`-R`), space scale (`-s`), and center point (`-c`) of the command runs saving and loading particles match. However, this is not enforced. 


## Library used

This tool makes use of two libraries:
- `getopt` &mdash; Preinstalled on Linux. When compiling for Windows, a port of `getopt` (taken from [Chunde's repository](https://github.com/Chunde/getopt-for-windows)) is injected.
- `ffmpeg` &mdash; Once installed, the environment variable `FFMPEG_PATH` must be specified with the path to the installation directory of `ffmpeg`. When building for Windows, be sure to add the `${FFMPEG_PATH}/bin` directory in the PATH, so that dll can be used by the executable.

## Debugging tools

```bash
ncu -o <out_file> <command>
compute-sanitizer --tool memcheck --log-file <out_file> <command>
```