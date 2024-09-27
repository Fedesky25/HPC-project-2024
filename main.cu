#include <iostream>
#include <iomanip>
#include "utils.cuh"
#include "cli.hpp"
#include "getopt.h"
#include "tiles.cuh"
#include "particle_generator.cuh"
#include "fstream"
#include "complex_functions.cuh"
#include "canvas.cuh"
#include "evolution.cuh"
#include "frames.cuh"
#include "omp.h"


int main(int argc, char * argv[]) {
    if(argc < 2) {
        print_usage();
        return 1;
    }

    Configuration config;
    auto err = parse_args(argc, argv, &config);
    if(err) return 1;

    if(optind >= argc) {
        std::cerr << "Missing function to plot" << std::endl;
        return 1;
    }
    auto fn_choice = strtofn(argv[optind]);
    if(fn_choice == FunctionChoice::NONE) {
        std::cerr << "Function string name not recognized" << std::endl;
        return 1;
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Output file: " << config.output << std::endl;
    std::cout << "  Complex numbers: " << config.vars.z[0] << ' ' << config.vars.z[1] << ' ' << config.vars.z[2] << std::endl;
    std::cout << "  Real and int numbers: " << config.vars.x << ", " << config.vars.n << std::endl;
    std::cout << "  Canvas: " << config.canvas << std::endl;
    std::cout << "  Evolution: " << config.evolution << std::endl;

    auto start_computation = std::chrono::steady_clock::now();

    complex_t min, max;
    config.bounds(&min, &max);
    uint64_t N = config.particle_number();

    complex_t * points;
    uint32_t canvas_count;
    auto frame_size = config.canvas.height * config.canvas.width;
    auto frame_mem = frame_size * sizeof(uint32_t);
    auto signed_fc = (int32_t) config.evolution.frame_count;

    switch (config.mode) {
        case ExecutionMode::Serial:
            points = particles_serial(min, max, N);
            break;
        case ExecutionMode::OpenMP:
        {
            points = particles_omp(min, max, N);
            canvas_count = omp_get_max_threads();
            auto canvases = create_canvas_host(canvas_count, &config.canvas);
            evolve_omp(&config, canvases, points, N, fn_choice);
            break;
        }
        case ExecutionMode::GPU:
        {
            Tiles tiles(&config);
            unsigned tiles_count = tiles.total();
            std::cout << "  Tiles: " << tiles.rows << 'x' << tiles.cols << '=' << tiles_count << " with "
                      << (float) N / (float) tiles_count << " particles each" << std::endl;
            points = particles_gpu(min, max, N);
            auto tile_offsets = tiles.sort(min, max, points, N);
            canvas_count = get_canvas_count_serial(tile_offsets, tiles_count);
            auto canvases = create_canvas_device(canvas_count, &config.canvas);
            evolve_gpu(&config, canvases, canvas_count, points, N,
                       tile_offsets, tiles_count, fn_choice);
            cudaFree(tile_offsets);
            cudaFree(points);

            std::ofstream raw_output(config.output);
            uint32_t *h_frame, *d_frame[2];
            h_frame = (uint32_t*) malloc(frame_mem);
            cudaMalloc(d_frame, frame_mem);
            cudaMalloc(d_frame+1, frame_mem);
            std::cout << "Frame buffers: CPU=" << (((frame_mem-1)>>20)+1) << "MB, GPU="
                      << (((frame_mem*2-1)>>20)+1) << "MB" << std::endl << std::fixed;
            std::cout << "Frame computation: iter. | c (us) | w (ms)" << std::endl;
            std::cout.width(6);

            float time_write, time_compute;
            auto begin = std::chrono::steady_clock::now();
            compute_frame_gpu(
                    0, signed_fc,
                    canvases, canvas_count,
                    d_frame[0], frame_size,
                    &config.background);
            cudaDeviceSynchronize();
            auto _end = std::chrono::steady_clock::now();
            time_compute = (std::chrono::duration<float,std::micro>(_end-begin)).count();
            std::cout << "                   " << std::setw(5) << 0
                      << " | " << std::setw(6) << time_compute
                      << " | " << std::endl;


            for(int32_t i=1; i<signed_fc; i++) {
                #pragma omp parallel sections num_threads(2)
                {
                    #pragma omp section
                    {
                        auto start = std::chrono::steady_clock::now();
                        cudaMemcpy(h_frame, d_frame[(i&1)^1], frame_mem, cudaMemcpyDeviceToHost);
                        raw_output.write(reinterpret_cast<const char *>(h_frame), frame_mem);
                        auto end = std::chrono::steady_clock::now();
                        time_write = (std::chrono::duration<float, std::milli>(end - start)).count();
                    }
                    #pragma omp section
                    {
                        auto start = std::chrono::steady_clock::now();
                        compute_frame_gpu(i, signed_fc, canvases, canvas_count, d_frame[i&1], frame_size, &config.background);
                        cudaDeviceSynchronize();
                        auto end = std::chrono::steady_clock::now();
                        time_compute = (std::chrono::duration<float,std::micro>(end-start)).count();
                    }
                }
                std::cout << "                   " << std::setw(5) << i
                          << " | " << std::setw(6) << time_compute
                          << " | " << std::setw(6) << time_write << std::endl;
            }

            begin = std::chrono::steady_clock::now();
            cudaMemcpy(h_frame, d_frame[(signed_fc-1)&1], frame_mem, cudaMemcpyDeviceToHost);
            raw_output.write(reinterpret_cast<const char *>(h_frame), frame_mem);
            _end = std::chrono::steady_clock::now();
            time_write = (std::chrono::duration<float,std::milli>(_end-begin)).count();
            std::cout << "                   " << std::setw(5) << signed_fc
                      << " |        | " << std::setw(6) << time_write << std::endl;
            break;
        }
    }

    auto end_computation = std::chrono::steady_clock::now();
    float time_all = (std::chrono::duration<float,std::ratio<1>>(end_computation-start_computation)).count();
    std::cout << "All computations completed in " << time_all << 's' << std::endl;
    std::cout << "Run the command:  ffmpeg -f rawvideo -pixel_format rgba -video_size "
              << config.canvas.width << 'x' << config.canvas.height << " -framerate "
              << config.evolution.frame_rate << " -i " << config.output << " <output>" << std::endl;

    return 0;
}
