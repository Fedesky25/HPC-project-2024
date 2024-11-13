#include <iostream>
#include <cstdio>
#include "utils.cuh"
#include "cli.cuh"
#include "getopt.h"
#include "tiles.cuh"
#include "particle_generator.cuh"
#include "complex_functions.cuh"
#include "canvas.cuh"
#include "evolution.cuh"
#include "video.cuh"
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

    auto output_filepath_len = strlen(config.output);
    auto raw_output = output_filepath_len > 4 && 0 == strcmp(config.output+output_filepath_len-4, ".raw");
    const char * raw_output_file = config.output;
    if(!raw_output) {
        auto raw_output_m = new char [output_filepath_len + 5];
        strcpy(raw_output_m, config.output);
        strcpy(raw_output_m+output_filepath_len, ".raw");
        raw_output_m[output_filepath_len+4] = '\0';
        raw_output_file = raw_output_m;
    }

    if(verbose) {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Output file: " << config.output;
        if(!raw_output) std::cout << " (raw: " << raw_output_file << ')';
        std::cout << std::endl;
        std::cout << "  Complex numbers: " << config.vars.z[0] << ' ' << config.vars.z[1] << ' ' << config.vars.z[2] << std::endl;
        std::cout << "  Real and int numbers: " << config.vars.x << ", " << config.vars.n << std::endl;
        std::cout << "  Canvas: " << config.canvas << std::endl;
        std::cout << "  Evolution: " << config.evolution << std::endl;
    }

    auto start_computation = std::chrono::steady_clock::now();

    complex_t min, max;
    config.bounds(&min, &max);
    uint64_t N = config.particle_number();

    complex_t * points;
    auto frame_size = config.canvas.height * config.canvas.width;

    switch (config.mode) {
        case ExecutionMode::Serial:
        {
            points = particles_serial(min, max, N, config.lloyd_iterations);
            auto canvas = new CanvasPixel [frame_size];
            evolve_serial(&config, canvas, points, N, fn_choice);
            write_video_serial(raw_output_file, canvas, frame_size, config.evolution.frame_count, config.evolution.life_time, config.background);
            delete[] canvas;
            break;
        }
        case ExecutionMode::OpenMP:
        {
            points = particles_omp(min, max, N, config.lloyd_iterations);
            auto canvas_count = omp_get_max_threads();
            auto canvases = create_canvas_host(canvas_count, &config.canvas);
            evolve_omp(&config, canvases, points, N, fn_choice);
            write_video_omp(
                    raw_output_file, canvases, canvas_count, frame_size,
                    config.evolution.frame_count, config.evolution.life_time, config.background);
            break;
        }
        case ExecutionMode::GPU:
        {

            int gpu_count;
            cudaGetDeviceCount(&gpu_count);
            if(gpu_count < 1) {
                std::cerr << "No CUDA capable devices were detected: change parallelization type" << std::endl;
                return 1;
            }
            else if(gpu_count > 1) cudaSetDevice(0);

            Tiles tiles(&config);
            unsigned tiles_count = tiles.total();
            if(verbose) std::cout << "  Tiles: " << tiles.rows << 'x' << tiles.cols << '=' << tiles_count << " with "
                                  << (float) N / (float) tiles_count << " particles each" << std::endl;
            points = particles_gpu(min, max, N, config.lloyd_iterations);
            auto tile_offsets = tiles.sort(min, max, points, N);
            auto canvas_count = get_canvas_count_serial(tile_offsets, tiles_count);
            auto canvases = create_canvas_device(canvas_count, &config.canvas);
            evolve_gpu(&config, canvases, canvas_count, points, N,
                       tile_offsets, tiles_count, fn_choice);
            cudaFree(tile_offsets);
            cudaFree(points);
            write_video_gpu(
                    config.output, canvases, canvas_count, frame_size,
                    config.evolution.frame_count, config.evolution.life_time, &config.background);
            break;
        }
    }

    auto end_computation = std::chrono::steady_clock::now();
    float time_all = (std::chrono::duration<float,std::ratio<1>>(end_computation-start_computation)).count();
    std::cout << "All computations completed in " << time_all << 's' << std::endl;

    if(raw_output) {
        std::cout << "Run the command:  ffmpeg -f rawvideo -pixel_format rgb"
                  << ((config.background.A == 1.0f) ? "24" : "a")
                  << " -video_size " << config.canvas.width << 'x' << config.canvas.height
                  << " -framerate " << config.evolution.frame_rate
                  << " -i " << config.output << " <output>" << std::endl;
    }
    else {
        auto command = new char [100 + 2*output_filepath_len];
        strcpy(command, "ffmpeg -f rawvideo -pixel_format rgb");
        strcpy(command+36, (config.background.A == 1.0f) ? "24" : "a ");
        sprintf(command+38, " -video_size %dx%d -framerate %d -i %s %s",
                config.canvas.width, config.canvas.height, config.evolution.frame_rate,
                raw_output_file, config.output);
        std::cout << "Running: " << command << std::endl;
        std::cout.flush();
        auto error = system(command);
        if(error) std::cout << "Errors occurred: leaving '" << raw_output_file << "' behind" << std::endl;
        else std::remove(raw_output_file);
    }

    #if 0
    std::ofstream test("test.raw");
    auto frame_buffer = (uint32_t*) malloc(frame_mem);
    auto rgba_bg = config.background.toRGBA();
    for(uint32_t i=0; i<frame_size; i++) frame_buffer[i] = rgba_bg;
    for(unsigned i=0; i<10; i++) test.write(reinterpret_cast<const char *>(frame_buffer), frame_mem);

    uint32_t color, offset;
    for(unsigned c=0; c<10; c++) {
        offset = config.canvas.width*25*(c+1) - 25;
        color = HSLA_to_RGBA(icenc((float)c*0.1f), icenc(0.5), icenc(0.5), icenc(1));
        for(uint32_t i=50; i<config.canvas.width; i++) frame_buffer[offset+i] = color;
        for(unsigned i=0; i<10; i++) test.write(reinterpret_cast<const char *>(frame_buffer), frame_mem);
    }
    test.close();
    #endif

    return 0;
}
