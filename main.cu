#include "config.cuh"
#include "cli.cuh"
#include "getopt.h"
#include "tiles.cuh"
#include "particle_generator.cuh"
#include "complex_functions.cuh"
#include "canvas.cuh"
#include "evolution.cuh"
#include "frames.cuh"
#include "video.cuh"
#include "omp.h"


int main(int argc, char * argv[]) {
    if(argc < 2) {
        print_usage();
        return 1;
    }

    Configuration config;
    auto error = parse_args(argc, argv, &config);
    if(error) return 1;

    EXIT_IF(optind >= argc, "Missing function to plot")
    auto fn_choice = strtofn(argv[optind]);
    EXIT_IF(fn_choice == FunctionChoice::NONE, "Function string name not recognized")

    if(verbose) {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Output file: " << config.output << std::endl;
        std::cout << "  Complex numbers: " << config.vars.z[0] << ' ' << config.vars.z[1] << ' ' << config.vars.z[2] << std::endl;
        std::cout << "  Real and int numbers: " << config.vars.x << ", " << config.vars.n << std::endl;
        std::cout << "  Canvas: " << config.canvas << std::endl;
        std::cout << "  Evolution: " << config.evolution << std::endl;
    }

    auto start_computation = std::chrono::steady_clock::now();

    complex_t min, max;
    config.bounds(&min, &max);
    auto N = config.particle_number();

    complex_t * points;
    auto frame_size = config.canvas.height * config.canvas.width;

    switch (config.mode) {
        case ExecutionMode::Serial:
        {
            points = particles_serial(min, max, N, config.lloyd_iterations);
            auto canvas = new CanvasPixel [frame_size];
            evolve_serial(&config, canvas, points, N, fn_choice);
            write_video_serial(config, canvas);
            delete[] canvas;
            break;
        }
        case ExecutionMode::OpenMP:
        {
            points = particles_omp(min, max, N, config.lloyd_iterations);
            auto canvas_count = omp_get_max_threads();
            auto canvases = create_canvas_host(canvas_count, &config.canvas);
            evolve_omp(&config, canvases, points, N, fn_choice);
            write_video_omp(config, canvases, canvas_count);
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

            KernelSizes::set_SM();
            float tile_count_target = std::min(1024.f, (float) N / ((float) KernelSizes::get_SM() * 2.3f));

            Tiles tiles(&config, tile_count_target);
            unsigned tiles_count = tiles.total();
            if(verbose) {
                std::cout << "  Tiles: " << tiles.rows << 'x' << tiles.cols << '=' << tiles_count
                          << " (target: " << tile_count_target << ") with "
                          << (float) N / (float) tiles_count << " particles each" << std::endl;
                std::cout << "\nCUDA SM count: " << KernelSizes::get_SM() << "\nRegisters used by kernels: \n";
                pgen_print_regs();
                tiles_print_regs();
                std::cout << " - evolve: " << get_evolve_regs() << "\n";
                frame_print_regs();
                std::cout << std::endl;
            }

            points = particles_gpu(min, max, N, config.lloyd_iterations);
            auto tile_offsets = tiles.sort(min, max, points, N);
            auto canvas_count = get_canvas_count_serial(tile_offsets, tiles_count);
            auto canvases = create_canvas_device(canvas_count, &config.canvas);
            evolve_gpu(&config, canvases, canvas_count, points, N,
                       tile_offsets, tiles_count, fn_choice);
            cudaFree(tile_offsets);
            cudaFree(points);
            write_video_gpu(config, canvases, canvas_count);
            break;
        }
    }

    auto end_computation = std::chrono::steady_clock::now();
    float time_all = (std::chrono::duration<float,std::ratio<1>>(end_computation-start_computation)).count();
    std::cout << "All computations completed in " << time_all << 's' << std::endl;

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
