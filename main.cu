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
#include <fstream>

int main(int argc, char * argv[]) {
    if(argc < 2) {
        print_usage();
        return 1;
    }

    Configuration config;
    auto error = parse_args(argc, argv, &config);
    if(error) return 1;

    auto fn_choice = FunctionChoice::NONE;
    if(config.evolution.frame_count) {
        EXIT_IF(optind >= argc, "Missing function to plot")
        fn_choice = strtofn(argv[optind]);
        EXIT_IF(fn_choice == FunctionChoice::NONE, "Function string name not recognized")
    }

    if(verbose) {
        std::cout << "Configuration:";
        std::cout << "\n  Output file: " << config.output;
        std::cout << "\n  Complex numbers: " << config.vars.z[0] << ' ' << config.vars.z[1] << ' ' << config.vars.z[2];
        std::cout << "\n  Real and int numbers: " << config.vars.x << ", " << config.vars.n;
        std::cout << "\n  Canvas: " << config.canvas;
        std::cout << "\n  Evolution: " << config.evolution << std::endl;
    }

    complex_t min, max;
    config.bounds(&min, &max);

    uint32_t N = 0;
    complex_t * points = nullptr;
    if(0 == config.lloyd_iterations && config.particles_file != nullptr) {
        if(verbose) std::cout << '\n';
        std::cout << "Reading particles from " << config.particles_file << std::flush;
        std::ifstream f(config.particles_file);
        if(!f.good()) {
            std::cerr << "\nFile '" << config.particles_file << "' does not exists!" << std::endl;
            return 1;
        }
        f >> N;
        points = (complex_t *) malloc(N * sizeof(complex_t));
        for(uint32_t i=0; i<N; i++) f >> points[i];
        f.close();
        std::cout << " -> done" << std::endl;
    }
    else N = config.particle_number();
    auto frame_size = config.canvas.height * config.canvas.width;

    auto start_computation = std::chrono::steady_clock::now();

    switch (config.mode) {
        case ExecutionMode::Serial:
        {
            if(!points) points = particles_serial(min, max, N, config.lloyd_iterations);
            if(!config.evolution.frame_count) break;
            auto canvas = new CanvasPixel [frame_size];
            evolve_serial(&config, canvas, points, N, fn_choice);
            write_video_serial(config, canvas);
            delete[] canvas;
            break;
        }
        case ExecutionMode::OpenMP:
        {
            if(!points) points = particles_omp(min, max, N, config.lloyd_iterations);
            if(!config.evolution.frame_count) break;
            auto canvas_count = omp_get_max_threads();
            auto canvases = create_canvas_host(canvas_count, &config.canvas);
            evolve_omp(&config, canvases, points, N, fn_choice);
            auto rows = reshape_canvas_host(canvas_count, canvases, config.canvas);
            free_canvas_host(canvas_count, canvases);
            write_video_omp(config, rows);
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
                std::cout << "\nTiles: " << tiles.rows << 'x' << tiles.cols << '=' << tiles_count
                          << " (target: " << tile_count_target << ") with "
                          << (float) N / (float) tiles_count << " particles each" << std::endl;
                std::cout << "CUDA SM count: " << KernelSizes::get_SM();
                WHEN_OK(
                    std::cout << "\nRegisters used by kernels: \n";
                    pgen_print_regs();
                    tiles_print_regs();
                    std::cout << " - evolve: " << get_evolve_regs() << "\n";
                    frame_print_regs();
                    std::cout << std::endl;
                )
            }

            complex_t * d_points;
            if(points) {
                CATCH_CUDA_ERROR(cudaMalloc(&d_points, N*sizeof(complex_t)))
                cudaMemcpy(d_points, points, N*sizeof(complex_t), cudaMemcpyHostToDevice);
            }
            else {
                d_points = particles_gpu(min, max, N, config.lloyd_iterations);
                if(config.particles_file && config.lloyd_iterations) {
                    points = (complex_t*) malloc(N * sizeof(complex_t));
                    cudaMemcpy(points, d_points, N*sizeof(complex_t), cudaMemcpyDeviceToHost);
                }
            }
            if(!config.evolution.frame_count) break;
            auto tile_offsets = tiles.sort(min, max, d_points, N);
            auto canvas_count = get_canvas_count_serial(tile_offsets, tiles_count);
            auto canvases = create_canvas_device(canvas_count, &config.canvas);
            evolve_gpu(&config, canvases, canvas_count, d_points, N,
                       tile_offsets, tiles_count, fn_choice);
            cudaFree(tile_offsets);
            cudaFree(d_points);
            write_video_gpu(config, canvases, canvas_count);
            break;
        }
    }

    auto end_computation = std::chrono::steady_clock::now();
    float time_all = (std::chrono::duration<float,std::ratio<1>>(end_computation-start_computation)).count();
    std::cout << "All computations completed in " << time_all << 's' << std::endl;

    if(config.particles_file && config.lloyd_iterations) {
        std::cout << "Saving particles inside " << config.particles_file << std::flush;
        std::ofstream f(config.particles_file);
        f << N;
        for(uint32_t i=0; i<N; i++) f << points[i];
        f << std::endl;
        f.close();
        std::cout << " -> done!" << std::endl;
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
