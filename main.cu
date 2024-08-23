#include <iostream>
#include "utils.cuh"
#include "cli.hpp"
#include "getopt.h"
#include "tiles.cuh"
#include "particle_generator.cuh"
#include "fstream"
#include "complex_functions.cuh"
#include "canvas.cuh"
#include "omp.h"

#define CANVAS_OUTPUT { \
    auto size = sizeof(CanvasPixel) * config.canvas.width * config.canvas.height * canvas_count; \
    size = 1 + ((size-1) >> 20);                                                                 \
    std::cout << "Number of canvases: " << canvas_count << " (" << size << "MB)" << std::endl;   \
}


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

    complex_t min, max;
    config.bounds(&min, &max);
    uint64_t N = config.particle_number();

    complex_t * points;
    uint32_t canvas_count;

    switch (config.mode) {
        case ExecutionMode::Serial:
            points = particles_serial(min, max, N);
            break;
        case ExecutionMode::OpenMP:
        {
            points = particles_omp(min, max, N);
            canvas_count = omp_get_max_threads();
            CANVAS_OUTPUT
            auto canvases = create_canvas_device(canvas_count, &config.canvas);
            break;
        }
        case ExecutionMode::GPU:
        {
            Tiles tiles(&config);
            unsigned tiles_count = tiles.total();
            std::cout << "  Tiles: " << tiles.rows << 'x' << tiles.cols << '=' << tiles_count << " with "
                      << (float) N / (float) tiles_count << " particles each" << std::endl;
            points = particles_gpu(min, max, N);
            uint_fast16_t *tile_map, *count_per_tile;
            timers(1) tick(0)
            tiles.sort(min, max, points, N, &tile_map, &count_per_tile);
            canvas_count = 0;
            for(unsigned i=0; i<tiles_count; i++) {
                if(count_per_tile[i] > canvas_count) canvas_count = count_per_tile[i];
            }
            tock_ms(0)
            std::cout << "Particles sorted by tiles in " << t_elapsed << "ms" << std::endl;
            CANVAS_OUTPUT
            auto canvases = create_canvas_device(canvas_count, &config.canvas);
            break;
        }
    }


    return 0;
}
