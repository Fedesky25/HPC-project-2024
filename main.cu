#include <iostream>
#include "utils.cuh"
#include "cli.hpp"
#include "getopt.h"
#include "tiles.cuh"
#include "particle_generator.cuh"
#include "fstream"
#include "complex_functions.cuh"

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
    auto fn = strtofn(argv[optind]);
    if(fn == FunctionChoice::NONE) {
        std::cerr << "Function string name not recognized" << std::endl;
        return 1;
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Output file: " << config.output << std::endl;
    std::cout << "  Complex numbers: " << config.vars.z[0] << ' ' << config.vars.z[1] << ' ' << config.vars.z[2] << std::endl;
    std::cout << "  Real and int numbers: " << config.vars.x << ", " << config.vars.n << std::endl;
    std::cout << "  Canvas: " << config.canvas << std::endl;
    std::cout << "  Evolution: " << config.evolution << std::endl;

    unsigned width, height;
    config.sizes(&width, &height);
    Tiles tiles;
    tiles.cover(width, height);
    std::cout << "  Tiles: " << tiles.rows << 'x' << tiles.cols << " (" << tiles.total() << ")" << std::endl;

    complex_t min, max;
    config.bounds(&min, &max);
    uint64_t N = config.particle_number();

    complex_t * points;

    switch (config.mode) {
        case ExecutionMode::Serial:
            points = particles_serial(min, max, N);
            break;
        case ExecutionMode::OpenMP:
            points = particles_omp(min, max, N);
            break;
        case ExecutionMode::GPU:
            points = particles_gpu(min, max, N);
            break;
    }


    return 0;
}
