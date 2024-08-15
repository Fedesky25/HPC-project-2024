#include <iostream>
#include "utils.cuh"
#include "cli.hpp"
#include "tiles.cuh"
#include "particle_generator.cuh"
#include "fstream"

int main(int argc, char * argv[]) {
    if(argc < 2) {
        print_usage();
        return 1;
    }

    Configuration config;
    auto err = parse_args(argc, argv, &config);
    if(err) return 2;


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
//    auto p1 = particles_omp(min, max, N);
//    tiles.distribute(min, max, p1, N);
//    auto particles = particles_mixed(min, max, N);
//    tiles.distribute(min, max, particles, N);


    return 0;
}
