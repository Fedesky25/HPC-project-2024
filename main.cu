#include <iostream>
#include "utils.cuh"
#include "cli.cuh"


int main(int argc, char * argv[]) {
    if(argc < 2) {
        print_usage();
        return 1;
    }

    Configuration config;
    auto err = parse_args(argc, argv, &config);
    if(err) return 2;

    TilesCount tiles;
//    tiles.cover(resolution.width + distance*6, resolution.height + distance*6);

    std::cout << "Configuration:" << std::endl;
    std::cout << "Complex numbers: " << config.vars.z[0] << ' ' << config.vars.z[1] << ' ' << config.vars.z[2] << std::endl;
    std::cout << "Real and int numbers: " << config.vars.x << ", " << config.vars.n << std::endl;
    std::cout << "Screen and tiles: " << config.canvas.width << 'x' << config.canvas.height << std::endl;
//    std::cout << "Tiles: " << tiles.rows << 'x' << tiles.cols << " (" << tiles.total()
//              << ") -> " << (float)resolution.width/(float)tiles.cols
//              << 'x' << (float)resolution.height/(float)tiles.rows << std::endl;
    return 0;
}
