#include <iostream>
#include "utils.cuh"


int main(int argc, char * argv[]) {
    if(argc < 2) {
        print_usage();
        return 1;
    }

    TilesCount tiles;
    complex_t z1(0), z2(0), z3(0);
    unsigned distance = 10;
    ScreenResolution resolution;

    char * rest;

    for(int i=1; i<argc; i++) {
        if(argv[i][0] != '-') return invalid_option(argv[i]);
        if(argv[i][1] == '-') {
            // long name
            const char * str = argv[i]+2;
            if(strcmp(str, "distance") == 0) {
                if(i + 1 == argc) return missing_value(str);
                distance = strtoul(argv[++i], &rest, 10);
                if(*rest != '\0') return malformed_value(str);
                if(distance < 1) {
                    std::cout << "Distance cannot be lower than 1" << std::endl;
                    return 2;
                }
            }
            else if(strcmp(str, "resolution") == 0) {
                if(i+1 == argc) return missing_value(str);
                resolution.parse(argv[++i]);
                if(!resolution) return malformed_value(str);
            }
            else return invalid_option(str);
        }
        else {
            // short name
            switch (argv[i][1]) {
                case 'd':
                    if(i+1 == argc) return missing_value("distance");
                    distance = strtoul(argv[++i], &rest, 10);
                    if(*rest != '\0') return malformed_value("distance");
                    if(distance < 1) {
                        std::cout << "Distance cannot be lower than 1" << std::endl;
                    }
                    break;
                case 'r':
                    if(i+1 == argc) return missing_value("resolution");
                    resolution.parse(argv[++i]);
                    if(!resolution) return malformed_value("resolution");
                    break;
                case '\0':
                default:
                    return invalid_option(argv[i]+1);
            }
        }
    }

    tiles.cover(resolution.width + distance*6, resolution.height + distance*6);

    std::cout << "Configuration:" << std::endl;
    std::cout << "Complex numbers: " << z1 << ' ' << z2 << ' ' << z3 << std::endl;
    std::cout << "Resolution: " << resolution.width << 'x' << resolution.height << std::endl;
    std::cout << "Tiles: " << tiles.rows << 'x' << tiles.cols << " (" << tiles.total()
              << ") -> " << (float)resolution.width/(float)tiles.cols << 'x' << (float)resolution.height/(float)tiles.rows << std::endl;
    return 0;
}
