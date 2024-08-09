#include <iostream>
#include "utils.cuh"


int main(int argc, char * argv[]) {
    if(argc < 2) {
        print_usage();
        return 1;
    }

    TilesCount tiles;
    complex_t z[3] = { 1.0, {0.0,1.0}, 2.0 };
    double r = 0.5;
    long n = 0;
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
            else if(strcmp(str, "screen") == 0) {
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
                    if(i+1 == argc) return missing_value("distance (d)");
                    distance = strtoul(argv[++i], &rest, 10);
                    if(*rest != '\0') return malformed_value("distance (d)");
                    if(distance < 1) {
                        std::cout << "Distance cannot be lower than 1" << std::endl;
                    }
                    break;
                case 's':
                    if(i+1 == argc) return missing_value("screen (s)");
                    resolution.parse(argv[++i]);
                    if(!resolution) return malformed_value("screen (s)");
                    break;
                case 'n':
                    if(i+1 == argc) return missing_value("integer number (n)");
                    n = strtol(argv[++i], &rest, 10);
                    if(*rest != '\0') return malformed_value("integer number (n)");
                    break;
                case 'r':
                    if(i+1 == argc) return missing_value("real number (r)");
                    r = strtod(argv[++i], &rest);
                    if(*rest != '\0') return malformed_value("real number (r)");
                    break;
                case 'c':
                {
                    auto index = argv[i][2] - '1';
                    if(index<0 || index>2) {
                        std::cout << "Index of complex number can be 1,2, or 3" << std::endl;
                        return 2;
                    }
                    if(i+1 == argc) return missing_value("complex number (c)");
                    z[index] = parse_complex(argv[++i]);
                    if(std::isnan(z[index].real()) || std::isnan(z[index].imag()))
                        return malformed_value("complex number (c)");
                    break;
                }

                case '\0':
                default:
                    return invalid_option(argv[i]+1);
            }
        }
    }

    tiles.cover(resolution.width + distance*6, resolution.height + distance*6);

    std::cout << "Configuration:" << std::endl;
    std::cout << "Complex numbers: " << z[0] << ' ' << z[1] << ' ' << z[2] << std::endl;
    std::cout << "Real and int numbers: " << r << ", " << n << std::endl;
    std::cout << "Screen and tiles: " << resolution.width << 'x' << resolution.height << std::endl;
    std::cout << "Tiles: " << tiles.rows << 'x' << tiles.cols << " (" << tiles.total()
              << ") -> " << (float)resolution.width/(float)tiles.cols
              << 'x' << (float)resolution.height/(float)tiles.rows << std::endl;
    return 0;
}
