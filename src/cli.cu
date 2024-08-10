#include "cli.cuh"
#include <iostream>

void print_usage() {

}

int invalid_option(const char * name) {
    std::cout << '"' << name << "\" is not a valid option" << std::endl;
    return 2;
}

int missing_value(const char * name) {
    std::cout << "Missing value of option \"" << name << '"' <<std::endl;
    return 2;
}

int malformed_value(const char * name) {
    std::cout << "Malformed value of option \"" << name << '"' <<std::endl;
    return 2;
}