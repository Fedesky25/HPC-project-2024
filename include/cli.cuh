#ifndef HPC_PROJECT_2024_CLI_CUH
#define HPC_PROJECT_2024_CLI_CUH

#include "utils.cuh"

/** Prints the usage of the CLI */
void print_usage();

/**
 * Parses the provided arguments to complete the configuration
 * @param argc argument count
 * @param argv argument values
 * @param config pointer to configuration struct
 * @return whether errors occurred or not
 */
bool parse_args(int argc, char * argv[], Configuration * config);

#endif //HPC_PROJECT_2024_CLI_CUH
