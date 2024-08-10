#ifndef HPC_PROJECT_2024_CLI_CUH
#define HPC_PROJECT_2024_CLI_CUH

struct Options {

};

void print_usage();
int invalid_option(const char * name);
int missing_value(const char * name);
int malformed_value(const char * name);

#endif //HPC_PROJECT_2024_CLI_CUH
