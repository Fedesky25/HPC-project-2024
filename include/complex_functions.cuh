//
// Created by Sofyh02 on 16/08/2024.
//

#ifndef HPC_PROJECT_2024_COMPLEX_FUNCTIONS_CUH
#define HPC_PROJECT_2024_COMPLEX_FUNCTIONS_CUH

#include "utils.cuh"

typedef complex_t (*ComplexFunction_t)(complex_t, FnVariables*);

ComplexFunction_t get_function_from_string(const char * str);

#endif //HPC_PROJECT_2024_COMPLEX_FUNCTIONS_CUH
