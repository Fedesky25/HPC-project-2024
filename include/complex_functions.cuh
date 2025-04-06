//
// Created by Sofyh02 on 16/08/2024.
//

#ifndef HPC_PROJECT_2024_COMPLEX_FUNCTIONS_CUH
#define HPC_PROJECT_2024_COMPLEX_FUNCTIONS_CUH

#include "config.cuh"

enum class FunctionChoice {
    Poly1, Poly2, Poly3, PolyFact,
    PowInt, PowReal, PowComplex,
    ExpSimple, ExpParametric, ExpPowInt, ExpPowReal, Zxp, PowIntExp, PowRealExp,
    LogSimple, LogParametric, LogMul,
    SinSimple, SinParametric, CosSimple, CosParametric, TanSimple, TanParametric,
    SinhSimple, SinhParametric, CoshSimple, CoshParametric, TanhSimple, TanhParametric,
    Conjugate, Fraction, Fibonacci, Gamma,
    NONE = INT_MAX,
};

/**
 * Gets the function choice given its string representation
 * @param str string representation
 * @return choice
 */
FunctionChoice strtofn(const char * str);

typedef complex_t (*ComplexFunction_t)(complex_t, FnVariables*);

/**
 * Given the choice, it returns a host pointer to a host function
 * @param choice function choice
 * @return host function
 */
__host__ ComplexFunction_t get_function_host(FunctionChoice choice);

/**
 * Given the choice, it returns a pointer to a device function which can be passed to a kernel
 * @param choice function choice
 * @return device function
 */
__host__ ComplexFunction_t get_function_global(FunctionChoice choice);

/**
 * Given the choice, it returns a device pointer to a device function
 * @param choice function choice
 * @return device function
 */
__device__ ComplexFunction_t get_function_device(FunctionChoice choice);

#endif //HPC_PROJECT_2024_COMPLEX_FUNCTIONS_CUH
