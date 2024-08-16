//
// Created by Sofyh02 on 16/08/2024.
//

#ifndef HPC_PROJECT_2024_COMPLEX_FUNCTIONS_CUH
#define HPC_PROJECT_2024_COMPLEX_FUNCTIONS_CUH

#include "utils.cuh"

/** @return z1*z+z2 */
__device__ complex_t polynomial1(complex_t z, FnVariables variables);

/** @return z1*z^2+z2*z+z3 */
__device__ complex_t polynomial2(complex_t z, FnVariables variables);

/** @return z1*z^3+z2*z+z3 */
__device__ complex_t polynomial3(complex_t z, FnVariables variables);

/** @return (z-z1)(z-z2)(z-z3) */
__device__ complex_t polynomial_fact(complex_t z, FnVariables variables);

/** @return z^k */
__device__ complex_t int_power(complex_t z, FnVariables variables);

/** @return (z_k)^r */
__device__ complex_t real_power(complex_t z, FnVariables variables);

/** @return (z_k)^z1 */
__device__ complex_t complex_power(complex_t z, FnVariables variables);

/** @return e^z */
__device__ complex_t exponential0(complex_t z, FnVariables variables);

/** @return e^(z1*z) */
__device__ complex_t exponential1(complex_t z, FnVariables variables);

/** @return e^(z^n) */
__device__ complex_t exponential2(complex_t z, FnVariables variables);

/** @return e^[(z_k)^r] */
__device__ complex_t exponential3(complex_t z, FnVariables variables);

/** @return (z1_k)^z */
__device__ complex_t exponential4(complex_t z, FnVariables variables);

/** @return z^n * e^z */
__device__ complex_t exponential5(complex_t z, FnVariables variables);

/** @return (z_k)^r * e^z */
__device__ complex_t exponential6(complex_t z, FnVariables variables);

/** @return z1 * ln(z_k) */
__device__ complex_t log1(complex_t z, FnVariables variables);

/** @return z * ln(z_k) */
__device__ complex_t log2(complex_t z, FnVariables variables);

/** @return ln[(z+z1)_k] */
__device__ complex_t log3(complex_t z, FnVariables variables);

/** @return sin(z) */
__device__ complex_t sine(complex_t z, FnVariables variables);

/** @return cos(z) */
__device__ complex_t cosine(complex_t z, FnVariables variables);

/** @return sin(z + z1) */
__device__ complex_t sine_sum(complex_t z, FnVariables variables);

/** @return sin(z * z1) */
__device__ complex_t sine_mult(complex_t z, FnVariables variables);

/** @return cos(z * z1) */
__device__ complex_t cosine_mult(complex_t z, FnVariables variables);

/** @return z1 * sin(z * z2) */
__device__ complex_t mult_sine(complex_t z, FnVariables variables);

/** @return conj(z)*i */
__device__ complex_t conjugate_i(complex_t z, FnVariables variables);

/** @return conj(z) * z1 */
__device__ complex_t conjugate_z(complex_t z, FnVariables variables);

/** @return (z^2 -z1)(z-z2)^2 / (z^2 - z3)  */
__device__ complex_t fraction(complex_t z, FnVariables variables);


__device__ complex_t fibonacci(complex_t z, FnVariables variables);
__device__ complex_t gamma(complex_t z, FnVariables variables);

#endif //HPC_PROJECT_2024_COMPLEX_FUNCTIONS_CUH
