//
// Created by Sofyh02 on 16/08/2024.
//
#include "complex_functions.cuh"

__device__ complex_t polynomial1(complex_t z, FnVariables variables){
    return variables.z[0] * z + variables.z[2];
}

__device__ complex_t polynomial2(complex_t z, FnVariables variables){
    return variables.z[0] * (z*z) + variables.z[1] * z + variables.z[2];
}

__device__ complex_t polynomial3(complex_t z, FnVariables variables){
    return variables.z[0] * (z*z*z) + variables.z[1] * z + variables.z[2];
}

__device__ complex_t polynomial_fact(complex_t z, FnVariables variables){
    return (z-variables.z[0])*(z-variables.z[1])*(z-variables.z[2]);
}

__device__ complex_t complex_log(complex_t z, long k){
    complex_t phase;
    phase.real(0);
    phase.imag(cuda::std::arg(z) + 2*PI*k);
    return (log(cuda::std::abs(z)) + phase);
}

__device__ complex_t int_power(complex_t z, FnVariables variables){
    return cuda::std::pow(z, variables.n);
}

__device__ complex_t real_power(complex_t z, FnVariables variables){
    return cuda::std::exp(complex_log(z, variables.n) * variables.x);
}

__device__ complex_t complex_power(complex_t z, FnVariables variables){
    return cuda::std::exp(complex_log(z, variables.n) * variables.z[0]);
}

__device__ complex_t exponential0(complex_t z, FnVariables variables){
    return cuda::std::exp(z);
}

__device__ complex_t exponential1(complex_t z, FnVariables variables){
    return cuda::std::exp(z * variables.z[0]);
}

__device__ complex_t exponential2(complex_t z, FnVariables variables){
    return cuda::std::exp(int_power(z, variables));
}

__device__ complex_t exponential3(complex_t z, FnVariables variables){
    return cuda::std::exp(real_power(z, variables));
}

__device__ complex_t exponential4(complex_t z, FnVariables variables){
    return cuda::std::exp(complex_log(variables.z[0], variables.n) * z);
}

__device__ complex_t exponential5(complex_t z, FnVariables variables){
    return int_power(z, variables) * cuda::std::exp(z);
}

__device__ complex_t exponential6(complex_t z, FnVariables variables){
    return real_power(z, variables) * cuda::std::exp(z);
}

__device__ complex_t log1(complex_t z, FnVariables variables){
    return variables.z[0] * complex_log(z, variables.n);
}

__device__ complex_t log2(complex_t z, FnVariables variables){
    return z * complex_log(z, variables.n);
}

__device__ complex_t log3(complex_t z, FnVariables variables){
    return complex_log(z + variables.z[0], variables.n);
}

__device__ complex_t sine(complex_t z, FnVariables variables){
    return cuda::std::sin(z);
}

__device__ complex_t cosine(complex_t z, FnVariables variables){
    return cuda::std::cos(z);
}

__device__ complex_t sine_sum(complex_t z, FnVariables variables){
    return cuda::std::sin(z + variables.z[0]);
}

__device__ complex_t sine_mult(complex_t z, FnVariables variables){
    return cuda::std::sin(z * variables.z[0]);
}

__device__ complex_t cosine_mult(complex_t z, FnVariables variables){
    return cuda::std::cos(z * variables.z[0]);
}
__device__ complex_t mult_sine(complex_t z, FnVariables variables){
    return variables.z[0] * cuda::std::sin(z * variables.z[1]);
}

__device__ complex_t tangent(complex_t z, FnVariables variables){
    return cuda::std::tan(z);
}

__device__ complex_t conjugate_i(complex_t z, FnVariables variables){
    complex_t i;
    i.real(0);
    i.imag(1);
    return i * cuda::std::conj(z);
}

__device__ complex_t conjugate_z(complex_t z, FnVariables variables){
    return variables.z[0] * cuda::std::conj(z);
}

__device__ complex_t fraction(complex_t z, FnVariables variables){
    return (z*z - variables.z[0]) * cuda::std::pow(z - variables.z[1], 2) / (z*z + variables.z[2]);
}

__device__ complex_t fibonacci(complex_t z, FnVariables variables){
    double phi = (1+cuda::std::sqrt(5))/2;
    return (cuda::std::pow(phi, z) - (cos(PI*z) * cuda::std::pow(phi, -z)))/cuda::std::sqrt(5);
}

__device__ complex_t gamma(complex_t z, FnVariables variables){
    // TODO define dx and tolerance
    double dx, tol;
    double x = 0;
    complex_t gamma = 0, prev;
    do {
        prev = gamma;
        x += dx;
        gamma += (cuda::std::pow(x, z - (complex_t)1) * cuda::std::exp(-x));
    } while(cuda::std::norm(gamma-prev) > tol);
    return gamma;
}

__device__ complex_t hsine(complex_t z, FnVariables variables){
    return cuda::std::sinh(z);
}

__device__ complex_t hsine_sum(complex_t z, FnVariables variables){
    return cuda::std::sinh(z + variables.z[0]);
}

__device__ complex_t hsine_mult(complex_t z, FnVariables variables){
    return cuda::std::sinh(z * variables.z[0]);
}

__device__ complex_t hcosine(complex_t z, FnVariables variables){
    return cuda::std::cosh(z);
}

__device__ complex_t hcosine_sum(complex_t z, FnVariables variables){
    return cuda::std::cosh(z + variables.z[0]);
}

__device__ complex_t hcosine_mult(complex_t z, FnVariables variables){
    return cuda::std::cosh(z * variables.z[0]);
}

__device__ complex_t htangent(complex_t z, FnVariables variables){
    return cuda::std::tanh(z);
}

__device__ complex_t htangent_sum(complex_t z, FnVariables variables){
    return cuda::std::tanh(z + variables.z[0]);
}

__device__ complex_t htangent_mult(complex_t z, FnVariables variables){
    return cuda::std::tanh(z * variables.z[0]);
}