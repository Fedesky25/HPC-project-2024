//
// Created by Sofyh02 on 16/08/2024.
//
#include "complex_functions.cuh"

#define PHI 1.618033988749895
#define INV_SQRT_5 0.4472135954999579

__device__ complex_t polynomial1(complex_t z, FnVariables* variables){
    return variables->z[0] * z + variables->z[2];
}

__device__ complex_t polynomial2(complex_t z, FnVariables* variables){
    return variables->z[0] * (z*z) + variables->z[1] * z + variables->z[2];
}

__device__ complex_t polynomial3(complex_t z, FnVariables* variables){
    return variables->z[0] * (z*z*z) + variables->z[1] * z + variables->z[2];
}

__device__ complex_t polynomial_fact(complex_t z, FnVariables* variables){
    return (z-variables->z[0])*(z-variables->z[1])*(z-variables->z[2]);
}

__device__ complex_t complex_log(complex_t z, long k){
    complex_t phase;
    phase.real(0);
    phase.imag(cuda::std::arg(z) + 2*PI*k);
    return (log(cuda::std::abs(z)) + phase);
}

__device__ complex_t int_power(complex_t z, FnVariables* variables){
    return cuda::std::pow(z, variables->n);
}

__device__ complex_t real_power(complex_t z, FnVariables* variables){
    return cuda::std::exp(complex_log(z, variables->n) * variables->x);
}

__device__ complex_t complex_power(complex_t z, FnVariables* variables){
    return cuda::std::exp(complex_log(z, variables->n) * variables->z[0]);
}

__device__ complex_t exponential0(complex_t z, FnVariables* variables){
    return cuda::std::exp(z);
}

__device__ complex_t exponential1(complex_t z, FnVariables* variables){
    return cuda::std::exp(z * variables->z[0]);
}

__device__ complex_t exponential2(complex_t z, FnVariables* variables){
    return cuda::std::exp(int_power(z, variables));
}

__device__ complex_t exponential3(complex_t z, FnVariables* variables){
    return cuda::std::exp(real_power(z, variables));
}

__device__ complex_t exponential4(complex_t z, FnVariables* variables){
    return cuda::std::exp(complex_log(variables->z[0], variables->n) * z);
}

__device__ complex_t exponential5(complex_t z, FnVariables* variables){
    return int_power(z, variables) * cuda::std::exp(z);
}

__device__ complex_t exponential6(complex_t z, FnVariables* variables){
    return real_power(z, variables) * cuda::std::exp(z);
}

__device__ complex_t log1(complex_t z, FnVariables* variables){
    return variables->z[0] * complex_log(z, variables->n);
}

__device__ complex_t log2(complex_t z, FnVariables* variables){
    return z * complex_log(z, variables->n);
}

__device__ complex_t log3(complex_t z, FnVariables* variables){
    return complex_log(z + variables->z[0], variables->n);
}

__device__ complex_t sine(complex_t z, FnVariables* variables){
    return cuda::std::sin(z);
}

__device__ complex_t cosine(complex_t z, FnVariables* variables){
    return cuda::std::cos(z);
}

__device__ complex_t sine_sum(complex_t z, FnVariables* variables){
    return cuda::std::sin(z + variables->z[0]);
}

__device__ complex_t sine_mult(complex_t z, FnVariables* variables){
    return cuda::std::sin(z * variables->z[0]);
}

__device__ complex_t cosine_mult(complex_t z, FnVariables* variables){
    return cuda::std::cos(z * variables->z[0]);
}
__device__ complex_t mult_sine(complex_t z, FnVariables* variables){
    return variables->z[0] * cuda::std::sin(z * variables->z[1]);
}

__device__ complex_t tangent(complex_t z, FnVariables* variables){
    return cuda::std::tan(z);
}

__device__ complex_t conjugate_i(complex_t z, FnVariables* variables){
    complex_t i;
    i.real(0);
    i.imag(1);
    return i * cuda::std::conj(z);
}

__device__ complex_t conjugate_z(complex_t z, FnVariables* variables){
    return variables->z[0] * cuda::std::conj(z);
}

__device__ complex_t fraction(complex_t z, FnVariables* variables){
    return (z*z - variables->z[0]) * cuda::std::pow(z - variables->z[1], 2) / (z*z + variables->z[2]);
}

__device__ complex_t fibonacci(complex_t z, FnVariables* variables){
    return (cuda::std::pow(PHI, z) - (cos(PI*z) * cuda::std::pow(PHI, -z))) * INV_SQRT_5;
}

__device__ complex_t gamma(complex_t z, FnVariables* variables){
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

__device__ complex_t hsine(complex_t z, FnVariables* variables){
    return cuda::std::sinh(z);
}

__device__ complex_t hsine_sum(complex_t z, FnVariables* variables){
    return cuda::std::sinh(z + variables->z[0]);
}

__device__ complex_t hsine_mult(complex_t z, FnVariables* variables){
    return cuda::std::sinh(z * variables->z[0]);
}

__device__ complex_t hcosine(complex_t z, FnVariables* variables){
    return cuda::std::cosh(z);
}

__device__ complex_t hcosine_sum(complex_t z, FnVariables* variables){
    return cuda::std::cosh(z + variables->z[0]);
}

__device__ complex_t hcosine_mult(complex_t z, FnVariables* variables){
    return cuda::std::cosh(z * variables->z[0]);
}

__device__ complex_t htangent(complex_t z, FnVariables* variables){
    return cuda::std::tanh(z);
}

__device__ complex_t htangent_sum(complex_t z, FnVariables* variables){
    return cuda::std::tanh(z + variables->z[0]);
}

__device__ complex_t htangent_mult(complex_t z, FnVariables* variables){
    return cuda::std::tanh(z * variables->z[0]);
}

template<unsigned N>
constexpr uint64_t bytes_to_uint(const char * str) {
    static_assert(N > 0 && N < 9, "Number of bytes must be in the range [1, 8]");
    uint64_t num = static_cast<uint8_t>(str[0]);
    for(unsigned i=0; i<N; i++) {
        num <<= 8;
        num += static_cast<uint8_t>(str[i]);
    }
    return num;
}


__device__ ComplexFunction_t d_poly1 = polynomial1;
__device__ ComplexFunction_t d_poly2 = polynomial2;
__device__ ComplexFunction_t d_poly3 = polynomial3;
__device__ ComplexFunction_t d_poly_fact = polynomial_fact;

__device__ ComplexFunction_t d_powi = int_power;
__device__ ComplexFunction_t d_powr = real_power;
__device__ ComplexFunction_t d_powc = complex_power;

__device__ ComplexFunction_t d_exp = exponential0;
__device__ ComplexFunction_t d_exp_mul = exponential1;
__device__ ComplexFunction_t d_exp_pow_n = exponential2;
__device__ ComplexFunction_t d_exp_pow_r = exponential3;
__device__ ComplexFunction_t d_zxp = exponential4;
__device__ ComplexFunction_t d_pow_n_exp = exponential5;
__device__ ComplexFunction_t d_pow_r_exp = exponential6;

__device__ ComplexFunction_t d_ln = log1;
__device__ ComplexFunction_t d_mul_ln = log2;
__device__ ComplexFunction_t d_ln_sum = log3;

__device__ ComplexFunction_t d_sin = sine;
__device__ ComplexFunction_t d_sin_sum = sine_sum;
__device__ ComplexFunction_t d_sin_mul = sine_mult;
__device__ ComplexFunction_t d_mul_sin_mul = mult_sine;
__device__ ComplexFunction_t d_cos = cosine;
__device__ ComplexFunction_t d_cos_mul = cosine_mult;
__device__ ComplexFunction_t d_tan = tangent;

__device__ ComplexFunction_t d_sinh = hsine;
__device__ ComplexFunction_t d_sinh_sum = hsine_sum;
__device__ ComplexFunction_t d_sinh_mul = hsine_mult;
__device__ ComplexFunction_t d_cosh = hcosine;
__device__ ComplexFunction_t d_cosh_sum = hcosine_sum;
__device__ ComplexFunction_t d_cosh_mul = hcosine_mult;
__device__ ComplexFunction_t d_tanh = htangent;
__device__ ComplexFunction_t d_tanh_sum = htangent_sum;
__device__ ComplexFunction_t d_tanh_mul = htangent_mult;

__device__ ComplexFunction_t d_mul_conj = conjugate_z;
__device__ ComplexFunction_t d_frac = fraction;
__device__ ComplexFunction_t d_gamma = gamma;
__device__ ComplexFunction_t d_fib = fibonacci;


ComplexFunction_t get_function_from_string(const char * str) {
    auto len = strlen(str);
    ComplexFunction_t fn = nullptr;
    switch (len) {
        case 2:
            switch (bytes_to_uint<2>(str)) {
                case bytes_to_uint<2>("^n"):
                    cudaMemcpyFromSymbol(&fn, d_powi, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<2>("^r"):
                    cudaMemcpyFromSymbol(&fn, d_powr, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<2>("^c"):
                    cudaMemcpyFromSymbol(&fn, d_powc, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<2>("ln"):
                    cudaMemcpyFromSymbol(&fn, d_ln, sizeof(ComplexFunction_t));
                    break;
            }
            break;
        case 3:
            switch (bytes_to_uint<3>(str)) {
                case bytes_to_uint<3>("*ln"):
                    cudaMemcpyFromSymbol(&fn, d_mul_ln, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<3>("ln+"):
                    cudaMemcpyFromSymbol(&fn, d_ln_sum, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<3>("exp"):
                    cudaMemcpyFromSymbol(&fn, d_exp, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<3>("zxp"):
                    cudaMemcpyFromSymbol(&fn, d_zxp, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<3>("sin"):
                    cudaMemcpyFromSymbol(&fn, d_sin, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<3>("cos"):
                    cudaMemcpyFromSymbol(&fn, d_cos, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<3>("tan"):
                    cudaMemcpyFromSymbol(&fn, d_tan, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<3>("fib"):
                    cudaMemcpyFromSymbol(&fn, d_fib, sizeof(ComplexFunction_t));
                    break;
            }
            break;
        case 4:
            switch (bytes_to_uint<4>(str)) {
                case bytes_to_uint<4>("sin+"):
                    cudaMemcpyFromSymbol(&fn, d_sin_sum, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<4>("sin*"):
                    cudaMemcpyFromSymbol(&fn, d_sin_mul, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<4>("cos*"):
                    cudaMemcpyFromSymbol(&fn, d_cosh_mul, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<4>("sinh"):
                    cudaMemcpyFromSymbol(&fn, d_sinh, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<4>("cosh"):
                    cudaMemcpyFromSymbol(&fn, d_cosh, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<4>("tanh"):
                    cudaMemcpyFromSymbol(&fn, d_tanh, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<4>("exp*"):
                    cudaMemcpyFromSymbol(&fn, d_exp_mul, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<4>("frac"):
                    cudaMemcpyFromSymbol(&fn, d_frac, sizeof(ComplexFunction_t));
                    break;
            }
            break;
        case 5:
            switch (bytes_to_uint<5>(str)) {
                case bytes_to_uint<5>("poly1"):
                    cudaMemcpyFromSymbol(&fn, d_poly1, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("poly2"):
                    cudaMemcpyFromSymbol(&fn, d_poly2, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("poly3"):
                    cudaMemcpyFromSymbol(&fn, d_poly3, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("poly*"):
                    cudaMemcpyFromSymbol(&fn, d_poly_fact, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("exp^n"):
                    cudaMemcpyFromSymbol(&fn, d_exp_pow_n, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("exp^r"):
                    cudaMemcpyFromSymbol(&fn, d_exp_pow_r, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("*sin*"):
                    cudaMemcpyFromSymbol(&fn, d_mul_sin_mul, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("sinh+"):
                    cudaMemcpyFromSymbol(&fn, d_sinh_sum, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("sinh*"):
                    cudaMemcpyFromSymbol(&fn, d_sinh_mul, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("cosh+"):
                    cudaMemcpyFromSymbol(&fn, d_cosh_sum, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("cosh*"):
                    cudaMemcpyFromSymbol(&fn, d_cosh_mul, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("tanh+"):
                    cudaMemcpyFromSymbol(&fn, d_tanh_sum, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("tanh*"):
                    cudaMemcpyFromSymbol(&fn, d_tanh_mul, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("*conj"):
                    cudaMemcpyFromSymbol(&fn, d_mul_conj, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("gamma"):
                    cudaMemcpyFromSymbol(&fn, d_gamma, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<5>("binet"):
                    cudaMemcpyFromSymbol(&fn, d_fib, sizeof(ComplexFunction_t));
                    break;
            }
            break;
        case 6:
            switch (bytes_to_uint<6>(str)) {
                case bytes_to_uint<6>("^n*exp"):
                    cudaMemcpyFromSymbol(&fn, d_pow_n_exp, sizeof(ComplexFunction_t));
                    break;
                case bytes_to_uint<6>("^r*exp"):
                    cudaMemcpyFromSymbol(&fn, d_pow_r_exp, sizeof(ComplexFunction_t));
                    break;
            }
    }
    return fn;
}