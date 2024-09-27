//
// Created by Sofyh02 on 16/08/2024.
//
#include "complex_functions.cuh"

#define PHI        1.6180339887498950
#define INV_SQRT_5 0.4472135954999579
#define SQRT_2PI   2.5066282746310002

__device__ __host__ complex_t complex_log(complex_t z, long k){
    complex_t phase;
    phase.real(0);
    phase.imag(cuda::std::arg(z) + 2*PI*k);
    return (log(cuda::std::abs(z)) + phase);
}

// ------------------------------------------------------------------------------------- polynomial

__device__ __host__ complex_t polynomial1(complex_t z, FnVariables* variables){
    return variables->z[0] * z + variables->z[1];
}

__device__ __host__ complex_t polynomial2(complex_t z, FnVariables* variables){
    return variables->z[0] * (z*z) + variables->z[1] * z + variables->z[2];
}

__device__ __host__ complex_t polynomial3(complex_t z, FnVariables* variables){
    return variables->z[0] * (z*z*z) + variables->z[1] * z + variables->z[2];
}

__device__ __host__ complex_t polynomial_fact(complex_t z, FnVariables* variables){
    return (z-variables->z[0])*(z-variables->z[1])*(z-variables->z[2]);
}

// ------------------------------------------------------------------------------------- power

__device__ __host__ complex_t int_power(complex_t z, FnVariables* variables){
    return cuda::std::pow(z, variables->n);
}

__device__ __host__ complex_t real_power(complex_t z, FnVariables* variables){
    return cuda::std::exp(complex_log(z, variables->n) * variables->x);
}

__device__ __host__ complex_t complex_power(complex_t z, FnVariables* variables){
    return cuda::std::exp(complex_log(z, variables->n) * variables->z[0]);
}

// ------------------------------------------------------------------------------------- exponential

__device__ __host__ complex_t exp_simple(complex_t z, FnVariables*){
    return cuda::std::exp(z);
}

__device__ __host__ complex_t exp_parametric(complex_t z, FnVariables* variables){
    return cuda::std::exp(z * variables->z[0] + variables->z[1]);
}

__device__ __host__ complex_t exp_pow_int(complex_t z, FnVariables* variables){
    return cuda::std::exp(int_power(z, variables));
}

__device__ __host__ complex_t exp_pow_real(complex_t z, FnVariables* variables){
    return cuda::std::exp(real_power(z, variables));
}

__device__ __host__ complex_t zxp(complex_t z, FnVariables* variables){
    return cuda::std::exp(complex_log(variables->z[0], variables->n) * z);
}

__device__ __host__ complex_t pow_int_exp(complex_t z, FnVariables* variables){
    return int_power(z, variables) * cuda::std::exp(z);
}

__device__ __host__ complex_t pow_real_exp(complex_t z, FnVariables* variables){
    return real_power(z, variables) * cuda::std::exp(z);
}

// ------------------------------------------------------------------------------------- logarithm

__device__ __host__ complex_t log_simple(complex_t z, FnVariables* variables){
    return complex_log(z, variables->n);
}

__device__ __host__ complex_t log_parametric(complex_t z, FnVariables* vars){
    return vars->z[0] * complex_log(vars->z[1] + z * vars->z[2], vars->n);
}

__device__ __host__ complex_t log_mul(complex_t z, FnVariables* vars){
    return z * complex_log(z, vars->n);
}

// ------------------------------------------------------------------------------------- trigonometric

__device__ __host__ inline complex_t sin_simple(complex_t z, FnVariables *) {
    return cuda::std::sin(z);
}

__device__ __host__ inline complex_t cos_simple(complex_t z, FnVariables *) {
    return cuda::std::cos(z);
}

__device__ __host__ inline complex_t tan_simple(complex_t z, FnVariables *) {
    return cuda::std::tan(z);
}

__device__ __host__ complex_t sin_parametric(complex_t z, FnVariables * vars) {
    return vars->z[0] * cuda::std::sin(vars->z[1] + z * vars->z[2]);
}

__device__ __host__ complex_t cos_parametric(complex_t z, FnVariables * vars) {
    return vars->z[0] * cuda::std::cos(vars->z[1] + z * vars->z[2]);
}

__device__ __host__ complex_t tan_parametric(complex_t z, FnVariables * vars) {
    return vars->z[0] * cuda::std::tan(vars->z[1] + z * vars->z[2]);
}

// ------------------------------------------------------------------------------------- hyperbolic

__device__ __host__ inline complex_t sinh_simple(complex_t z, FnVariables *) {
    return cuda::std::sinh(z);
}

__device__ __host__ inline complex_t cosh_simple(complex_t z, FnVariables *) {
    return cuda::std::cosh(z);
}

__device__ __host__ inline complex_t tanh_simple(complex_t z, FnVariables *) {
    return cuda::std::tanh(z);
}

__device__ __host__ complex_t sinh_parametric(complex_t z, FnVariables * vars) {
    return vars->z[0] * cuda::std::sinh(vars->z[1] + z * vars->z[2]);
}

__device__ __host__ complex_t cosh_parametric(complex_t z, FnVariables * vars) {
    return vars->z[0] * cuda::std::cosh(vars->z[1] + z * vars->z[2]);
}

__device__ __host__ complex_t tanh_parametric(complex_t z, FnVariables * vars) {
    return vars->z[0] * cuda::std::tanh(vars->z[1] + z * vars->z[2]);
}

// ------------------------------------------------------------------------------------- special

__device__ __host__ complex_t conjugate_z(complex_t z, FnVariables* variables){
    return variables->z[0] * cuda::std::conj(z);
}

__device__ __host__ complex_t fraction(complex_t z, FnVariables* variables){
    return (z*z - variables->z[0]) * cuda::std::pow(z - variables->z[1], 2) / (z*z + variables->z[2]);
}

__device__ __host__ complex_t fibonacci(complex_t z, FnVariables* variables){
    return (cuda::std::pow(PHI, z) - (cos(PI*z) * cuda::std::pow(PHI, -z))) * INV_SQRT_5;
}

double gamma_p_host[] = {
        1975.3739023578852322, -4397.3823927922428918, 3462.6328459862717019, -1156.9851431631167820,
        154.53815050252775060, -6.2536716123689161798, 0.034642762454736807441, -7.4776171974442977377e-7,
        6.3041253821852264261e-8, -2.7405717035683877489e-8, 4.0486948817567609101e-9 };

__device__ double gamma_p_device[] = {
        1975.3739023578852322, -4397.3823927922428918, 3462.6328459862717019, -1156.9851431631167820,
        154.53815050252775060, -6.2536716123689161798, 0.034642762454736807441,-7.4776171974442977377e-7,
        6.3041253821852264261e-8, -2.7405717035683877489e-8, 4.0486948817567609101e-9 };

/** @see https://en.wikipedia.org/wiki/Lanczos_approximation */
__device__ __host__ complex_t gamma_positive_half_plane(complex_t z) {
    z -= 1.0;
    complex_t A = 0.9999999999999999298;
    for(unsigned i=0; i<11; i++) {
        #ifdef __CUDA_ARCH__
        A += gamma_p_device[i] / (z + (double) i);
        #else
        A += gamma_p_host[i] / (z + (double) i);
        #endif
    }
    complex_t t = z + 8.5;
    return SQRT_2PI * cuda::std::pow(t, z+0.5) * cuda::std::exp(-t) * A;
}

__device__ __host__ complex_t gamma(complex_t z, FnVariables*) {
    complex_t y;
    if(z.real() >= 0.5) y = gamma_positive_half_plane(z);
    else y = PI / (cuda::std::sin(z*PI) * gamma_positive_half_plane(1.0 - z));
    if(cuda::std::abs(y.imag()) < DBL_EPSILON) y.imag(0.0);
    return y;
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

__device__ ComplexFunction_t d_fns[] = {
    polynomial1, polynomial2, polynomial3, polynomial_fact,
    int_power, real_power, complex_power,
    exp_simple, exp_parametric, exp_pow_int, exp_pow_real, zxp, pow_int_exp, pow_real_exp,
    log_simple, log_parametric, log_mul,
    sin_simple, sin_parametric, cos_simple, cos_parametric, tan_simple, tan_parametric,
    sinh_simple, sinh_parametric, cosh_simple, cosh_parametric, tanh_simple, tanh_parametric,
    conjugate_z, fraction, fibonacci, gamma
};

ComplexFunction_t h_fns[] = {
        polynomial1, polynomial2, polynomial3, polynomial_fact,
        int_power, real_power, complex_power,
        exp_simple, exp_parametric, exp_pow_int, exp_pow_real, zxp, pow_int_exp, pow_real_exp,
        log_simple, log_parametric, log_mul,
        sin_simple, sin_parametric, cos_simple, cos_parametric, tan_simple, tan_parametric,
        sinh_simple, sinh_parametric, cosh_simple, cosh_parametric, tanh_simple, tanh_parametric,
        conjugate_z, fraction, fibonacci, gamma
};

FunctionChoice strtofn(const char * str) {
    auto len = strlen(str);
    switch (len) {
        case 2:
            switch (bytes_to_uint<2>(str)) {
                case bytes_to_uint<2>("^n"):
                    return FunctionChoice::PowInt;
                case bytes_to_uint<2>("^r"):
                    return FunctionChoice::PowReal;
                case bytes_to_uint<2>("^c"):
                    return FunctionChoice::PowComplex;
                case bytes_to_uint<2>("ln"):
                    return FunctionChoice::LogSimple;
                default:
                    return FunctionChoice::NONE;
            }
        case 3:
            switch (bytes_to_uint<3>(str)) {
                case bytes_to_uint<3>("*ln"):
                    return FunctionChoice::LogMul;
                case bytes_to_uint<3>("$ln"):
                    return FunctionChoice::LogParametric;
                case bytes_to_uint<3>("exp"):
                    return FunctionChoice::ExpSimple;
                case bytes_to_uint<3>("zxp"):
                    return FunctionChoice::Zxp;
                case bytes_to_uint<3>("sin"):
                    return FunctionChoice::SinSimple;
                case bytes_to_uint<3>("cos"):
                    return FunctionChoice::CosSimple;
                case bytes_to_uint<3>("tan"):
                    return FunctionChoice::TanSimple;
                case bytes_to_uint<3>("fib"):
                    return FunctionChoice::Fibonacci;
                default:
                    return FunctionChoice::NONE;
            }
        case 4:
            switch (bytes_to_uint<4>(str)) {
                case bytes_to_uint<4>("$sin"):
                    return FunctionChoice::SinParametric;
                case bytes_to_uint<4>("$cos"):
                    return FunctionChoice::CosParametric;
                case bytes_to_uint<4>("$tan"):
                    return FunctionChoice::TanParametric;
                case bytes_to_uint<4>("sinh"):
                    return FunctionChoice::SinhSimple;
                case bytes_to_uint<4>("cosh"):
                    return FunctionChoice::CoshSimple;
                case bytes_to_uint<4>("tanh"):
                    return FunctionChoice::TanhSimple;
                case bytes_to_uint<4>("$exp"):
                    return FunctionChoice::ExpParametric;
                case bytes_to_uint<4>("frac"):
                    return FunctionChoice::Fraction;
                default:
                    return FunctionChoice::NONE;
            }
        case 5:
            switch (bytes_to_uint<5>(str)) {
                case bytes_to_uint<5>("poly1"):
                    return FunctionChoice::Poly1;
                case bytes_to_uint<5>("poly2"):
                    return FunctionChoice::Poly2;
                case bytes_to_uint<5>("poly3"):
                    return FunctionChoice::Poly3;
                case bytes_to_uint<5>("poly*"):
                    return FunctionChoice::PolyFact;
                case bytes_to_uint<5>("exp^n"):
                    return FunctionChoice::ExpPowInt;
                case bytes_to_uint<5>("exp^r"):
                    return FunctionChoice::ExpPowReal;
                case bytes_to_uint<5>("$sinh"):
                    return FunctionChoice::SinhParametric;
                case bytes_to_uint<5>("$cosh"):
                    return FunctionChoice::CoshParametric;
                case bytes_to_uint<5>("$tanh"):
                    return FunctionChoice::TanhParametric;
                case bytes_to_uint<5>("*conj"):
                    return FunctionChoice::Conjugate;
                case bytes_to_uint<5>("gamma"):
                    return FunctionChoice::Gamma;
                case bytes_to_uint<5>("binet"):
                    return FunctionChoice::Fibonacci;
                default:
                    return FunctionChoice::NONE;
            }
        case 6:
            switch (bytes_to_uint<6>(str)) {
                case bytes_to_uint<6>("^n*exp"):
                    return FunctionChoice::PowIntExp;
                case bytes_to_uint<6>("^r*exp"):
                    return FunctionChoice::PowRealExp;
                default:
                    return FunctionChoice::NONE;
            }
        default:
            return FunctionChoice::NONE;
    }
}

__host__ ComplexFunction_t get_function_host(FunctionChoice choice) {
    return h_fns[(int)choice];
}

__host__ ComplexFunction_t get_function_global(FunctionChoice choice) {
    ComplexFunction_t fn;
    cudaMemcpyFromSymbol(&fn, d_fns[(int)choice], sizeof(ComplexFunction_t));
    return fn;
}

__device__ ComplexFunction_t get_function_device(FunctionChoice choice) {
    return d_fns[(int)choice];
}