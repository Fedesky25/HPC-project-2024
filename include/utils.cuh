//
// Created by feder on 06/04/2025.
//

#ifndef HPC_PROJECT_2024_UTILS_CUH
#define HPC_PROJECT_2024_UTILS_CUH

#include <cstdint>
#include <chrono>
#include <cuda_runtime_api.h>

#define PI 3.1415926535897932384626433


#define EXIT_IF(COND, MSG) if(COND) { std::cerr << MSG << std::endl; exit(1); }

#ifdef __PRETTY_FUNCTION__
    #define FN_NAME_HERE __PRETTY_FUNCTION__
#elif defined(_MSVC_LANG)
    #define FN_NAME_HERE __FUNCSIG__
#else
    #define FN_NAME_HERE "[unknown function signature]"
#endif

//#define FN_NAME_HERE ""

#define CATCH_CUDA_ERROR(EXPR) { \
    auto err = (EXPR);                \
    if(err) {                         \
        auto details = cudaGetErrorString(err); \
        std::cerr << "Cuda Error: " << details; \
        std::cerr << "\n in function ";         \
        std::cerr << FN_NAME_HERE;    \
        std::cerr << " (l:" << __LINE__ << ")" << std::endl; \
        exit(1);                      \
    }                                 \
}

#ifdef NDEBUG
    #define PRINT(X)
    #define PRINTLN(X)
#else
    #include <iostream>
    #define PRINT(X) std::cout << X;
    #define PRINTLN(X) std::cout << X << std::endl;
#endif


#if __cplusplus >= 201700L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201700L)
    #define CONSTEXPR_IF if constexpr
#else
// we hope compiler removes unused branch
    #define CONSTEXPR_IF if
    #if defined(__GNUC__) || defined(__MINGW32__) || defined(__MINGW64__)
        #warning C++ version is 14 or less
    #elif defined(_MSC_VER) || defined(__clang__)
        #pragma message("C++ version is 14 or less (" __c)
    #endif
    #include <algorithm>
#endif


#ifdef __CUDACC__
    #define BOTH __host__ __device__
#else
    #define BOTH
#endif


#if CUDART_VERSION >= 11020
    #include <cuda/std/complex>
    using complex_t = cuda::std::complex<double>;
    #define C_NORM cuda::std::norm
    #define C_ABS cuda::std::abs
#else
    #include "thrust/complex.h"
    using complex_t = thrust::complex<double>;
    #define C_NORM thrust::norm
    BOTH inline double C_ABS(const complex_t & z) {
        using namespace std;
        return hypot(z.real(), z.imag());
    }
#endif


#if CUDART_VERSION < 12000
#include <iostream>
// For some reason thsi is not defined in libcu++ 11.8
inline std::ostream& operator<<(std::ostream& stream, const complex_t& z) {
    return stream << '(' << z.real() << ',' << z.imag() << ')';
}
#endif


#define timers(N) std::chrono::steady_clock::time_point _tp[(N)<<1]; float t_elapsed;
#define tick(I) _tp[(I)<<1] = std::chrono::steady_clock::now();
#define tock(I, RATIO) { \
    _tp[1+((I)<<1)] = std::chrono::steady_clock::now(); \
    t_elapsed = (std::chrono::duration<float, RATIO>(_tp[1+((I)<<1)] - _tp[(I)<<1])).count(); \
}
#define tock_us(I) tock(I, std::micro)
#define tock_ms(I) tock(I, std::milli)
#define tock_s(I) tock(I, std::ratio<1>)

#endif //HPC_PROJECT_2024_UTILS_CUH
