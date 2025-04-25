//
// Created by feder on 06/04/2025.
//

#ifndef HPC_PROJECT_2024_UTILS_CUH
#define HPC_PROJECT_2024_UTILS_CUH

#include <cstdint>
#include <chrono>
#include <iostream>

#if __cplusplus < 201700L || (defined(_MSVC_LANG) && _MSVC_LANG < 201700L)
    // in lower versions of c++ size_t is not defined globally
    #include <cstddef>
#endif
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

inline void internal_print_cuda_err(cudaError_t err, const char * fn_name, size_t line) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n >> in function: "
              << fn_name << " [L:" << line << ']' << std::endl;
    exit(2);
}

#define CATCH_CUDA_ERROR(EXPR) { \
    auto _err = (EXPR);           \
    if(_err) internal_print_cuda_err(_err, FN_NAME_HERE, __LINE__); \
}

#ifdef NDEBUG
    #define PRINT(X)
    #define PRINTLN(X)
#else
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
// For some reason this is not defined in libcu++ 11.8
inline std::ostream& operator<<(std::ostream& stream, const complex_t& z) {
    return stream << '(' << z.real() << ',' << z.imag() << ')';
}
#endif


struct KernelSizes {
    unsigned grid, block;

    void cover(unsigned N);
    void warp_cover(unsigned N);

    static void set_SM();
    inline static unsigned get_SM() { return SM_count; }
private:
    static unsigned SM_count;
};


#define timers(N) std::chrono::steady_clock::time_point _tp[(N)<<1]; float t_elapsed;
#define tick(I) _tp[(I)<<1] = std::chrono::steady_clock::now();
#define tock(I, RATIO) { \
    _tp[1+((I)<<1)] = std::chrono::steady_clock::now(); \
    t_elapsed = (std::chrono::duration<float, RATIO>(_tp[1+((I)<<1)] - _tp[(I)<<1])).count(); \
}
#define tock_us(I) tock(I, std::micro)
#define tock_ms(I) tock(I, std::milli)
#define tock_s(I) tock(I, std::ratio<1>)


#define TIMEIT(VAR, BODY) { \
    auto start = std::chrono::steady_clock::now(); \
    BODY;                   \
    auto end = std::chrono::steady_clock::now();   \
    VAR += (std::chrono::duration<float, std::milli>(end-start)).count(); \
}


#define DEF_OPAQUE_FN(NAME, ARGS) \
    template void NAME<false> ARGS; \
    template void NAME<true> ARGS;\
    template<bool opaque>         \
    void NAME ARGS



#endif //HPC_PROJECT_2024_UTILS_CUH
