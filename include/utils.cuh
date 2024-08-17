#ifndef HPC_PROJECT_2024_UTILS_CUH
#define HPC_PROJECT_2024_UTILS_CUH

#include <cstdint>
#include <chrono>
#include "cuda/std/complex"

#define PI 3.1415926535897932384626433

#ifdef DEBUG
    #include <iostream>
    #define PRINT(X) std::cout << X;
    #define PRINTLN(X) std::cout << X << std::endl;
#else
    #define PRINT(X)
    #define PRINTLN(X)
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


template<unsigned N>
constexpr uint64_t str_to_num(const char str[N+1]) {
    static_assert(N < 9, "Number of characters cannot exceed 8");
    auto res = (uint64_t)str[0];
    for(unsigned i=1; i<N; i++) res += (uint64_t)str[i] << i*8;
    return res;
}

using complex_t = cuda::std::complex<double>;


struct CanvasAdapter {
    /** Size in pixel of the canvas */
    uint32_t width = 1920, height = 1080;
    /** Center complex point in the canvas */
    complex_t center = 0.0;
    /** How many pixel is the unitary distance */
    double scale = 100.0; // px^(-1)

    friend std::ostream& operator<<(std::ostream& os, CanvasAdapter& cv);

    /**
     * Converts a complex number into its pixel indexes
     * @param z input complex number
     * @return -1 if out of bounds, else the index of the pixel
     */
    #ifdef __CUDACC__
    __device__ __host__
    #endif
    int32_t where(complex_t z);
};

struct FnVariables {
    complex_t z[3] = {1.0, {0.0, 1.0}, {0.7071067811865476, 0.7071067811865476}};
    double x = PI;
    long n = 0;
};

struct EvolutionOptions {
    double speed_factor = 1.0, delta_time = 1e-9;
    float ms_per_frame = 50.0 / 3.0; // 60Hz
    uint32_t frame_count = 900; // 15s
    // default time-scale = 6e-8

    friend std::ostream& operator<<(std::ostream& os, EvolutionOptions& cv);
};

struct Configuration {
    const char * output = "plot.webp";
    FnVariables vars;
    CanvasAdapter canvas;
    EvolutionOptions evolution;
    unsigned long particle_distance = 10;
    unsigned long margin = 4;

    /**
     * Computes the portion of the Gauss plane where the particles begin
     * @param min pointer to the lower left complex number
     * @param max pointer to the upper right complex number
     */
    void bounds(complex_t * min, complex_t * max) const;

    /**
     * Computes the complete pixel width and height including the margin
     * @param width pointer to the width variable
     * @param height pointer to the height variable
     */
    void sizes(unsigned * width, unsigned * height) const;

    /**
     * Computes the number of particle to simulate
     * @return number of particles
     */
    uint32_t particle_number() const;
};

#endif //HPC_PROJECT_2024_UTILS_CUH
