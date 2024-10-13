#ifndef HPC_PROJECT_2024_UTILS_CUH
#define HPC_PROJECT_2024_UTILS_CUH

#include <cstdint>
#include <chrono>
#include <iostream>
#include "color.cuh"

#define PI 3.1415926535897932384626433

#ifdef DEBUG
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

#if CUDART_VERSION >= 11000
    #include <cuda/std/complex>
    using complex_t = cuda::std::complex<double>;
#else
    class complex_t {
    private:
        double _re, _im;
    public:
        inline double real() const { return _re; }
        inline double imag() const { return _im; }
        inline void real(double v) { _re = v; }
        inline void imag(double v) { _im = v; }
        inline complex_t() : _re(0), _im(0) {}
        inline explicit complex_t(double v) : _re(v), _im(0) {}
        inline complex_t(double re, double im) : _re(re), _im(im) {}
    };
#endif

#if CUDART_VERSION < 12000
// For some reason thsi is not defined in libcu++ 11.8
inline std::ostream& operator<<(std::ostream& stream, const complex_t& z) {
    return stream << '(' << z.real() << ',' << z.imag() << ')';
}
#endif


struct CanvasAdapter {
    /** Center complex point in the canvas */
    complex_t center = 0.0;
    /** How many pixel is the unitary distance */
    double scale = 100.0; // px^(-1)
    /** Size in pixel of the canvas */
    uint32_t width = 1920, height = 1080;

    friend std::ostream& operator<<(std::ostream& os, CanvasAdapter& cv);

    /**
     * Converts a complex number into its pixel indexes
     * @param z input complex number
     * @return -1 if out of bounds, else the index of the pixel
     */
    BOTH int32_t where(complex_t z);
};

struct FnVariables {
    complex_t z[3] = {1.0, {0.0, 1.0}, {0.7071067811865476, 0.7071067811865476}};
    double x = PI;
    long n = 0;
};

struct EvolutionOptions {
    double speed_factor = 1.0, delta_time = 2e-3;
    int32_t frame_count = 900; // 15s
    int32_t life_time = 600;
    uint32_t frame_rate = 60;
    // default time-scale = 0.12

    friend std::ostream& operator<<(std::ostream& os, EvolutionOptions& cv);
};

enum class ExecutionMode { Serial, OpenMP, GPU };

struct Configuration {
    FnVariables vars;
    CanvasAdapter canvas;
    EvolutionOptions evolution;
    RGBA background{21*div_255, 21*div_255, 25*div_255, 1};
//    FixedHSLA background{ IC_16b, icenc(0.06), icenc(0.15), ICE_1 };
    const char * output = "plot.raw";
    unsigned long particle_distance = 10;
    unsigned long margin = 4;
    unsigned long lloyd_iterations = 8;
    ExecutionMode mode = ExecutionMode::GPU;

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

/**
 * Copies the provided data into the device memory and returns a device pointer to such copy
 * @tparam T
 * @param obj host pointer to data
 * @return device pointer to a copy of obj
 */
template<class T>
T* devicify(T* obj) {
    T* d_obj;
    cudaMalloc(&d_obj, sizeof(T));
    cudaMemcpy(d_obj, obj, sizeof(T), cudaMemcpyHostToDevice);
    return d_obj;
}

#endif //HPC_PROJECT_2024_UTILS_CUH
