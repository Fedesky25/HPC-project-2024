#ifndef HPC_PROJECT_2024_UTILS_CUH
#define HPC_PROJECT_2024_UTILS_CUH

#include <cstdint>
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

template<unsigned N>
constexpr uint64_t str_to_num(const char str[N+1]) {
    static_assert(N < 9, "Number of characters cannot exceed 8");
    auto res = (uint64_t)str[0];
    for(unsigned i=1; i<N; i++) res += (uint64_t)str[i] << i*8;
    return res;
}

using complex_t = cuda::std::complex<double>;

struct CanvasPixel {
    uint16_t age = -1, multiplicity = 0;

    /**
     * Computes and saves the hue given the speed
     * @param square_speed
     * @param factor
     */
    __device__ __host__ inline void set_color(double square_speed, double factor) {
        hue = static_cast<uint16_t>(65536.0 * square_speed / (square_speed + factor));
        // 2^(16) / (1 + factor/square_speed)
    }

    /**
     * Computes the hue
     * @return hue in the range [0,1]
     */
    __device__ __host__ inline float get_color() const {
        return (float)hue * 1.0172526041666666e-5f;
        // v / 2^(16) * (2/3)
    }

private:
    uint16_t hue = 0;
};

using Canvas = CanvasPixel*;

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
    int32_t where(complex_t z) const;

    /**
     * Allocates
     * @tparam GPU
     * @param count
     * @return
     */
    template<bool GPU>
    Canvas * create_proto_canvas(uint32_t count) {
        Canvas * array;
        if constexpr (GPU) {
            cudaMalloc(&array, count * sizeof(Canvas));
        } else {
            array = (Canvas*) malloc(count * sizeof(Canvas));
        }
        uint32_t bytes = width * height * sizeof(CanvasPixel);
        for(uint32_t i=0; i < count; i++) {
            if constexpr (GPU) {
                cudaMalloc(array+i, bytes);
            } else {
                array[i] = (Canvas) malloc(bytes);
            }
        }
        return array;
    }
};

struct FnVariables {
    complex_t z[3] = {1.0, {0.0, 1.0}, {0.7071067811865476, 0.7071067811865476}};
    double x = PI;
    long n = 0;
};

struct Configuration {
    const char * output = "plot.webp";
    FnVariables vars;
    CanvasAdapter canvas;
    double color_multiplier = 1.0;
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
