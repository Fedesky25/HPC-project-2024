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

struct PixelIndex {
    int32_t row, col;
    inline PixelIndex() : row(-1), col(-1) {}
    inline PixelIndex(int32_t r, int32_t c) : row(r), col(c) {};
    inline explicit operator bool () const { return row == -1 || col == -1; }
};

struct Canvas {
    /** Size in pixel of the canvas */
    uint32_t width = 1920, height = 1080;
    /** Center complex point in the canvas */
    complex_t center = 0.0;
    /** How many pixel is the unitary distance */
    double scale = 100.0; // px^(-1)

    friend std::ostream& operator<<(std::ostream& os, Canvas& cv);

    /**
     * Converts a complex number into its pixel indexes
     * @param z input complex number
     * @return pixel index
     */
    PixelIndex where(complex_t z) const;
};

struct FnVariables {
    complex_t z[3] = {1.0, {0.0, 1.0}, {0.7071067811865476, 0.7071067811865476}};
    double x = PI;
    long n = 0;
};

struct Configuration {
    const char * output = "plot.webp";
    FnVariables vars;
    Canvas canvas;
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

    /**
     * Computes the color the particle has given its speed
     * @param speed_squared square of the speed of the particle
     * @return the color hue
     */
    inline double color(double speed_squared) const {
        return 0.66667 / (1.0 + color_multiplier / speed_squared);
    }
};

#endif //HPC_PROJECT_2024_UTILS_CUH
