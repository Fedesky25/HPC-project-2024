#ifndef HPC_PROJECT_2024_CONFIG_CUH
#define HPC_PROJECT_2024_CONFIG_CUH

#include <cstdint>
#include <chrono>
#include <iostream>
#include "color.cuh"
#include "complex_functions.cuh"


template<unsigned N>
class Timers {
public:
    __forceinline__ void tick_(unsigned index = 0) { start[index] = std::chrono::steady_clock::now(); }

    template<class Ratio = std::ratio<1>>
    __forceinline__ void tock_(float & v, unsigned index = 0) {
        static_assert(std::_Is_ratio_v<Ratio>);
        auto end = std::chrono::steady_clock::now();
        v = (std::chrono::duration<float, Ratio>(end - start[index])).count();
    }

    template<class Ratio = std::ratio<1>>
    __forceinline__ void tock_tick(float & v, unsigned index = 0) {
        static_assert(std::_Is_ratio_v<Ratio>);
        auto end = std::chrono::steady_clock::now();
        v = (std::chrono::duration<float, Ratio>(end - start[index])).count();
        start[index] = std::chrono::steady_clock::now();
    }
private:
    std::chrono::steady_clock::time_point start[N];
};

template<unsigned N>
constexpr uint64_t str_to_num(const char str[N+1]) {
    static_assert(N < 9, "Number of characters cannot exceed 8");
    auto res = (uint64_t)str[0];
    for(unsigned i=1; i<N; i++) res += (uint64_t)str[i] << i*8;
    return res;
}

extern int verbose;

struct CanvasAdapter {
    /** Center complex point in the canvas */
    complex_t center = 0.0;
    /** How many pixel is the unitary distance */
    double scale = 100.0; // px^(-1)
    /** Size in pixel of the canvas */
    int32_t width = 1920, height = 1080;

    friend std::ostream& operator<<(std::ostream& os, CanvasAdapter& cv);

    /**
     * Converts a complex number into its pixel indexes
     * @param z input complex number
     * @return -1 if out of bounds, else the index of the pixel
     */
    BOTH int32_t where(complex_t z);
};

struct EvolutionOptions {
    double speed_factor = 1.0, delta_time = 2e-3;
    int32_t frame_count = 900; // 15s
    int32_t life_time = 600;
    int32_t frame_rate = 60;
    // default time-scale = 0.12

    friend std::ostream& operator<<(std::ostream& os, EvolutionOptions& cv);
};

enum class ExecutionMode { Serial, OpenMP, GPU };

struct Configuration {
    FnVariables vars;
    CanvasAdapter canvas;
    EvolutionOptions evolution;
    YUVA background{0.186f, 0.02f, 0.f, 1};
//    FixedHSLA background{ IC_16b, icenc(0.06), icenc(0.15), ICE_1 };
    const char * output = "plot.mp4";
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

#endif //HPC_PROJECT_2024_CONFIG_CUH
