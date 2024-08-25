#ifndef HPC_PROJECT_2024_CANVAS_CUH
#define HPC_PROJECT_2024_CANVAS_CUH

#include "utils.cuh"

struct ARGB {
    uint8_t a, r, g, b;
    inline ARGB() = default;
    inline ARGB(uint8_t red, uint8_t green, uint8_t blue) : a(0), r(red), g(green), b(blue) {}

    /**
     * Print the ARGB values in hexadecimal format
     * @param file output file
     */
    void print(FILE * file) const;

    /**
     * Prints the RGB values using 4 characters (base64 variant encoding)
     * @param file output file
     */
    void print_base64(FILE * file) const;
};

struct CanvasPixel {
    /** @return whether a particle passed through the pixel  */
    explicit inline operator bool() const { return age != UINT16_MAX; }

    /** Resets the pixel as if a particle never passed through it */
    __device__ __host__ inline void reset() {
        age = UINT16_MAX;
        multiplicity = 0;
    }

    /**
     * Updates the age of this pixel. Multiplicity is increased if necessary
     * @param age value of the age
     * @returns whether the operation was successful
     */
    __device__ __host__ bool update_age(uint16_t age);

    /**
     * Computes and saves the hue given the speed
     * @param square_speed
     * @param factor
     */
    __device__ __host__ void set_color(double square_speed, double factor);

    /**
     * Computes the color the pixel has at the given time
     * @todo take total lifetime as argument
     * @param time time index
     * @return color of the pixel now
     */
    __device__ __host__ ARGB get_color(uint16_t time) const;

private:
    ARGB color;
    uint16_t age = UINT16_MAX, multiplicity = 0;
};

using Canvas = CanvasPixel*;

/**
 * Creates and initializes an array of canvas on the host
 * @param count number of canvas to create
 * @param adapter
 * @return host pointer to array of canvas
 */
Canvas * create_canvas_host(uint32_t count, CanvasAdapter * adapter);

/**
 * Creates and initializes an array of canvas on the device
 * @param count number of canvas to create
 * @param adapter
 * @return device pointer to array of canvas
 */
Canvas * create_canvas_device(uint32_t count, CanvasAdapter * adapter);

/**
 * Computes on the CPU the number of canvas to create
 * @param offsets array of first index of particle in each tile
 * @param tiles number of tiles
 * @return canvas number
 */
uint32_t get_canvas_count_serial(const uint32_t * offsets, uint32_t tiles);

/**
 * Computes on the GPU the number of canvas to create
 * @param count_per_tile particle count per tile
 * @param tiles number of tiles
 * @return canvas number
 */
uint32_t get_canvas_count_device(uint32_t * count_per_tile, uint32_t tiles);


#endif //HPC_PROJECT_2024_CANVAS_CUH
