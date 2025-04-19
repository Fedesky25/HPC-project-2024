#ifndef HPC_PROJECT_2024_TILES_CUH
#define HPC_PROJECT_2024_TILES_CUH

#include "config.cuh"

struct Tiles {
    uint_fast16_t rows = 0, cols = 0;

    inline Tiles() = default;
    explicit Tiles(Configuration * config, float target = 1024.f);

    /** @returns total number of tiles */
    inline auto total() const { return rows * cols; }

    /**
     * Computes the best way to cover the provided (pixel) width and height with tiles as
     * square as possible and as close as possible to 1024 in number
     * @param width
     * @param height
     * @param target target number of tiles
     */
    void cover(unsigned width, unsigned height, float target = 1024.f);

    /**
     *
     * @param min lower-left vertex
     * @param max upper-right vertex
     * @param particles device array of particles to be sorted
     * @param N number of particles
     * @returns device pointer to array containing the index of the first particle of each tile
     */
    uint32_t * sort(complex_t& min, complex_t& max, complex_t* particles, uint32_t N) const;
};


#endif //HPC_PROJECT_2024_TILES_CUH
