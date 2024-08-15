#ifndef HPC_PROJECT_2024_TILES_CUH
#define HPC_PROJECT_2024_TILES_CUH

#include "utils.cuh"

struct Tiles {
    complex_t * points = nullptr;
    uint_fast16_t * counts = nullptr;
    uint32_t max_count = 0;
    uint_fast16_t rows = 0, cols = 0;

    ~Tiles();
    inline Tiles() = default;
    inline Tiles(unsigned width, unsigned height) { cover(width, height); }

    /** @returns total number of tiles */
    inline auto total() const { return rows * cols; }

    /**
     * Computes the best way to cover the provided (pixel) width and height with tiles as
     * square as possible and as close as possible to 1024 in number
     * @param width
     * @param height
     */
    void cover(unsigned width, unsigned height);

    /**
     *
     * @param min lower-left vertex
     * @param max upper-right vertex
     * @param particles device array of particles to be sorted
     * @param N number of particles
     * @param tile_map_ptr pointer to sorted device array mapping particles to the belonging tile
     * @param count_per_tile_ptr pointer to device array counting number of particle in each tile
     */
    void sort(
            complex_t& min, complex_t& max,
            complex_t* particles, uint32_t N,
            uint_fast16_t ** tile_map_ptr, uint32_t ** count_per_tile_ptr) const;
};


#endif //HPC_PROJECT_2024_TILES_CUH
