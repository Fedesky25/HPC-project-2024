#ifndef HPC_PROJECT_2024_TILES_HPP
#define HPC_PROJECT_2024_TILES_HPP

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
     * Groups the N uniformly distributes particles into the tiles
     * @param min lower-left corner
     * @param max upper-right corner
     * @param particles array of particles
     * @param N number of particles
     */
    void distribute(complex_t& min, complex_t& max, complex_t * particles, uint64_t N);
};


#endif //HPC_PROJECT_2024_TILES_HPP
