#include "tiles.cuh"
#include "thrust/sort.h"
#include "lower_bound.cuh"

Tiles::~Tiles() {
    delete[] points;
    delete[] counts;
}

void Tiles::cover(unsigned int width, unsigned int height) {
    unsigned rev = 0;
    if(height > width) {
        rev = width;
        width = height;
        height = rev;
    }
    float min = INFINITY;
    float ratio = (float) width / (float) height;
    for(unsigned r=1; r <= 32; r++) {
        auto c = static_cast<unsigned>(std::round(ratio*r));
        while(r * c > 1024) c--;
        auto d = std::abs((float) c / (float) r - ratio);
        if(d <= min) {
            rows = r;
            cols = c;
            min = d;
        }
    }
    if(rev) {
        rev = rows;
        rows = cols;
        cols = rev;
    }
    delete[] counts;
    auto N = total();
    counts = new uint_fast16_t [N];
    for(uint_fast16_t i=0; i<N; i++) counts[i] = 0;
}

__global__ compute_tile(
        uint32_t N, complex_t * particles, uint_fast16_t * tile_map,
        complex_t min, double hscale, double vscale, uint_fast16_t cols
) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= N) return;
    auto c = static_cast<uint_fast16_t>(hscale * (particles[i].real() - min.real()));
    auto r = static_cast<uint_fast16_t>(vscale * (particles[i].imag() - min.imag()));
    tile_map[i] = c + r*cols;
}

__global__ compute_particle_per_tile(uint32_t N, uint_fast16_t * tile_map, uint64_t * count) {
    count[threadIdx.x] = 0;
    auto i = lower_bound(threadIdx.x, tile_map, N);
    while(tile_map[i] == threadIdx.x) {
        count[threadIdx.x]++;
        i++;
    }

}

void Tiles::sort(complex_t &min, complex_t &max,
                 complex_t *particles, uint32_t N,
                 uint_fast16_t **tile_map_ptr, uint32_t **count_per_tile_ptr
) const {
    cudaMalloc(tile_map_ptr, N * sizeof(uint_fast16_t));
    auto tile_map = *tile_map_ptr;
    cudaMalloc(count_per_tile_ptr, total() * sizeof(uint64_t));
    auto count_per_tile = *count_per_tile_ptr;
    auto hscale = cols / (max.real() - min.real());
    auto vscale = rows / (max.imag() - min.imag());
    auto M = 1 + (N - 1)/total();
    compute_tile<<<M, total()>>>(N, particles, tile_map, min, hscale, vscale, cols);
    thrust::sort_by_key(thrust::device, tile_map, tile_map + N, particles);
    compute_particle_per_tile<<<1, total()>>>(N, tile_map, count_per_tile);
}
