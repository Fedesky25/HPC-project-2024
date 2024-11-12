#include "tiles.cuh"
#include "lower_bound.cuh"
#include "sorter.cuh"


Tiles::Tiles(Configuration * config) {
    unsigned width, height;
    config->sizes(&width, &height);
    cover(width, height);
}

void Tiles::cover(unsigned int width, unsigned int height) {
    float r = sqrt((float) width / (float) height);
    cols = static_cast<uint_fast16_t>(32*r);
    rows = static_cast<uint_fast16_t>(32/r);
}

__global__ void compute_tile(
        uint32_t N, complex_t * particles, unsigned * tile_map,
        complex_t min, double hscale, double vscale, uint_fast16_t cols
) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= N) return;
    auto c = static_cast<unsigned>(hscale * (particles[i].real() - min.real()));
    auto r = static_cast<unsigned>(vscale * (particles[i].imag() - min.imag()));
    tile_map[i] = c + r*cols;
}

__global__ void compute_offset_per_tile(uint32_t N, unsigned int * tile_map, uint32_t * offsets) {
    auto tile = threadIdx.x + blockIdx.x * blockDim.x;
    offsets[tile] = lower_bound(tile, tile_map, N);
}

uint32_t * Tiles::sort(complex_t &min, complex_t &max, complex_t *particles, uint32_t N) const {
    timers(2) tick(0)
    float times[4];
    uint32_t * offsets;
    auto hscale = cols / (max.real() - min.real());
    auto vscale = rows / (max.imag() - min.imag());
    auto tile_count = total();
    auto M = 1 + (N - 1)/tile_count;
    tick(1)
    KVSorter<unsigned, complex_t> tile_map(N, particles);
    cudaMalloc(&offsets, (1+tile_count) * sizeof(uint32_t));
    cudaDeviceSynchronize();
    tock_us(1) times[0] = t_elapsed; tick(1)
    compute_tile<<<M, tile_count>>>(N, particles, tile_map.keys(), min, hscale, vscale, cols);
    cudaDeviceSynchronize();
    tock_us(1) times[1] = t_elapsed; tick(1)
    // thrust::sort_by_key(thrust::device, tile_map, tile_map + N, particles);
    tile_map.sort();
    cudaDeviceSynchronize();
    tock_us(1) times[2] = t_elapsed; tick(1)
    auto block_dim = rows, grid_dim = cols;
    if(block_dim > grid_dim) {
        // for sure one between rows and cols is less than 32 i.e. the dimension of a warp
        block_dim = cols;
        grid_dim = rows;
    }
    compute_offset_per_tile<<<grid_dim, block_dim>>>(N, tile_map.keys(), offsets);
    cudaMemcpy(offsets + tile_count, &N, sizeof(uint32_t), cudaMemcpyHostToDevice);
    tock_us(1) times[3] = t_elapsed;
    tock_us(0)
    float m = 100.0f / t_elapsed;
    std::cout.precision(1);
    std::cout << "Particles sorted by tile in " << std::fixed << t_elapsed << "us {alloc: " << times[0]*m
              << ", comp: " << times[1]*m << ", sort: " << times[2]*m << ", offsets: " << times[3]*m << '}' << std::endl;
    return offsets;
}
