#include "tiles.cuh"
#include "lower_bound.cuh"
#include "sorter.cuh"
#include <iomanip>


Tiles::Tiles(Configuration * config, float target) {
    unsigned width, height;
    config->sizes(&width, &height);
    cover(width, height, target);
}

void Tiles::cover(unsigned int width, unsigned int height, float target) {
    float base = sqrt(target);
    float ratio = sqrt((float) width / (float) height);
    cols = std::round(base * ratio);
    rows = std::round(base / ratio);

    while(cols * rows > 1024) {
        if(cols > rows) cols--;
        else rows--;
    }
}

__global__ void compute_tile(
        uint32_t N, complex_t * particles, unsigned * tile_map,
        complex_t min, double hscale, double vscale,
        uint_fast16_t cols, uint_fast16_t rows
) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= N) return;
    auto c = static_cast<unsigned>(hscale * (particles[i].real() - min.real()));
    auto r = static_cast<unsigned>(vscale * (particles[i].imag() - min.imag()));
    tile_map[i] = ((c&1)+2*(r&1))*rows*cols + (r>>1)*cols + (c>>1);
}

__global__ void compute_offset_per_tile(uint32_t N, unsigned int * tile_map, uint32_t * offsets) {
    auto tile = threadIdx.x + blockIdx.x * blockDim.x;
    auto os = lower_bound(tile, tile_map, N);
    // os == N when tile is not inside tile_map, i.e. tile value is out of range
    if(os < N) offsets[tile] = os;
}

uint32_t * Tiles::sort(complex_t &min, complex_t &max, complex_t *particles, uint32_t N) const {
    timers(2) tick(0)
    KernelSizes sz;
    float times[4];
    uint32_t * offsets;
    auto hscale = 2 * cols / (max.real() - min.real());
    auto vscale = 2 * rows / (max.imag() - min.imag());
    auto tile_count = total();
    tick(1)
    KVSorter<unsigned, complex_t> tile_map(N, particles);
    cudaMalloc(&offsets, (1+4*tile_count) * sizeof(uint32_t));
    cudaDeviceSynchronize();
    tock_us(1) times[0] = t_elapsed; tick(1)
    sz.warp_cover(N);
    compute_tile<<<sz.grid, sz.block>>>(N, particles, tile_map.keys(), min, hscale, vscale, cols, rows);
    CATCH_CUDA_ERROR(cudaGetLastError())
    cudaDeviceSynchronize();
    tock_us(1) times[1] = t_elapsed; tick(1)
    // thrust::sort_by_key(thrust::device, tile_map, tile_map + N, particles);
    tile_map.sort();
    cudaDeviceSynchronize();
    tock_us(1) times[2] = t_elapsed; tick(1)
    sz.cover(4*tile_count);
    compute_offset_per_tile<<<sz.grid, sz.block>>>(N, tile_map.keys(), offsets);
    CATCH_CUDA_ERROR(cudaGetLastError())
    CATCH_CUDA_ERROR(cudaMemcpy(offsets + 4*tile_count, &N, sizeof(uint32_t), cudaMemcpyHostToDevice))
    tock_us(1) times[3] = t_elapsed;
    tock_us(0)
    if(verbose) std::cout << "Tiles-covering kernel sizes: " << sz.grid << 'x' << sz.block
                          << " = " << sz.size() << '/' << (4*tile_count) << '\n';
    float m = 100.0f / t_elapsed;
    std::cout.precision(1);
    std::cout << "Particles sorted by tile in " << std::fixed << t_elapsed << "us (allocation: "
              << std::fixed << std::setprecision(1) << times[0]*m << "%, compute indexes: "
              << std::fixed << std::setprecision(1) << times[1]*m << "%, sort: "
              << std::fixed << std::setprecision(1) << times[2]*m << "%, compute offsets: "
              << std::fixed << std::setprecision(1) << times[3]*m << "%)" << std::endl;
    return offsets;
}

WHEN_OK(
void tiles_print_regs() {
    cudaFuncAttributes attrs;
    cudaFuncGetAttributes(&attrs, &compute_tile);
    std::cout << " - compute_tile: " << attrs.numRegs << '\n';
    cudaFuncGetAttributes(&attrs, &compute_offset_per_tile);
    std::cout << " - compute_offset_per_tile: " << attrs.numRegs << '\n';
})
