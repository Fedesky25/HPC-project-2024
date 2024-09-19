#include "canvas.cuh"
#include "thrust/extrema.h"

void ARGB::print(FILE *file) const {
    fputc(((a&0xf0)>>4) + 'a', file);
    fputc( (a&0x0f)     + 'a', file);
    fputc(((r&0xf0)>>4) + 'a', file);
    fputc( (r&0x0f)     + 'a', file);
    fputc(((g&0xf0)>>4) + 'a', file);
    fputc( (g&0x0f)     + 'a', file);
    fputc(((b&0xf0)>>4) + 'a', file);
    fputc( (b&0x0f)     + 'a', file);
}

void ARGB::print_base64(FILE *file) const {
    int c = (r & 0x3f) + ';';
    fputc(c, file);
    c = ((r & 0xc0) >> 6) + ((b & 0x0f) << 2) + ';';
    fputc(c, file);
    c = ((b & 0xf0) >> 4) + (g & 0x03) + ';';
    fputc(c, file);
    c = ((c & 0xfc) >> 2) + ';';
    fputc(c, file);
}

__device__ __host__ int32_t CanvasPixel::time_distance(int32_t time, int32_t frame_count) const {
    int32_t mask = ((((int32_t) age) + 1) >> 16) - 1; // 0b00000000 if UINT16_MAX, 0xffffffff otherwise
    int32_t diff = (frame_count + time - (int32_t) age) % frame_count;
    return (mask & diff) + ((~mask) & UINT16_MAX);
}

__device__ __host__ int32_t CanvasPixel::time_distance_divergent(int32_t time, int32_t frame_count) const {
    if(age == UINT16_MAX) return UINT16_MAX;
    int32_t res = time - (int32_t) age;
    if(res < 0) res += frame_count;
    return res;
}

__device__ __host__ bool CanvasPixel::update_age(uint16_t _age) {
    bool ok = true;
    if(age == UINT16_MAX) age = _age;
    else if(age + multiplicity + 1 == _age) multiplicity++;
    else ok = false;
    return ok;
}

__device__ __host__ void CanvasPixel::set_color(double square_speed, double factor) {
    hue = static_cast<uint16_t>(IC_16b * square_speed / (square_speed + factor));
}

__device__ __host__ uint32_t CanvasPixel::get_color(int32_t delta, int32_t frame_count, const FixedHSLA * background) const {
    if(delta <= multiplicity) return HSLA_to_RGBA(hue, icenc(0.55f), icenc(0.55f), 0);
    else {
        FixedFraction f(delta, frame_count);
        FixedHSLA color{hue, icenc(0.55f), icenc(0.55f), 0};
        color.mixWith(*background, f);
        return color.toRGBA();
    }
}

Canvas * create_canvas_host(uint32_t count, CanvasAdapter * adapter) {
    timers(1) tick(0)
    auto p = new Canvas [count];
    auto area = adapter->height * adapter->width;
    #pragma omp parallel for schedule(static,1)
    for(int32_t i=0; i<count; i++) {
        auto c = new CanvasPixel [area];
        for(uint32_t j=0; j<area; j++) c[j].reset();
        p[i] = c;
    }
    tock_ms(0)
    std::cout << "Initialized " << (((area * sizeof(CanvasPixel) - 1) >> 20)+1) << "MB for " << count
              << " canvases in " << t_elapsed << "ms" << std::endl;
    return p;
}

__global__ void init_canvas_array(Canvas * array, uint32_t len) {
    auto canvas = array[blockIdx.x];
    for(unsigned i=threadIdx.x; i<len; i+=blockDim.x) canvas[i].reset();
}

Canvas * create_canvas_device(uint32_t count, CanvasAdapter * adapter) {
    timers(1) tick(0)
    auto array_bytes = count * sizeof(Canvas);
    auto len = adapter->width * adapter->height;
    auto canvas_bytes = len * sizeof(CanvasPixel);
    auto h_array = (Canvas*) malloc(array_bytes);
    for(uint32_t i=0; i<count; i++) cudaMalloc(&h_array[i], canvas_bytes);
    Canvas * d_array;
    cudaMalloc(&d_array, array_bytes);
    cudaMemcpy(d_array, h_array, array_bytes, cudaMemcpyHostToDevice);
    free(h_array);
    init_canvas_array<<<count, 1024>>>(d_array, len);
    tock_ms(0)
    std::cout << "Initialized " << (((canvas_bytes-1) >> 20)+1) << "MB for " << count
              << " canvases in " << t_elapsed << "ms" << std::endl;
    return d_array;
}

uint32_t get_canvas_count_serial(const uint32_t * offsets, uint32_t tiles) {
    timers(1) tick(0)
    auto size = (tiles+1) * sizeof(uint32_t);
    auto h_ofs = (uint32_t*) malloc(size);
    cudaMemcpy(h_ofs, offsets, size, cudaMemcpyDeviceToHost);
    uint32_t max_c = h_ofs[1], v;
    for(int i=1; i<tiles; i++) {
        v = h_ofs[i+1] - h_ofs[i];
        if(v > max_c) max_c = v;
    }
    free(h_ofs);
    tock_us(0)
    std::cout << "Canvas number found in " << t_elapsed << "us" << std::endl;
    return max_c;
}

uint32_t get_canvas_count_device(uint32_t * count_per_tile, uint32_t tiles) {
    timers(1) tick(0)
    uint32_t max;
    auto max_ptr = thrust::max_element(thrust::device, count_per_tile, count_per_tile+tiles);
    cudaMemcpy(&max, max_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    tock_us(0)
    std::cout << "Canvas number found in " << t_elapsed << "us" << std::endl;
    return max;
}
