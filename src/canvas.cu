// The possible hue of particle span the range [0, 2/3] and we encode it in an uint16_t
// This means that 2/3 is encoded by UINT16_MAX = 2^16 - 1 = 65535
// For consistency, all color channels which span [0,1] are encoded in the range [0, 3/2 * 65535] ~ [0, 98302]
// using 17 bits of an int32_t (the sign allows for differences without issues)
// When converting this 17-bit channel into a classical 8-bit channel one must
// multiply the number by 2/3 * (2^8 - 1)/(2^16 - 1) = 2/(3 * 257) = 2/771

#include "canvas.cuh"
#include "thrust/extrema.h"

#define SATURATION 0.55
#define LIGHTNESS 0.55

#define BG_R 21
#define BG_G 21
#define BG_B 25

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

__device__ __host__ bool CanvasPixel::update_age(uint16_t _age) {
    bool ok = true;
    if(age == UINT16_MAX) age = _age;
    else if(age + multiplicity + 1 == _age) multiplicity++;
    else ok = false;
    return ok;
}

__device__ __host__ uint8_t fixed_pqt_to_component(int32_t p, int32_t q, int32_t t) {
    if(t < 0) t += 98302;
    else if(t > 98302) t -= 98302;
    int32_t result = p;
    if(t < 0x4000) result += 6 * (q-p) * t;
    else if(t < 49152) result = q; // 49152 = 3 * 2^14
    else if(t < 0xffff) result += 6 * (q-p) * (0xffff - t);
    return rounding_division(result << 1, 771);
}

/**
 * Transform an HSLA value using 17-bit encoding in range [0, 98302] to the 8-bit encoded RGBA value
 * @see https://stackoverflow.com/questions/2353211/hsl-to-rgb-color-conversion
 * @param h hue
 * @param s saturation
 * @param l lightness
 * @return RGBA value as a uint32_t
 */
__device__ __host__ uint32_t fixed_HSLA_to_RGBA(int32_t h, int32_t s, int32_t l, int32_t alpha) {
    uint8_t bgra[4];
    bgra[3] = rounding_division(alpha << 1, 771);
    if(s == 0) {
        bgra[0] = bgra[1] = bgra[2] = rounding_division(l << 1, 771);
    }
    else {
        // S*(L<0.5? L : 1-L)  =>  s * (l<49152 ? l : 98302 - l) / 98302
        // but the product may exceed the value represented by int32_t i.e. 2^31 - 1
        // e.g. s=98302, l=49151  =>  4_831_641_602 > 2_147_483_648 = 2^31 - 1
        // therefore pre-divide both side by 2 and finally divide by 3/2*65535/4 = 24575.625
        auto q = l + (s >> 1) * ((l < 49152 ? l : 98302-l) >> 1) * 2 / 24576;
        auto p = 2 * l - q;
        bgra[0] = fixed_pqt_to_component(p, q, h - 32767);    // blue
        bgra[1] = fixed_pqt_to_component(p, q, h);            // green
        bgra[2] = fixed_pqt_to_component(p, q, h + 32767);    // red
    }
    uint32_t result = *((uint32_t*)bgra);
    return result;
}

__device__ __host__ void CanvasPixel::set_color(double square_speed, double factor) {
    hue = static_cast<uint16_t>(UINT16_MAX * square_speed / (square_speed + factor));
}

__device__ __host__ uint32_t CanvasPixel::get_color(int32_t time, int32_t frame_count) const {
    uint8_t color[4] = { BG_B, BG_G, BG_R, 0 };
    auto delta = time - age;
    if(delta < 0) delta += frame_count;
    // 0.55 * 3/2 * 65535 = 54066.375
    if(delta <= multiplicity) return fixed_HSLA_to_RGBA(hue, 54066, 54066, 0);
    else {
        // Background = rgb(21,21,25) = hsl(2/3, 0.087, 0.09) = HSL(65535, 8552, 8847)
        // TODO
    }
    uint32_t result = *((uint32_t*)color);
    return result;
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
    auto count =  1 + ((len - 1) >> 10);
    canvas += count * threadIdx.x;
    count -= (threadIdx.x == 1023) * (len & 1023);
    for(uint32_t i=0; i<count; i++) canvas[i].reset();
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
