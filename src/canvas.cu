// The possible hue of particle span the range [0, 2/3] and we encode it in an uint16_t
// The integer value M representing 2/3 is chosen such that
//  - 3/2*M, which encodes 1, should be an exact int
//  - 3/4*M, which encodes 0.5, should be an exact int
//  - the down-conversion to a 8-bit range [0,255] is easy
// Thus we choose M = 255 * 256 = 65280  =>  3/2*M = 97920
// The down-conversion is easy since x -> 255/97920*x = x / (3*128) = (x>>7)/3

#include "canvas.cuh"
#include "thrust/extrema.h"

#define MANUAL_DIV_3 0
#define INT_FOR_1       97920
#define INT_FOR_2over3  65280
#define INT_FOR_1over2  48960
#define INT_FOR_1over3  32640
#define INT_FOR_1over4  24480
#define INT_FOR_1over6  16320
#define INT_FOR_1over12  8160

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

/**
 * Perform the rounded division by 3*128
 * @param x integer value between 0 and 97920
 * @return
 */
__device__ __host__ inline uint8_t reduce_to_255(int32_t x) {
    #if MANUAL_DIV_3
    // x/(128*3) --> (2x/(3*128) + 1)/2 = (x/(3*64) + 1)>>1;
    // since 1/3 = 1/4 + 1/16 + 1/64 + 1/256 + 1/1024 + ...
    // divide by 3 up with powers up to 4^6
    // because 97920 / 4^6 = 97920 >> 12 = 23 < 2^6 = 64
    // and     97920 / 4^5 = 97920 >> 10 = 95 > 2^6 = 64
    int32_t q, y= 0;
    x = x >> 6;
    for(int i=0; i<6; i++) {
        q = x >> 2;
        x = q + (x&3);
        y += q;
    }
    return (y+1) >> 1;
    #else
    return (x/192 + 1) >> 1;
    #endif
}

__device__ __host__ uint8_t fixed_ldt_to_component(int32_t l, int32_t d, int32_t t) {
    // maximum value for q = 92920*(1 + 1/2) = 146880
    // maximum value for p =
    if(t < 0) t += INT_FOR_1;
    else if(t > INT_FOR_1) t -= INT_FOR_1;
    int32_t result = l - d;
    if(t < INT_FOR_1over6) result += (d*t)/INT_FOR_1over12;
    else if(t < INT_FOR_1over2) result += 2*d;
    else if(t < INT_FOR_2over3) result += d*(INT_FOR_2over3 - t)/INT_FOR_1over12;
    return reduce_to_255(result)
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
    bgra[3] = reduce_to_255(alpha);
    if(s == 0) {
        bgra[0] = bgra[1] = bgra[2] = reduce_to_255(l);
    }
    else {
        // S*(L<0.5? L : 1-L)  =>  s * (l<48960 ? l : 97920 - l) / 97920
        // but the product may exceed the value represented by int32_t i.e. 2^31 - 1
        // e.g. s=97920, l=48960  =>  4_794_163_200 > 2_147_483_648 = 2^31 - 1
        // therefore pre-divide both side by 2 and finally divide by 97920/4 = 24480
        auto delta = (s >> 1) * ((l < INT_FOR_1over2 ? l : INT_FOR_1-l) >> 1) / INT_FOR_1over4;
        bgra[0] = fixed_ldt_to_component(l, delta, h - INT_FOR_1over3);    // blue
        bgra[1] = fixed_ldt_to_component(l, delta, h);                      // green
        bgra[2] = fixed_ldt_to_component(l, delta, h + INT_FOR_1over3);    // red
    }
    uint32_t result = *((uint32_t*)bgra);
    return result;
}

__device__ __host__ void CanvasPixel::set_color(double square_speed, double factor) {
    hue = static_cast<uint16_t>(INT_FOR_2over3 * square_speed / (square_speed + factor));
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
