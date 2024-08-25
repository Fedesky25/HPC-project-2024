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

/**
 * @see https://stackoverflow.com/questions/2353211/hsl-to-rgb-color-conversion
 * @param t
 * @return
 */
__device__ __host__ double xToRGB(double t) {
    constexpr double q = LIGHTNESS + SATURATION - LIGHTNESS*SATURATION;
    constexpr double p = 2.0*LIGHTNESS - q;
    constexpr double f = (q-p)*2.0;

    if (t < 0.0) t += 3.0;
    if (t > 3.0) t -= 3.0;
    if (t < 0.5) return p + f * t;
    if (t < 1.5) return q;
    if (t < 2.0) return p + f * (2 - t);
    return p;
}

__device__ __host__ void CanvasPixel::set_color(double square_speed, double factor) {
    // saturation & lightness are fixed to 0.55
    auto x = 2.0 * square_speed / (square_speed + factor); // 1.0 / (1.0 + factor/ square_speed)
    // h = x / 3
    color.r = static_cast<uint8_t>(cuda::std::round(255.0 * xToRGB(x + 1)));
    color.g = static_cast<uint8_t>(cuda::std::round(255.0 * xToRGB(x)));
    color.b = static_cast<uint8_t>(cuda::std::round(255.0 * xToRGB(x - 1)));
}

__device__ __host__ ARGB CanvasPixel::get_color(uint16_t time) const {
    ARGB result;
    if(time < age || time >= age + multiplicity + 200) {
        result.r = BG_R;
        result.g = BG_G;
        result.b = BG_B;
    }
    else if(time <= age + multiplicity) result = color;
    else {
         auto x = (time - age - multiplicity) * 0.005;
         result.r = static_cast<uint8_t>(cuda::std::round(x*BG_R + (1-x)*color.r));
         result.g = static_cast<uint8_t>(cuda::std::round(x*BG_G + (1-x)*color.g));
         result.b = static_cast<uint8_t>(cuda::std::round(x*BG_B + (1-x)*color.b));
    }
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
