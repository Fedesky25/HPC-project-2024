//
// Created by feder on 25/08/2024.
//

#include "frames.cuh"

void compute_frame_serial(
        int32_t time, int32_t frame_count,
        Canvas * canvas_array, unsigned canvas_count,
        uint32_t * frame, uint32_t size,
        const FixedHSLA * background
) {
    for(uint32_t i=0; i<size; i++) {
        unsigned c = 0;
        while(c < canvas_count && !canvas_array[c][i].alive(time)) c++;
        if(c == canvas_count) continue;
        auto px = &(canvas_array[c][i]);
        for(; c < canvas_count; c++) {
            if(canvas_array[c][i] < *px)
                px = &(canvas_array[c][i]);
        }
        frame[i] = px->get_color(time, frame_count, background);
    }
}


void compute_frame_omp(
        int32_t time, int32_t frame_count,
        Canvas * canvas_array, unsigned canvas_count,
        uint32_t * frame, uint32_t size,
        const FixedHSLA * background
) {
    #pragma omp parallel for schedule(static)
    for(int32_t i=0; i<size; i++) {
        unsigned c = 0;
        while(c < canvas_count && !canvas_array[c][i].alive(time)) c++;
        if(c == canvas_count) continue;
        auto px = &(canvas_array[c][i]);
        for(; c < canvas_count; c++) {
            if(canvas_array[c][i] < *px)
                px = &(canvas_array[c][i]);
        }
        frame[i] = px->get_color(time, frame_count, background);
    }
}


__global__ void compute_frame_no_divergence(
        int32_t time, int32_t frame_count,
        Canvas * canvas_array, unsigned canvas_count,
        uint32_t * frame, unsigned offset,
        const FixedHSLA * background
) {
    unsigned pixel_index = offset + threadIdx.x + blockIdx.x * blockDim.x;
    unsigned selected_canvas = 0;
    int32_t time_delta, selected_time_delta = canvas_array[0][pixel_index].time_distance(time, frame_count);
    for(unsigned i=1; i<canvas_count; i++) {
        time_delta = canvas_array[i][pixel_index].time_distance(time, frame_count);
        if(time_delta < selected_time_delta) {
            selected_time_delta = time_delta;
            selected_canvas = i;
        }
        __syncthreads();
    }
    auto px = &canvas_array[selected_canvas][pixel_index];
    frame[pixel_index] = px->get_color(time, frame_count, background);
}

void compute_frame_gpu(
        int32_t time, int32_t frame_count,
        Canvas * canvas_array, unsigned canvas_count,
        uint32_t * frame, uint32_t size,
        const FixedHSLA * background
) {
    uint32_t block_count = size >> 10;
    compute_frame_no_divergence<<<block_count, 1024>>>(
            time, frame_count, canvas_array, canvas_count, frame, 0, background);
    block_count = (size & 1023) >> 5;
    if(block_count) compute_frame_no_divergence<<<block_count, 32>>>(
            time, frame_count, canvas_array, canvas_count, frame, size & (~1023), background);
    if(size & 31) compute_frame_no_divergence<<<1, size & 31>>>(
            time, frame_count, canvas_array, canvas_count, frame, size & (~31), background);
}
