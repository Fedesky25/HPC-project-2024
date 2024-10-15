//
// Created by feder on 25/08/2024.
//

#include "frames.cuh"


void compute_frame_omp(
        int32_t time, int32_t frame_count,
        const Canvas * canvas_array, unsigned canvas_count,
        uint32_t * frame, uint32_t size,
        const RGBA * background
) {
    #pragma omp parallel for schedule(static)
    for(int32_t i=0; i<size; i++) {
        unsigned selected_canvas = 0;
        int32_t time_delta, selected_time_delta = canvas_array[0][i].time_distance(time, frame_count);
        for(unsigned c=1; c<canvas_count; c++) {
            time_delta = canvas_array[c][i].time_distance(time, frame_count);
            if(time_delta < selected_time_delta) {
                selected_time_delta = time_delta;
                selected_canvas = c;
            }
        }
        auto px = &canvas_array[selected_canvas][i];
        //frame[i] = px->get_color(selected_time_delta, frame_count, background);
    }
}

template<bool opaque>
__global__ void compute_frame_no_divergence(
        int32_t time, int32_t frame_count,
        const Canvas * canvas_array, unsigned canvas_count,
        uint32_t * frame, int32_t lifetime, unsigned offset,
        const RGBA * background
) {
    constexpr auto bytes = opaque ? 3 : 4;
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
    RGBA color;
    auto inv_lifetime = 1.0f / (float) lifetime;
    auto px = &canvas_array[selected_canvas][pixel_index];
    if(selected_time_delta >= lifetime + px->multiplicity){
        background->write<opaque>(frame + bytes * pixel_index);
    }
    else{
        color.from_hue(px->hue);
        if(selected_time_delta < px->multiplicity) color.A = 1.0f;
        else {
            color.A = (float) (lifetime+px->multiplicity-selected_time_delta) * inv_lifetime;
            color.over<opaque>(&background);
        }
        color.write<opaque>(frame + bytes * pixel_index);
    }
}

template<bool opaque>
void compute_frame_gpu(
        int32_t time, int32_t frame_count,
        const Canvas * canvas_array, unsigned canvas_count,
        uint32_t * frame, uint32_t size, int32_t lifetime,
        const RGBA * background
) {
    uint32_t block_count = size >> 10;
    compute_frame_no_divergence<opaque><<<block_count, 1024>>>(
            time, frame_count, canvas_array, canvas_count, frame, lifetime, 0, background);
    block_count = (size & 1023) >> 5;
    if(block_count) compute_frame_no_divergence<opaque><<<block_count, 32>>>(
            time, frame_count, canvas_array, canvas_count, frame, lifetime, size & (~1023), background);
    if(size & 31) compute_frame_no_divergence<opaque><<<1, size & 31>>>(
            time, frame_count, canvas_array, canvas_count, frame, lifetime, size & (~31), background);
}
