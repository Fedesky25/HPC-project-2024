//
// Created by feder on 25/08/2024.
//

#include "frames.cuh"


template<bool opaque>
BOTH void write_color(const YUVA & clr, AVFrame * frame, int x, int y) {
    frame->data[0][x + y*frame->linesize[0]] = byte_clr_chl1(clr.Y);
    frame->data[1][x + y*frame->linesize[1]] = byte_clr_chl2(clr.U);
    frame->data[2][x + y*frame->linesize[2]] = byte_clr_chl2(clr.V);
    CONSTEXPR_IF(opaque) frame->data[3][x + y*frame->linesize[3]] = byte_clr_chl1(clr.A);
}


DEF_OPAQUE_FN(compute_frame_serial, (
        int32_t time, int32_t frame_count, int32_t lifetime,
        const Canvas& canvas, AVFrame * frame, const YUVA * background
)) {
    YUVA brush;
    auto inv_lifetime = 1.0f / (float) lifetime;
    uint8_t bytes_bg[] = {
            byte_clr_chl1(background->Y),
            byte_clr_chl2(background->U),
            byte_clr_chl2(background->V),
            byte_clr_chl1(background->A),
    };
    for (int y = 0; y < frame->height; y++) {
        for (int x = 0; x < frame->width; x++) {
            int i = x + y*frame->width;
            auto delta = canvas[i].time_distance(time, frame_count) - canvas[i].multiplicity;
            if(delta >= lifetime) {
                frame->data[0][x + y*frame->linesize[0]] = bytes_bg[0];
                frame->data[1][x + y*frame->linesize[1]] = bytes_bg[1];
                frame->data[2][x + y*frame->linesize[2]] = bytes_bg[2];
                CONSTEXPR_IF(opaque) frame->data[3][x + y*frame->linesize[3]] =  bytes_bg[3];
            }
            else {
                brush.from_hue(canvas[i].hue);
                brush.A = 1.0f;
                if (delta >= 0) {
                    brush.A -= (float) delta * inv_lifetime;
                    brush.over<opaque>(background);
                }
                write_color<opaque>(brush, frame, x, y);
            }
        }
    }
}

DEF_OPAQUE_FN(compute_frame_omp, (
        int32_t time, int32_t frame_count, int32_t lifetime,
        const Canvas * canvas_array, unsigned canvas_count,
        AVFrame * frame, const YUVA * background
)) {
    YUVA brush;
    auto inv_lifetime = 1.0f / (float) lifetime;
    uint8_t bytes_bg[] = {
            byte_clr_chl1(background->Y),
            byte_clr_chl2(background->U),
            byte_clr_chl2(background->V),
            byte_clr_chl1(background->A),
    };
    #pragma omp parallel for schedule(static) private(brush)
    for (int y = 0; y < frame->height; y++) {
        for (int x = 0; x < frame->width; x++) {
            int i = x + y*frame->width;
            unsigned selected_canvas = 0;
            int32_t dt, time_delta = canvas_array[0][i].time_distance(time, frame_count);
            for(unsigned c=1; c<canvas_count; c++) {
                dt = canvas_array[c][i].time_distance(time, frame_count);
                if(dt < time_delta) {
                    time_delta = dt;
                    selected_canvas = c;
                }
            }
            auto pixel = canvas_array[selected_canvas][i];
            time_delta -= pixel.multiplicity;
            if(time_delta >= lifetime) {
                frame->data[0][x + y*frame->linesize[0]] = bytes_bg[0];
                frame->data[1][x + y*frame->linesize[1]] = bytes_bg[1];
                frame->data[2][x + y*frame->linesize[2]] = bytes_bg[2];
                CONSTEXPR_IF(opaque) frame->data[3][x + y*frame->linesize[3]] =  bytes_bg[3];
            }
            else {
                brush.from_hue(pixel.hue);
                brush.A = 1.0f;
                if (time_delta >= 0)
                    brush.A -= (float) time_delta * inv_lifetime;
                brush.over<opaque>(background);
            }
            write_color<opaque>(brush, frame, x, y);
        }
    }
}

const FrameKernelArguments * FrameKernelArguments::init(AVFrame *frame, bool opaque) {
    width = frame->width;
    height = frame->height;
    for(int i=0; i < 3+!opaque; i++) {
        line_size[i] = frame->linesize[i];
        auto size = frame->linesize[i] * height;
        CATCH_CUDA_ERROR(cudaMalloc(channels+i,size))
    }
    if(opaque) channels[3] = nullptr;
    FrameKernelArguments * this_d;
    CATCH_CUDA_ERROR(cudaMalloc(&this_d, sizeof(FrameKernelArguments)))
    CATCH_CUDA_ERROR(cudaMemcpy(this_d, this, sizeof(FrameKernelArguments), cudaMemcpyHostToDevice));
    return this_d;
}

void FrameKernelArguments::copy_into(AVFrame *frame) const {
    auto N = channels[3] == nullptr ? 3 : 4;
    for(int i=0; i < N; i++) {
        auto size = frame->linesize[i] * height;
        CATCH_CUDA_ERROR(cudaMemcpy(frame->data[i], channels[i], size, cudaMemcpyDeviceToHost))
    }
}

void FrameKernelArguments::free() {
    auto N = channels[3] == nullptr ? 3 : 4;
    for(int i=0; i < N; i++) CATCH_CUDA_ERROR(cudaFree(channels[i]));
}


template __global__ void compute_frame_kernel<true>(int32_t time, const FrameKernelArguments * args);
template __global__ void compute_frame_kernel<false>(int32_t time, const FrameKernelArguments * args);

template<bool opaque>
__global__ void compute_frame_kernel(int32_t time, const FrameKernelArguments * args) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x >= args->width || y >= args->height) return;
    auto pixel_index = x + y * args->width;
    unsigned selected_canvas = 0;
    int32_t dt, time_delta = args->canvas_array[0][pixel_index].time_distance(time, args->frame_count);
    for(unsigned i=1; i<args->canvas_count; i++) {
        dt = args->canvas_array[i][pixel_index].time_distance(time, args->frame_count);
        if(dt < time_delta) {
            time_delta = dt;
            selected_canvas = i;
        }
        // cannot sync because of forking
        // __syncthreads();
    }
    auto px = args->canvas_array[selected_canvas][pixel_index];
    time_delta -= px.multiplicity;
    if(time_delta >= args->lifetime) {
        args->channels[0][x + y*args->line_size[0]] = byte_clr_chl1(args->background.Y);
        args->channels[1][x + y*args->line_size[1]] = byte_clr_chl2(args->background.U);
        args->channels[2][x + y*args->line_size[2]] = byte_clr_chl2(args->background.V);
        CONSTEXPR_IF(opaque) args->channels[3][x + y*args->line_size[3]] = byte_clr_chl1(args->background.A);
    }
    else {
        YUVA color;
        color.from_hue(px.hue);
        color.A = 1.0f;
        if(time_delta >= 0) {
            color.A -= (float) time_delta / (float) args->lifetime;
            color.over<opaque>(&args->background);
        }
        args->channels[0][x + y*args->line_size[0]] = byte_clr_chl1(color.Y);
        args->channels[1][x + y*args->line_size[1]] = byte_clr_chl2(color.U);
        args->channels[2][x + y*args->line_size[2]] = byte_clr_chl2(color.V);
        CONSTEXPR_IF(opaque) args->channels[3][x + y*args->line_size[3]] = byte_clr_chl1(color.A);
    }
}

template<bool opaque>
__global__ void compute_frame_no_divergence(
        int32_t time, int32_t frame_count,
        const Canvas * canvas_array, unsigned canvas_count,
        unsigned char * frame, int32_t lifetime, unsigned offset,
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
            color.over<opaque>(background);
        }
        color.write<opaque>(frame + bytes * pixel_index);
    }
}

template<bool opaque>
void compute_frame_gpu(
        int32_t time, int32_t frame_count,
        const Canvas * canvas_array, unsigned canvas_count,
        unsigned char * frame, uint32_t size,
        int32_t lifetime, const RGBA * background
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

template void compute_frame_gpu<false>(int32_t time, int32_t frame_count,
                               const Canvas * canvas_array, unsigned canvas_count,
                               unsigned char * frame, uint32_t size,
                               int32_t lifetime, const RGBA * background);

template void compute_frame_gpu<true>(int32_t time, int32_t frame_count,
        const Canvas * canvas_array, unsigned canvas_count,
        unsigned char * frame, uint32_t size,
        int32_t lifetime, const RGBA * background);
