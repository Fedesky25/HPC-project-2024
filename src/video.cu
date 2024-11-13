//
// Created by feder on 27/09/2024.
//

#include "video.cuh"
#include "frames.cuh"
#include <fstream>
#include <iomanip>
#include <omp.h>


#define PRINT_TIMES(TIME) {\
    std::cout << "   " << std::setw(5) << (TIME)                                  \
              << " | " << std::setw(5) << tc[0] << " | " << std::setw(5) << tw[0] \
              << " | " << std::setw(5) << tc[1] << " | " << std::setw(5) << tw[1] \
              << " | " << std::setw(5) << tc[2] << " | " << std::setw(5) << tw[2] \
              << " | " << std::setw(5) << tc[3] << " | " << std::setw(5) << tw[3] \
              << " | " << std::setw(5) << tc[4] << " | " << std::setw(5) << tw[4] \
              << " | " << std::setw(5) << tc[5] << " | " << std::setw(5) << tw[5] \
              << " | " << std::setw(5) << tc[6] << " | " << std::setw(5) << tw[6] \
              << " | " << std::setw(5) << tc[7] << " | " << std::setw(5) << tw[7] \
              << std::endl;                                                       \
    tc[0] = tc[1] = tc[2] = tc[3] = tc[4] = tc[5] = tc[6] = tc[7] = 0;            \
    tw[0] = tw[1] = tw[2] = tw[3] = tw[4] = tw[5] = tw[6] = tw[7] = 0;            \
    }


#define PRINT_SUMMARY(FACTOR) { \
    auto m = 0.1f / total;                    \
    auto writing = tw[0] + tw[1] + tw[2] + tw[3] + tw[4] + tw[5] + tw[6] + tw[7];     \
    auto computation = tc[0] + tc[1] + tc[2] + tc[3] + tc[4] + tc[5] + tc[6] + tc[7]; \
    std::cout << frame_count << " frames written in " << total << "s (computation: "  \
              << computation*m*(FACTOR) << "%, file write: " << writing*m << "%)" << std::endl; \
}




template<bool opaque>
void write_video_serial_internal(
        const char * filename, Canvas canvas,
        uint32_t frame_size, int32_t frame_count,
        int32_t lifetime, const RGBA & background
) {
    constexpr auto bytes = opaque ? 3 : 4;
    auto mem = bytes * frame_size;
    auto frame = new unsigned char [mem];
    std::ofstream out(filename);
    RGBA color;
    unsigned char bg_bytes[bytes];
    background.write<opaque>(bg_bytes);
    auto inv_lifetime = 1.0f / (float) lifetime;
    float tc[8] = {0}, tw[8] = {0};
    if(verbose) std::cout << "Frame computation (iteration, (computation [ms], writing [ms]) * 8):" << std::endl << std::setprecision(1);
    timers(2)
    tick(0)
    for(int32_t t=0; t<frame_count; t++) {
        tick(1)
        for(uint32_t i=0; i<frame_size; i++) {
            auto delta = canvas[i].time_distance(t, frame_count);
            if(delta >= lifetime + canvas[i].multiplicity) {
                for(int b=0; b<bytes; b++) frame[bytes*i + b] = bg_bytes[b];
            }
            else {
                color.from_hue(canvas[i].hue);
                if(delta < canvas[i].multiplicity) color.A = 1.0f;
                else {
                    color.A = (float) (lifetime+canvas[i].multiplicity-delta) * inv_lifetime;
                    color.over<opaque>(&background);
                }
                color.write<opaque>(frame + bytes*i);
            }
        }
        tock_ms(1)
        tc[t&7] += t_elapsed;
        tick(1)
        out.write(reinterpret_cast<const char *>(frame), mem);
        tock_ms(1)
        tw[t&7] += t_elapsed;
        if((t&7) == 7 && verbose) PRINT_TIMES(t+1)
    }
    tock_s(0)
    auto total = t_elapsed;
    auto remaining = (frame_count-1)&7;
    if(verbose) {
        if(remaining) {
            std::cout << "   " << std::setw(5) << frame_count
                      << " | " << std::setw(5) << tc[0] << " | " << std::setw(5) << tw[0];
            for(int32_t j=1; j<remaining; j++)
                std::cout << " | " << std::setw(5) << tc[j] << " | " << std::setw(5) << tw[j];
            std::cout << std::endl;
        }
        std::cout << "  :: total " << total << 's' << std::endl;
    }
    else PRINT_SUMMARY(1)
    delete [] frame;
}

void write_video_serial(
        const char * filename, Canvas canvas,
        uint32_t frame_size, int32_t frame_count,
        int32_t lifetime, const RGBA & background
) {
    if(background.A == 1.0f) write_video_serial_internal<true>(filename, canvas, frame_size, frame_count, lifetime, background);
    else write_video_serial_internal<false>(filename, canvas, frame_size, frame_count, lifetime, background);
}


template<bool opaque>
void write_video_omp_internal(
        const char * filename,
        const Canvas * canvases, uint32_t canvas_count,
        uint32_t frame_size, int32_t frame_count,
        int32_t lifetime, const RGBA & background
) {
    std::ofstream out(filename);
    constexpr auto bytes = opaque ? 3 : 4;
    auto mem = bytes * frame_size;
    unsigned char * frame_buffers[2];
    frame_buffers[0] = new unsigned char [mem];
    frame_buffers[1] = new unsigned char [mem];
    unsigned char bg_bytes[bytes];
    background.write<opaque>(bg_bytes);
    auto inv_lifetime = 1.0f / (float) lifetime;
    float tc[8] = {0}, tw[8] = {0};
    tw[0] = -1.0;

    auto frame_size_signed = (int32_t) frame_size;

    omp_set_nested(1);

    if(verbose) std::cout << "Frame computation (iteration, (computation [ms], writing [ms]) * 8):" << std::endl << std::setprecision(1);
    auto start_all = std::chrono::steady_clock::now();
    for(int32_t t=0; t<frame_count; t++) {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                if(t > 0) {
                    auto start = std::chrono::steady_clock::now();
                    out.write(reinterpret_cast<const char *>(frame_buffers[(t-1)&1]), mem);
                    auto end = std::chrono::steady_clock::now();
                    tw[t&7] += (std::chrono::duration<float, std::milli>(end-start)).count();
                }
            }
            #pragma omp section
            {
                auto start = std::chrono::steady_clock::now();
                auto frame = frame_buffers[t&1];
                #pragma omp parallel
                {
                    RGBA color;
                    CanvasPixel * pixel;
                    int32_t delta_min, delta;
                    #pragma omp for schedule(static)
                    for(int32_t i=0; i<frame_size_signed; i++) {
                        pixel = &canvases[0][i];
                        delta_min = canvases[0][i].time_distance(t, frame_count);
                        for(uint32_t c=1; c<canvas_count; c++) {
                            delta = canvases[c][i].time_distance(t, frame_count);
                            if(delta < delta_min) {
                                delta_min = delta;
                                pixel = &canvases[c][i];
                            }
                        }
                        if(delta_min >= lifetime + pixel->multiplicity) {
                            for(int b=0; b<bytes; b++) frame[bytes*i + b] = bg_bytes[b];
                        }
                        else {
                            color.from_hue(pixel->hue);
                            if(delta_min < pixel->multiplicity) color.A = 1.0f;
                            else {
                                color.A = (float) (lifetime+pixel->multiplicity-delta_min) * inv_lifetime;
                                color.over<opaque>(&background);
                            }
                            color.write<opaque>(frame + bytes*i);
                        }
                    }
                }
                auto end = std::chrono::steady_clock::now();
                tc[t&7] += (std::chrono::duration<float, std::milli>(end-start)).count();
            }
        }
        if((t&7) == 7 && verbose) PRINT_TIMES(t+1)
    }
    out.write(reinterpret_cast<const char *>(frame_buffers[(frame_count-1)&1]), mem);
    auto end_all = std::chrono::steady_clock::now();
    float total = (std::chrono::duration<float, std::ratio<1>>(end_all-start_all)).count();

    if(verbose) std::cout << "  :: total " << total << 's' << std::endl;
    else PRINT_SUMMARY(1)

    delete [] frame_buffers[0];
    delete [] frame_buffers[1];
}


void write_video_omp(
        const char * filename,
        const Canvas * canvases, uint32_t canvas_count,
        uint32_t frame_size, int32_t frame_count,
        int32_t lifetime, const RGBA & background
) {
    if(background.A == 1.0f) write_video_omp_internal<true>(filename, canvases, canvas_count, frame_size, frame_count, lifetime, background);
    else write_video_omp_internal<false>(filename, canvases, canvas_count, frame_size, frame_count, lifetime, background);
}


template<bool opaque>
void write_video_gpu_internal(
        const char * filename,
        const Canvas * canvases, uint32_t canvas_count,
        uint32_t frame_size, int32_t frame_count, int32_t lifetime,
        const RGBA * background
) {
    std::ofstream raw_output(filename);
    auto frame_mem = frame_size * sizeof(uint32_t);
    unsigned char *h_frame, *d_frame[2];
    h_frame = (unsigned char*) malloc(frame_mem);
    cudaMalloc(d_frame, frame_mem);
    cudaMalloc(d_frame+1, frame_mem);
    if(verbose) {
        std::cout << "Frame buffers: CPU=" << (((frame_mem - 1) >> 20) + 1) << "MB, GPU="
                  << (((frame_mem * 2 - 1) >> 20) + 1) << "MB" << std::endl << std::fixed;
        std::cout << "Frame computation (iteration, (computation [us], writing [ms]) * 8):" << std::endl;
    }

    float tw[8] = {0}, tc[8] = {0};
    auto begin = std::chrono::steady_clock::now();
    compute_frame_gpu<opaque>(
            0, frame_count,
            canvases, canvas_count,
            d_frame[0], frame_size, lifetime,
            background);
    cudaDeviceSynchronize();
    auto _end = std::chrono::steady_clock::now();
    tc[0] += (std::chrono::duration<float,std::micro>(_end-begin)).count();
    auto start_all = begin;

    for(int32_t t=1; t < frame_count; t++) {
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                auto start = std::chrono::steady_clock::now();
                cudaMemcpy(h_frame, d_frame[(t & 1) ^ 1], frame_mem, cudaMemcpyDeviceToHost);
                raw_output.write(reinterpret_cast<const char *>(h_frame), frame_mem);
                auto end = std::chrono::steady_clock::now();
                tw[(t - 1) & 7] += (std::chrono::duration<float, std::milli>(end - start)).count();
            }
            #pragma omp section
            {
                auto start = std::chrono::steady_clock::now();
                compute_frame_gpu<opaque>(t, frame_count, canvases, canvas_count, d_frame[t & 1], frame_size, lifetime, background);
                cudaDeviceSynchronize();
                auto end = std::chrono::steady_clock::now();
                tc[t & 7] += (std::chrono::duration<float,std::micro>(end - start)).count();
            }
        }
        if((t & 7) == 0 && verbose) PRINT_TIMES(t)
    }
    begin = std::chrono::steady_clock::now();
    cudaMemcpy(h_frame, d_frame[(frame_count-1)&1], frame_mem, cudaMemcpyDeviceToHost);
    raw_output.write(reinterpret_cast<const char *>(h_frame), frame_mem);
    _end = std::chrono::steady_clock::now();
    tw[(frame_count-1)&7] += (std::chrono::duration<float,std::milli>(_end-begin)).count();
    if((frame_count & 7) == 0 && verbose) PRINT_TIMES(frame_count)
    _end = std::chrono::steady_clock::now();
    float total = (std::chrono::duration<float, std::ratio<1>>(_end-start_all)).count();
    if(verbose) std::cout << "  :: total " << total << 's' << std::endl;
    else PRINT_SUMMARY(1e-3f)

    cudaFree(d_frame[0]);
    cudaFree(d_frame[1]);
    free(h_frame);
}

void write_video_gpu(
        const char * filename,
        const Canvas * canvases, uint32_t canvas_count,
        uint32_t frame_size, int32_t frame_count,
        int32_t lifetime, const RGBA * background
) {
    if(background->A == 1.0f) write_video_gpu_internal<true>(filename, canvases, canvas_count, frame_size, frame_count, lifetime, background);
    else write_video_gpu_internal<false>(filename, canvases, canvas_count, frame_size, frame_count, lifetime, background);
}