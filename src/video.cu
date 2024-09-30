//
// Created by feder on 27/09/2024.
//

#include "video.cuh"
#include "frames.cuh"
#include <fstream>
#include <iomanip>

#define HEADER std::cout << "Frame computation: iter. | c (us) | w (ms)" << std::endl;

template<bool opaque>
void write_video_serial_internal(
        const char * filename, Canvas canvas,
        uint32_t frame_size, int32_t frame_count,
        const RGBA & background
) {
    constexpr auto bytes = opaque ? 3 : 4;
    auto mem = bytes * frame_size;
    auto frame = new unsigned char [mem];
    std::ofstream out(filename);
    RGBA color;
    unsigned char bg_bytes[bytes];
    background.write<opaque>(bg_bytes);
    auto inv_lifetime = 1.0f / (float) frame_count;
    float tc, tw;
    HEADER
    timers(2)
    tick(0)
    for(int32_t t=0; t<frame_count; t++) {
        tick(1)
        for(uint32_t i=0; i<frame_size; i++) {
            auto delta = canvas[i].time_distance(t, frame_count);
            if(delta >= frame_count + canvas[i].multiplicity) {
                for(int b=0; b<bytes; b++) frame[bytes*i + b] = bg_bytes[b];
            }
            else {
                color.from_hue(canvas[i].hue);
                if(delta >= canvas[i].multiplicity) color.A = 1.0f;
                else {
                    color.A = (float) (delta-canvas[i].multiplicity) * inv_lifetime;
                    color.over<opaque>(&background);
                }
                color.write<opaque>(frame + bytes*i);
            }
            std::cout << i << ' ';
        }
        tock_ms(1)
        tc = t_elapsed;
        tick(1)
        out.write(reinterpret_cast<const char *>(frame), mem);
        tock_ms(1)
        tw = t_elapsed;
        std::cout << "                   " << std::setw(5) << (t+1)
                  << " | " << std::setw(6) << tc
                  << " | " << std::setw(6) << tw << std::endl;
    }
    tock_s(0)
    std::cout << "  :: total " << t_elapsed << 's' << std::endl;
    delete [] frame;
}

void write_video_serial(
        const char * filename, Canvas canvas,
        uint32_t frame_size, int32_t frame_count,
        const RGBA & background
) {
    if(background.A == 1.0f) write_video_serial_internal<true>(filename, canvas, frame_size, frame_count, background);
    else write_video_serial_internal<false>(filename, canvas, frame_size, frame_count, background);
}



void write_video_omp(
        const char * filename,
        const Canvas * canvases, uint32_t canvas_count,
        uint32_t frame_size, int32_t frame_count,
        const FixedHSLA * background
) {

}



void write_video_gpu(
        const char * filename,
        const Canvas * canvases, uint32_t canvas_count,
        uint32_t frame_size, int32_t frame_count,
        const FixedHSLA * background
) {
    std::ofstream raw_output(filename);
    auto frame_mem = frame_size * sizeof(uint32_t);
    uint32_t *h_frame, *d_frame[2];
    h_frame = (uint32_t*) malloc(frame_mem);
    cudaMalloc(d_frame, frame_mem);
    cudaMalloc(d_frame+1, frame_mem);
    std::cout << "Frame buffers: CPU=" << (((frame_mem-1)>>20)+1) << "MB, GPU="
              << (((frame_mem*2-1)>>20)+1) << "MB" << std::endl << std::fixed;
    std::cout << "Frame computation: iter. | c (us) | w (ms)" << std::endl;
    std::cout.width(6);

    float time_write, time_compute;
    auto begin = std::chrono::steady_clock::now();
    compute_frame_gpu(
            0, frame_count,
            canvases, canvas_count,
            d_frame[0], frame_size,
            background);
    cudaDeviceSynchronize();
    auto _end = std::chrono::steady_clock::now();
    time_compute = (std::chrono::duration<float,std::micro>(_end-begin)).count();
    std::cout << "                   " << std::setw(5) << 0
              << " | " << std::setw(6) << time_compute
              << " | " << std::endl;


    for(int32_t i=1; i<frame_count; i++) {
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                auto start = std::chrono::steady_clock::now();
                cudaMemcpy(h_frame, d_frame[(i&1)^1], frame_mem, cudaMemcpyDeviceToHost);
                raw_output.write(reinterpret_cast<const char *>(h_frame), frame_mem);
                auto end = std::chrono::steady_clock::now();
                time_write = (std::chrono::duration<float, std::milli>(end - start)).count();
            }
            #pragma omp section
            {
                auto start = std::chrono::steady_clock::now();
                compute_frame_gpu(i, frame_count, canvases, canvas_count, d_frame[i&1], frame_size, background);
                cudaDeviceSynchronize();
                auto end = std::chrono::steady_clock::now();
                time_compute = (std::chrono::duration<float,std::micro>(end-start)).count();
            }
        }
        std::cout << "                   " << std::setw(5) << i
                  << " | " << std::setw(6) << time_compute
                  << " | " << std::setw(6) << time_write << std::endl;
    }

    begin = std::chrono::steady_clock::now();
    cudaMemcpy(h_frame, d_frame[(frame_count-1)&1], frame_mem, cudaMemcpyDeviceToHost);
    raw_output.write(reinterpret_cast<const char *>(h_frame), frame_mem);
    _end = std::chrono::steady_clock::now();
    time_write = (std::chrono::duration<float,std::milli>(_end-begin)).count();
    std::cout << "                   " << std::setw(5) << frame_count
              << " |        | " << std::setw(6) << time_write << std::endl;

    cudaFree(d_frame[0]);
    cudaFree(d_frame[1]);
    free(h_frame);
}