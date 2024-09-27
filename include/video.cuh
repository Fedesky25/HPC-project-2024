//
// Created by feder on 27/09/2024.
//

#ifndef HPC_PROJECT_2024_VIDEO_CUH
#define HPC_PROJECT_2024_VIDEO_CUH

#include "utils.cuh"
#include "canvas.cuh"


void write_video_serial(
        const char * filename, Canvas canvas,
        uint32_t frame_size, int32_t frame_count,
        const FixedHSLA * background);

void write_video_omp(
        const char * filename,
        const Canvas * canvases, uint32_t canvas_count,
        uint32_t frame_size, int32_t frame_count,
        const FixedHSLA * background);

void write_video_gpu(
        const char * filename,
        const Canvas * canvases, uint32_t canvas_count,
        uint32_t frame_size, int32_t frame_count,
        const FixedHSLA * background);

#endif //HPC_PROJECT_2024_VIDEO_CUH
