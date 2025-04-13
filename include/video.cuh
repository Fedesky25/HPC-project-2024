//
// Created by feder on 27/09/2024.
//

#ifndef HPC_PROJECT_2024_VIDEO_CUH
#define HPC_PROJECT_2024_VIDEO_CUH

#include "utils.cuh"
#include "canvas.cuh"
#include "config.cuh"

/**
 * Sequentially computes frames and writes them to the specified file
 * @param config application configuration
 * @param canvas canvas where particles trajectories were written
 */
void write_video_serial(const Configuration & config, Canvas canvas);

/**
 * Computes frames (parallelizing on pixels) and writes them to the output file
 * @param config application configuration
 * @param canvases pointer to list of canvases
 * @param canvas_count number of canvases
 */
void write_video_omp(const Configuration & config, const Canvas * canvases, uint32_t canvas_count);

void write_video_gpu(
        const char * filename,
        const Canvas * canvases, uint32_t canvas_count,
        uint32_t frame_size, int32_t frame_count,
        int32_t lifetime, const RGBA * background);

#endif //HPC_PROJECT_2024_VIDEO_CUH
