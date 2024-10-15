//
// Created by feder on 25/08/2024.
//

#ifndef HPC_PROJECT_2024_FRAMES_CUH
#define HPC_PROJECT_2024_FRAMES_CUH

#include "canvas.cuh"
#include <cstdint>


/**
 * Computes the frame at a specific time using OpenMP
 * @param time time instant in range [0, frame_count)
 * @param frame_count total number of frames in range [0, UINT16_MAX)
 * @param canvas_array (host) array of canvases
 * @param canvas_count number of canvases
 * @param frame output frame (host) buffer
 * @param size total number of pixels
 */
void compute_frame_omp(int32_t time, int32_t frame_count,
                       const Canvas * canvas_array, unsigned canvas_count,
                       uint32_t * frame, uint32_t size,
                       const RGBA * background);

/**
 * Computes the frame at a specific time using the GPU
 * @param time time instant in range [0, frame_count)
 * @param frame_count total number of frames in range [0, UINT16_MAX)
 * @param canvas_array (device) array of canvases
 * @param canvas_count number of canvases
 * @param frame output frame (device) buffer
 * @param size total number of pixels
 */
void compute_frame_gpu(int32_t time, int32_t frame_count,
                       const Canvas * canvas_array, unsigned canvas_count,
                       uint32_t * frame, uint32_t size, int32_t lifetime,
                       const RGBA * background);

#endif //HPC_PROJECT_2024_FRAMES_CUH
