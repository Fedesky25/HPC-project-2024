//
// Created by feder on 25/08/2024.
//

#ifndef HPC_PROJECT_2024_FRAMES_CUH
#define HPC_PROJECT_2024_FRAMES_CUH

#include "canvas.cuh"
#include "libavutil/frame.h"


/**
 * Computes the frame at a specific time serially
 * @param time time instant in range [0, frame_count)
 * @param frame_count total number of frames in range [0, UINT16_MAX)
 * @param lifetime lifetime of a particle
 * @param canvas the canvas
 * @param frame output frame (host) buffer
 * @param background background YUVA color
 */
template<bool opaque>
void compute_frame_serial(
        int32_t time, int32_t frame_count, int32_t lifetime,
        const Canvas& canvas, AVFrame * frame, const YUVA * background);


/**
 * Computes the frame at a specific time using OpenMP
 * @param time time instant in range [0, frame_count)
 * @param frame_count total number of frames in range [0, UINT16_MAX)
 * @param lifetime lifetime of a particle
 * @param canvas_array (host) array of canvases
 * @param canvas_count number of canvases
 * @param frame output frame (host) buffer
 * @param background background YUVA color
 */
template<bool opaque>
void compute_frame_omp(
        int32_t time, int32_t frame_count, int32_t lifetime,
        const Canvas * canvas_array, unsigned canvas_count,
        AVFrame * frame, const YUVA * background);


/** Struct containing the constant arguments the gpu kernel needs to compute a frame */
struct FrameKernelArguments {
    int32_t frame_count, lifetime;
    const Canvas * canvas_array;
    unsigned canvas_count;
    YUVA background;

    uint8_t * channels[4];
    int line_size[4];
    int width, height;

    /**
     * Allocates the channels and line_size on the gpu
     * @param frame reference (allocated) frame
     * @return a copy of this instance on the gpu
     */
    const FrameKernelArguments * init(AVFrame * frame, bool opaque);

    /**
     * Copies the contents into the given frame
     * @param frame
     */
    void copy_into(AVFrame * frame) const;

    /** Frees up the allocated channels */
    void free();
};

/**
 * Computes the frame at a specific time using the GPU
 * @tparam opaque whether the video has opaque background or not
 * @param time current simulation time index
 * @param args pointer to arguments allocated on the gpu
 */
template<bool opaque>
__global__ void compute_frame_kernel(int32_t time, const FrameKernelArguments * args);

/**
 * Computes the frame at a specific time using the GPU
 * @param time time instant in range [0, frame_count)
 * @param frame_count total number of frames in range [0, UINT16_MAX)
 * @param canvas_array (device) array of canvases
 * @param canvas_count number of canvases
 * @param frame output frame (device) buffer
 * @param size total number of pixels
 */
template<bool opaque>
void compute_frame_gpu(int32_t time, int32_t frame_count,
                       const Canvas * canvas_array, unsigned canvas_count,
                       unsigned char * frame, uint32_t size,
                       int32_t lifetime, const RGBA * background);

#endif //HPC_PROJECT_2024_FRAMES_CUH
