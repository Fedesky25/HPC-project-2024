//
// Created by Sofyh02 on 15/08/2024.
//
#include "evolution.cuh"
#include "lower_bound.cuh"
#include <omp.h>

__device__ __host__ void draw(Canvas* canvas, CanvasAdapter * adapter, EvolutionOptions options,
                                       complex_t (*func)(complex_t, FnVariables*), FnVariables* variables,
                                       complex_t z, unsigned int canvas_idx){
    complex_t v, dz;
    double D, elapsed;
    auto dt = options.delta_time;
    auto steps = options.frame_count;
    // Evolving particle looping on lifetime (steps)
    for(uint32_t j=0; j<steps; j++) {
        elapsed = 0.0;
        do {
            v = func(z, variables);
            dz = v * dt;
            D = (adapter->scale * cuda::std::abs(dz));
            if (D > 1) {
                dz /= D;
                elapsed += dt / D;
            } else {
                elapsed += dt;
            }
            auto pixel_idx = adapter->where(z);
            if (pixel_idx != -1) {
                canvas[canvas_idx][pixel_idx].update_age(j);
                canvas[canvas_idx][pixel_idx].set_color(cuda::std::norm(v), options.speed_factor);
            }
            z += dz;
        } while (elapsed < dt);
    }
}

__global__ void evolve_gpu(Canvas* canvas, CanvasAdapter * adapter, EvolutionOptions options, complex_t* particles,
                       uint32_t* belonging_tile, uint32_t* count,
                       complex_t (*func)(complex_t, FnVariables*), FnVariables* variables
                       ){
    auto tile_idx = threadIdx.x + threadIdx.y * blockDim.x;
    auto canvas_idx = blockIdx.x + blockIdx.y * gridDim.x;

    if(canvas_idx >= count[tile_idx]) return;

    auto particle_idx = lower_bound(tile_idx, belonging_tile, blockDim.x * blockDim.y);
    particle_idx += canvas_idx;
    auto z = particles[particle_idx];
    draw(canvas, adapter, options, func, variables, z, canvas_idx);
}

// Divide particle evolution between threads by #pragma omp parallel for.
// Each thread writes particles on its own canvas
Canvas* evolve_omp(Canvas* canvas, CanvasAdapter* adapter, EvolutionOptions options, complex_t* particles, uint64_t N_particles,
                complex_t (*func)(complex_t, FnVariables*), FnVariables* variables){

    #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < N_particles; i++) {
            auto tid = omp_get_thread_num();
            draw(canvas, adapter, options, func, variables, particles[i], tid);
        }

    return canvas;
}
