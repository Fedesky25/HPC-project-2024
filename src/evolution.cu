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

__global__ void evolve_gpu(Configuration * config, Canvas* canvas, complex_t* particles,
                           const uint32_t * offsets, ComplexFunction_t func
                       ){
    auto tile_idx = threadIdx.x;
    auto count = offsets[tile_idx+1] - offsets[tile_idx];
    auto canvas_idx = blockIdx.x;
    if(canvas_idx >= count) return;
    auto z = particles[offsets[tile_idx] + canvas_idx];
    draw(canvas, &config->canvas, config->evolution, func, &config->vars, z, canvas_idx);
}

// Divide particle evolution between threads by #pragma omp parallel for.
// Each thread writes particles on its own canvas
void evolve_omp(Canvas* canvas, CanvasAdapter* adapter, EvolutionOptions options, complex_t* particles, uint64_t N_particles,
                complex_t (*func)(complex_t, FnVariables*), FnVariables* variables){

    #pragma omp parallel
    {
        auto tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < N_particles; i++) {
            draw(canvas, adapter, options, func, variables, particles[i], tid);
        }
    }
}
