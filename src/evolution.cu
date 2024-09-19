//
// Created by Sofyh02 on 15/08/2024.
//
#include "evolution.cuh"
#include <omp.h>
#include <random>
#include "curand.h"

__device__ __host__ void draw(Canvas* canvas, CanvasAdapter * adapter, EvolutionOptions * options,
                              ComplexFunction_t func, FnVariables* variables,
                              complex_t z, uint32_t offset, unsigned int canvas_idx){
    complex_t v, dz;
    double D, elapsed;
    auto dt = options->delta_time;
    auto steps = options->frame_count;
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
                if(canvas[canvas_idx][pixel_idx]){
                    return;
                }
                canvas[canvas_idx][pixel_idx].update_age((offset + j) % steps);
                canvas[canvas_idx][pixel_idx].set_color(cuda::std::norm(v), options->speed_factor);
            }
            z += dz;
        } while (elapsed < dt);
    }
}

__global__ void evolve_kernel(Configuration * config, Canvas* canvas, complex_t* particles,
                           const uint32_t * offsets, const uint32_t * rand_offsets, ComplexFunction_t func
                       ){
    auto tile_idx = threadIdx.x;
    auto count = offsets[tile_idx+1] - offsets[tile_idx];
    auto canvas_idx = blockIdx.x;
    if(canvas_idx >= count) return;
    auto z = particles[offsets[tile_idx] + canvas_idx];
    auto offset = rand_offsets[offsets[tile_idx] + canvas_idx];
    draw(canvas, &config->canvas, &config->evolution, func, &config->vars, z, offset, canvas_idx);
}

void evolve_gpu(Configuration * config, Canvas* canvas, complex_t* particles, uint64_t N_particles,
                   const uint32_t* offsets, ComplexFunction_t func, uint32_t tiles_count, uint32_t canvas_count){

    uint32_t *d_rand_offsets;
    float *d_rand_floats;
    cudaMalloc((void **)&d_rand_offsets, N_particles * sizeof (uint32_t));
    cudaMalloc((void **)&d_rand_floats, N_particles * sizeof (float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, d_rand_floats, N_particles);

    for(uint64_t i=0; i<N_particles; i++){
        d_rand_offsets[i] = (uint32_t)(d_rand_floats[i] * (float) config->evolution.frame_count);
    }
    cudaFree(d_rand_floats);

    evolve_kernel<<<canvas_count, tiles_count>>>(config, canvas, particles, offsets, d_rand_offsets, func);

}

// Divide particle evolution between threads by #pragma omp parallel for.
// Each thread writes particles on its own canvas
void evolve_omp(Configuration* config, Canvas* canvas, complex_t* particles, uint64_t N_particles,
                ComplexFunction_t func){

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    #pragma omp parallel
    {
        auto tid = omp_get_thread_num();
        std::default_random_engine generator(seed + omp_get_thread_num());
        std::uniform_int_distribution<int> rand_int(0, (int) config->evolution.frame_count);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < N_particles; i++) {
            uint32_t offset = rand_int(generator);
            draw(canvas, &config->canvas, &config->evolution, func,
                 &config->vars, particles[i], offset, tid);
        }
    }
}
