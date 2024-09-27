//
// Created by Sofyh02 on 15/08/2024.
//
#include "evolution.cuh"
#include <omp.h>
#include <random>
#include "curand.h"

__device__ __host__ void draw(Canvas canvas, CanvasAdapter * adapter, EvolutionOptions * options,
                              ComplexFunction_t func, FnVariables* variables,
                              complex_t z, uint32_t offset){
    complex_t v, dz;
    double D, elapsed;
    auto dt = options->delta_time;
    auto steps = options->frame_count;
    // Evolving particle looping on lifetime (steps)
    for(int32_t j=0; j<steps; j++) {
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
                if(!canvas[pixel_idx].update_age((offset + j) % steps)) return;
                canvas[pixel_idx].set_color(cuda::std::norm(v), options->speed_factor);
            }
            z += dz;
        } while (elapsed < dt);
    }
}

__global__ void evolve_kernel(Configuration * config, Canvas* canvas, complex_t* particles,
                              const uint32_t * tile_offsets, const float * rand_offsets, ComplexFunction_t func
                       ){
    auto tile_idx = threadIdx.x;
    auto count = tile_offsets[tile_idx + 1] - tile_offsets[tile_idx];
    auto canvas_idx = blockIdx.x;
    if(canvas_idx >= count) return;
    auto particle_idx = tile_offsets[tile_idx] + canvas_idx;
    auto z = particles[particle_idx];
    auto offset = static_cast<uint32_t>(rand_offsets[particle_idx] * (float) config->evolution.frame_count);
    draw(canvas[canvas_idx], &config->canvas, &config->evolution, func, &config->vars, z, offset);
}

void evolve_gpu(Configuration * config,
                Canvas* canvas, uint32_t canvas_count,
                complex_t* particles, uint64_t N_particles,
                const uint32_t* tile_offsets, uint32_t tiles_count,
                FunctionChoice fn_choice
){
    timers(1)
    tick(0)
    float *d_rand_floats;
    cudaMalloc((void **)&d_rand_floats, N_particles * sizeof (float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, d_rand_floats, N_particles);
    tock_ms(0)
    std::cout << "Random time offsets generated in " << t_elapsed << "ms" << std::endl;

    tick(0);
    auto func = get_function_global(fn_choice);
    auto d_config = devicify(config);
    evolve_kernel<<<canvas_count, tiles_count>>>(d_config, canvas, particles, tile_offsets, d_rand_floats, func);
    cudaFree(d_config);
    cudaDeviceSynchronize();
    tock_s(0);
    std::cout << "Particle evolution computed in " << t_elapsed << 's' << std::endl;
}

// Divide particle evolution between threads by #pragma omp parallel for.
// Each thread writes particles on its own canvas
void evolve_omp(Configuration* config, Canvas* canvas,
                complex_t* particles, uint64_t N_particles,
                FunctionChoice fn_choice){

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    timers(1) tick(0)
    #pragma omp parallel
    {
        auto tid = omp_get_thread_num();
        std::default_random_engine generator(seed + tid);
        std::uniform_int_distribution<uint32_t> rand_int(0, config->evolution.frame_count - 1);
        auto func = get_function_host(fn_choice);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < N_particles; i++) {
            draw(canvas[tid], &config->canvas, &config->evolution, func,
                 &config->vars, particles[i], rand_int(generator));
        }
    }
    tock_s(0)
    std::cout << "Particle evolution computed in " << t_elapsed << 's' << std::endl;
}

void evolve_serial(Configuration* config, Canvas canvas,
                complex_t* particles, uint64_t N_particles,
                FunctionChoice fn_choice){

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    timers(1) tick(0)
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<uint32_t> rand_int(0, config->evolution.frame_count - 1);
    auto func = get_function_host(fn_choice);
    for (int64_t i = 0; i < N_particles; i++) {
        draw(canvas, &config->canvas, &config->evolution, func,
             &config->vars, particles[i], rand_int(generator));
    }
    tock_s(0)
    std::cout << "Particle evolution computed in " << t_elapsed << 's' << std::endl;
}
