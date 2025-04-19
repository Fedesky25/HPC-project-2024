//
// Created by Sofyh02 on 15/08/2024.
//
#include "evolution.cuh"
#include <omp.h>
#include <random>
#include <curand.h>
#include <iomanip>

#define PRINT_TIME { \
    std::cout << "Particle evolution computed in " << std::setprecision(3); \
    if(t_elapsed >= 1e3) std::cout << t_elapsed*1e-3 << 's' << std::endl;   \
    else std::cout << t_elapsed << "ms" << std::endl;                       \
}

__host__ void draw(
        Canvas canvas, const CanvasAdapter * adapter, const EvolutionOptions * options,
        ComplexFunction_t func, const FnVariables* variables,
        complex_t z, uint32_t offset
){
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
            D = (adapter->scale * C_ABS(dz));
            if (D > 1) {
                dz /= D;
                elapsed += dt / D;
            } else {
                elapsed += dt;
            }
            auto pixel_idx = adapter->where(z);

            if (pixel_idx != -1) {
                if(!canvas[pixel_idx].update_age((offset + j) % steps)) return;
                canvas[pixel_idx].set_color(C_NORM(v), options->speed_factor);
            }
            z += dz;
        } while (elapsed < dt);
    }
}

__global__ void evolve_kernel(
        Canvas * canvas_array, const CanvasAdapter * adapter,
        double speed_factor, double dt,
        complex_t* particles,
        const uint32_t * tile_offsets,
        const float * rand_offsets,
        ComplexFunction_t func,
        const FnVariables * fn_vars,
        int32_t frame_count, uint8_t subtile
){

    complex_t z;
    uint32_t offset;
    Canvas canvas;

    {
        auto tile_idx = threadIdx.x + subtile*blockDim.x;
        auto count = tile_offsets[tile_idx + 1] - tile_offsets[tile_idx];
        auto canvas_idx = blockIdx.x;
        if(canvas_idx >= count) return;
        auto particle_idx = tile_offsets[tile_idx] + canvas_idx;
        z = particles[particle_idx];
        offset = static_cast<uint32_t>(rand_offsets[particle_idx] * (float) frame_count);
        canvas = canvas_array[canvas_idx];
    };

    double elapsed;
    for(int32_t j=0; j<frame_count; j++) {
        elapsed = 0.0;
        do {
            auto pixel_idx = adapter->where(z);
            auto dz = func(z, fn_vars);
            double speed = C_NORM(dz);
            dz *= dt;
            {
                double D = (adapter->scale * C_ABS(dz));
                if (D > 1) {
                    dz /= D;
                    elapsed += dt / D;
                } else {
                    elapsed += dt;
                }
            };
            z += dz;

            if (pixel_idx != -1) {
                if(!canvas[pixel_idx].update_age((offset + j) % frame_count)) return;
                canvas[pixel_idx].set_color(speed, speed_factor);
            }
        } while (elapsed < dt);
    }
}

int get_evolve_regs() {
    cudaFuncAttributes attrs;
    cudaFuncGetAttributes(&attrs, &evolve_kernel);
    return attrs.numRegs;
}

void evolve_gpu(const Configuration * config,
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
    auto d_adapter = devicify(&config->canvas);
    auto d_vars = devicify(&config->vars);

    evolve_kernel<<<canvas_count, 10>>>(
            canvas, d_adapter,
            config->evolution.speed_factor, config->evolution.delta_time,
            particles, tile_offsets, d_rand_floats,
            func, d_vars,
            config->evolution.frame_count, 0);
    CATCH_CUDA_ERROR(cudaDeviceSynchronize());

    evolve_kernel<<<canvas_count, tiles_count>>>(
            canvas, &config->canvas,
            config->evolution.speed_factor, config->evolution.delta_time,
            particles, tile_offsets, d_rand_floats,
            func, d_vars,
            config->evolution.frame_count, 1);

    CATCH_CUDA_ERROR(cudaDeviceSynchronize());

    evolve_kernel<<<canvas_count, tiles_count>>>(
            canvas, &config->canvas,
            config->evolution.speed_factor, config->evolution.delta_time,
            particles, tile_offsets, d_rand_floats,
            func, d_vars,
            config->evolution.frame_count, 2);
    CATCH_CUDA_ERROR(cudaDeviceSynchronize());

    evolve_kernel<<<canvas_count, tiles_count>>>(
            canvas, &config->canvas,
            config->evolution.speed_factor, config->evolution.delta_time,
            particles, tile_offsets, d_rand_floats,
            func, d_vars,
            config->evolution.frame_count, 3);
    CATCH_CUDA_ERROR(cudaDeviceSynchronize());

    cudaFree(d_vars);
    cudaFree(d_adapter);
    cudaFree(d_rand_floats);
    cudaDeviceSynchronize();
    tock_ms(0);
    PRINT_TIME
}

// Divide particle evolution between threads by #pragma omp parallel for.
// Each thread writes particles on its own canvas
void evolve_omp(const Configuration * config, Canvas* canvas,
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
    tock_ms(0)
    PRINT_TIME
}

void evolve_serial(
                const Configuration * config, Canvas canvas,
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
    tock_ms(0)
    PRINT_TIME
}
