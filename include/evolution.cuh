//
// Created by Sofyh02 on 15/08/2024.
//

#ifndef HPC_PROJECT_2024_EVOLUTION_CUH
#define HPC_PROJECT_2024_EVOLUTION_CUH

#include "canvas.cuh"
#include "complex_functions.cuh"

int get_evolve_regs();

/**
 * Calculates and draws particle evolution on different canvases using the GPU
 * @param config device pointer to configuration
 * @param canvas device array of canvases
 * @param canvas_count number of canvases
 * @param particles array of particles ordered by tile
 * @param N_particles number of particles
 * @param fn_choice choice of function
 * @param offsets array of index of fist particle for each tile
 * @param tiles_count number of tiles
 */
void evolve_gpu(
        Configuration * config,
        Canvas* canvas, uint32_t canvas_count,
        complex_t* particles, uint64_t N_particles,
        const uint32_t * tile_offsets, uint32_t tiles_count,
        FunctionChoice fn_choice);

/**
 * Calculates and draws particle evolution on different canvases using OpenMP
 * @param config host pointer to configuration
 * @param canvas host array of canvases
 * @param particles host array of particles
 * @param N_particles number of particles
 * @param fn_choice choice of function
 */
void evolve_omp(
        Configuration* config, Canvas* canvas,
        complex_t* particles, uint64_t N_particles,
        FunctionChoice fn_choice);

/**
 * Calculates and draws particle evolution on different canvases using OpenMP
 * @param config host pointer to configuration
 * @param canvas host canvas
 * @param particles host array of particles
 * @param N_particles number of particles
 * @param fn_choice choice of function
 */
void evolve_serial(
        Configuration* config, Canvas canvas,
        complex_t* particles, uint64_t N_particles,
        FunctionChoice fn_choice);


#endif //HPC_PROJECT_2024_EVOLUTION_CUH
