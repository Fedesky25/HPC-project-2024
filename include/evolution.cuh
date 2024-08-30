//
// Created by Sofyh02 on 15/08/2024.
//

#ifndef HPC_PROJECT_2024_EVOLUTION_CUH
#define HPC_PROJECT_2024_EVOLUTION_CUH

#include "utils.cuh"
#include "canvas.cuh"
#include "complex_functions.cuh"

/**
 * Calculates and draws particle evolution on different canvases
 * @param config device pointer to configuration
 * @param canvas array of canvases
 * @param particles array of particles ordered by tile
 * @param N_particles
 * @param offsets array of index of fist particle for each tile
 * @param func
 */
void evolve_gpu(Configuration * config, Canvas* canvas, complex_t* particles, uint64_t N_particles,
                   const uint32_t* offsets, ComplexFunction_t func, uint32_t tiles_count, uint32_t canvas_count);

void evolve_omp(CanvasAdapter* adapter, EvolutionOptions options, complex_t* particles,
                   complex_t (*func)(complex_t, FnVariables*), FnVariables* variables);


#endif //HPC_PROJECT_2024_EVOLUTION_CUH
