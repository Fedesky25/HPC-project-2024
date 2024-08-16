//
// Created by Sofyh02 on 15/08/2024.
//

#ifndef HPC_PROJECT_2024_EVOLUTION_CUH
#define HPC_PROJECT_2024_EVOLUTION_CUH

#include "utils.cuh"
#include "canvas.cuh"

/**
 * Calculates and draws particle evolution on different canvases
 * @param canvas array of canvases
 * @param adapter getting pixel position from complex number
 * @param options (frame number, time step, velocity factor)
 * @param particles
 * @param belonging_tile
 * @param count number of particles in a tile
 * @param func
 * @param variables variables of the function
 */
__global__ void evolve(Canvas* canvas, CanvasAdapter* adapter, EvolutionOptions options, complex_t* particles,
                       uint32_t* belonging_tile, uint32_t* count, complex_t (*func)(complex_t), FnVariables* variables);

#endif //HPC_PROJECT_2024_EVOLUTION_CUH
