//
// Created by Sofyh02 on 10/08/2024.
//

#ifndef HPC_PROJECT_2024_PARTICLE_GENERATOR_CUH
#define HPC_PROJECT_2024_PARTICLE_GENERATOR_CUH

#include "utils.cuh"

/**
 * Generates N complex numbers uniformly distributed
 * in the rectangle of upper-left vertex z1 and lower-right vertex z2
 * @param z1 lower-left vertex
 * @param z2 upper-right vertex
 * @param sites
 * @param N number of sites
 */
void particles(complex_t z1, complex_t z2, complex_t sites[], uint64_t N);

#endif //HPC_PROJECT_2024_PARTICLE_GENERATOR_CUH
