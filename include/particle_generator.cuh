//
// Created by Sofyh02 on 10/08/2024.
//

#ifndef HPC_PROJECT_2024_PARTICLE_GENERATOR_CUH
#define HPC_PROJECT_2024_PARTICLE_GENERATOR_CUH

#include "utils.cuh"

/**
 * Serially N complex numbers uniformly distributed
 * in the rectangle of lower-left vertex z1 and upper-right vertex z2
 * @param z1 lower-left vertex
 * @param z2 upper-right vertex
 * @param N number of sites
 * @return particle sites
 */
complex_t * particles_serial(complex_t z1, complex_t z2, uint64_t N);

/**
 * OMP VERSION: Generates N complex numbers uniformly distributed
 * in the rectangle of lower-left vertex z1 and upper-right vertex z2
 * @param z1 lower-left vertex
 * @param z2 upper-right vertex
 * @param N number of sites
 * @return particle sites
 */
complex_t* particles_omp(complex_t z1, complex_t z2, int64_t N);

/**
 * CUDA&OMP VERSION: Generates N complex numbers uniformly distributed
 * in the rectangle of lower-left vertex z1 and upper-right vertex z2
 * @param z1 lower-left vertex
 * @param z2 upper-right vertex
 * @param N number of sites
 * @return particle sites
 */
complex_t* particles_gpu(complex_t z1, complex_t z2, int64_t N);

#endif //HPC_PROJECT_2024_PARTICLE_GENERATOR_CUH
