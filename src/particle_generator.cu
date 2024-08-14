//
// Created by Sofyh02 on 10/08/2024.
//

#include "particle_generator.cuh"
#include <cstdlib>
#include <omp.h>
#include <cuda/std/cmath>
#include <random>
#include <chrono>


/**
 * Generates M random complex numbers in the rectangle
 * @param z1 lower-left vertex
 * @param z2 upper-right vertex
 * @param rdm array of points
 * @param M number of sites
 */
void rand_complex(complex_t z1, complex_t z2, complex_t * rdm, uint64_t M) {
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for(uint64_t i=0; i<M; i++){
        rdm[i].real(real(z1) + distribution(generator)*(real(z2)-real(z1)));
        rdm[i].imag(imag(z1) + distribution(generator)*(imag(z2)-imag(z1)));
    }
}


complex_t* particles_serial(complex_t z1, complex_t z2, uint32_t N){

    auto sites = (complex_t*) malloc(N * sizeof(complex_t));

    auto n_density = 128*(int64_t)N;
    auto density = (complex_t*) malloc(n_density * sizeof(complex_t));
    rand_complex(z1, z2, density, n_density); // Random complex density points

    rand_complex(z1, z2, sites, N); // Random complex sites

    // Moving sites
    auto nearest = (uint32_t*) malloc(n_density * sizeof(uint32_t));// To save nearest sites
    for(uint16_t i=0; i<50; i++){  // Iterating to convergence
        for(uint64_t j=0; j<n_density; j++){ // Iterating on density points
            double current, min = INFINITY;
            for(uint64_t k=0; k<N; k++){  // Iterating on sites to save the nearest site to each density point
                current = cuda::std::norm(density[j]-sites[k]);
                if(current < min){
                   nearest[j] = k;
                   min = current;
               }
            }
        } // Here nearest[] has been filled in
        for(uint64_t k=0; k<N; k++){ // Iterating on sites
            double ctr = 0;
            sites[k] = 0;
            for(uint64_t j=0; j<n_density; j++){ // Iterating on nearest
                if(nearest[j] == k){  // Finding density points associated to the k-th site
                    sites[k] += density[j];
                    ctr++;
                }
            }
            if(ctr != 0) sites[k] /= ctr;
        }
    }
    free(density);
    free(nearest);
    return sites;
}


complex_t* particles_omp(complex_t z1, complex_t z2, uint32_t N){

    auto sites = (complex_t*) malloc(N * sizeof(complex_t));
    auto count = (int64_t*) malloc(N * sizeof(int64_t));
    PRINTLN("Generating " << N << " random sites");
    rand_complex(z1, z2, sites, N); // Random complex sites

    auto n_density = 128*(int64_t)N;
    auto density = (complex_t*) malloc(n_density * sizeof(complex_t));
    PRINTLN("Generating " << n_density << " density points")
    rand_complex(z1, z2, density, n_density); // Random complex density points
    auto nearest = (uint32_t*) malloc(n_density * sizeof(uint32_t));// To save nearest sites

    omp_set_num_threads(10);

    for(int16_t i=0; i<30; i++){  // Iterating to convergence
        PRINTLN("Iteration " << i+1)
        #pragma omp parallel for shared(nearest, density, sites) schedule(static)
        for (int64_t j = 0; j < n_density; j++) { // Iterating on density points
            double current, min = INFINITY;
            for (int64_t k = 0; k < N; k++) {  // Iterating on sites to save the nearest site to each density point
                current = cuda::std::norm(density[j] - sites[k]);
                if (current < min) {
                    nearest[j] = k;
                    min = current;
                }
            }
        }
        for(int64_t k=0; k<N; k++) {
            sites[k] = 0;
            count[k] = 0;
        }
        for(int64_t j=0; j<n_density; j++){ // Iterating on nearest
            sites[nearest[j]] += density[j];
            count[nearest[j]] ++;
        }

        #pragma omp parallel for shared(sites, count) schedule(static)
        for (int64_t k = 0; k < N; k++) {
            if (count[k] == 0) rand_complex(z1, z2, sites + k, 1);
            else sites[k] /= (double) count[k];
        }
    }
    free(density);
    free(nearest);
    free(count);
    return sites;
}

__global__ void compute_nearest(
        complex_t * density_points, int64_t N_density,
        complex_t * sites, uint32_t N_sites,
        uint32_t * nearest
){
    auto index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index >= N_density) return;
    double current, min = INFINITY;
    complex_t z = density_points[index];
    uint32_t n;
    for (uint32_t k = 0; k < N_sites; k++) {  // Iterating on sites to save the nearest site to each density point
        current = cuda::std::norm(z - sites[k]);
        if (current < min) {
            n = k;
            min = current;
        }
    }
    nearest[index] = n;
}

complex_t* particles_mixed(complex_t z1, complex_t z2, uint32_t N){

    auto sites = (complex_t*) malloc(N * sizeof(complex_t));
    auto count = (int64_t*) malloc(N * sizeof(int64_t));
    PRINTLN("Generating " << N << " random sites");
    rand_complex(z1, z2, sites, N); // Random complex sites

    auto n_density = 128*(int64_t)N;
    auto density = (complex_t*) malloc(n_density * sizeof(complex_t));
    PRINTLN("Generating " << n_density << " density points")
    rand_complex(z1, z2, density, n_density); // Random complex density points

    auto nearest = (uint32_t*) malloc(n_density * sizeof(uint32_t));// To save nearest sites

    auto M = ((n_density-1) >> 10) + 1; // (n_density + 1023) / 1024 = (n_density-1)/ 2^(10)
    complex_t *d_density, *d_sites;
    int64_t *d_nearest;
    cudaMalloc((void **)&d_density, n_density * sizeof (complex_t));
    cudaMalloc((void **)&d_sites, N * sizeof (complex_t));
    cudaMalloc((void **)&d_nearest, n_density * sizeof (int64_t));

    cudaMemcpy(d_density, density, n_density * sizeof (complex_t), cudaMemcpyHostToDevice);

    PRINT("Arranging particles: ");
    for(int16_t i=0; i<30; i++){  // Iterating to convergence
        cudaMemcpy(d_sites, sites, N * sizeof (complex_t), cudaMemcpyHostToDevice);
        compute_nearest<<<M, 1024>>>(d_density, n_density, d_sites, N, d_nearest);
        cudaMemcpy(nearest, d_nearest, n_density * sizeof (int64_t), cudaMemcpyDeviceToHost);

        for(int64_t k=0; k<N; k++) {
            sites[k] = 0;
            count[k] = 0;
        }
        for(int64_t j=0; j<n_density; j++){ // Iterating on nearest
            sites[nearest[j]] += density[j];
            count[nearest[j]] ++;
        }

        #pragma omp parallel for shared(sites, count) schedule(static)
        for (int64_t k = 0; k < N; k++) {
            if (count[k] == 0) rand_complex(z1, z2, sites + k, 1);
            else sites[k] /= (double)count[k];
        }
        PRINT(' ' << i+1)
    }
    PRINTLN(' ');
    cudaFree(d_density);
    cudaFree(d_nearest);
    free(density);
    free(nearest);
    free(count);
    return sites;
}