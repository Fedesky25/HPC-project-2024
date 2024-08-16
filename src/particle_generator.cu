//
// Created by Sofyh02 on 10/08/2024.
//

#include "particle_generator.cuh"
#include <cuda/std/cmath>
#include <chrono>
#include <curand.h>
#include <random>
#include <thrust/sort.h>
#include <omp.h>
#include "lower_bound.cuh"

#include <iostream>
#include <iomanip>

#define SETUP_CPU \
    auto sites = (complex_t*) malloc(N * sizeof(complex_t));                                        \
    auto n_density = 128*(int64_t)N;                                                                \
    auto density = (complex_t*) malloc(n_density * sizeof(complex_t));                              \
    auto nearest = (uint32_t*) malloc(n_density * sizeof(uint32_t));

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
    SETUP_CPU

    std::cout << "Random initial numbers (" << N << " sites, " << n_density << " density points)";
    timers(3) tick(0)
    rand_complex(z1, z2, sites, N);
    rand_complex(z1, z2, density, n_density);
    tock_ms(0) std::cout << " generated in " << t_elapsed << "ms" << std::endl;

    for(uint16_t i=0; i<20; i++){  // Iterating to convergence
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


int rand_complex_omp(
        complex_t min, complex_t max,
        complex_t * sites, uint32_t N_sites,
        complex_t * density, uint64_t N_density
) {
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    };
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto generators = new std::default_random_engine[num_threads];
    auto dist_real = new std::uniform_real_distribution<double>[num_threads];
    auto dist_imag = new std::uniform_real_distribution<double>[num_threads];
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        generators[t].seed(seed + t);
        auto re = std::uniform_real_distribution<double>::param_type(min.real(), max.real());
        auto im = std::uniform_real_distribution<double>::param_type(min.real(), max.real());
        dist_real[t].param(re);
        dist_imag[t].param(im);
    }
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0; i<N_sites; i++){
        auto t = omp_get_thread_num();
        sites[i].real(real(min) + dist_real[t](generators[t]));
        sites[i].imag(imag(min) + dist_imag[t](generators[t]));
    }
    #pragma omp parallel for schedule(static)
    for(uint64_t i=0; i<N_density; i++){
        auto t = omp_get_thread_num();
        density[i].real(real(min) + dist_real[t](generators[t]));
        density[i].imag(imag(min) + dist_imag[t](generators[t]));
    }
    delete[] generators;
    delete[] dist_real;
    delete[] dist_imag;
    return num_threads;
}


complex_t* particles_omp(complex_t z1, complex_t z2, uint32_t N){
    SETUP_CPU

    std::cout << "Random initial numbers (" << N << " sites, " << n_density << " density points)";  \
    timers(3) tick(0)
    rand_complex_omp(min, max, sites, N, density, n_density);
    tock_ms(0) std::cout << " generated in " << t_elapsed << "ms" << std::endl;

    auto count = (int64_t*) malloc(N * sizeof(int64_t));
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
    SETUP_CPU
    complex_t *d_density, *d_sites;
    uint32_t *d_nearest;
    cudaMalloc((void **)&d_density, n_density * sizeof (complex_t));
    cudaMalloc((void **)&d_sites, N * sizeof (complex_t));
    cudaMalloc((void **)&d_nearest, n_density * sizeof (uint32_t));

    std::cout << "Random initial numbers (" << N << " sites, " << n_density << " density points)";  \
    timers(3) tick(0)
    rand_complex_omp(z1, z2, sites, N, density, n_density);
    tock_ms(0) std::cout << " generated in " << t_elapsed << "ms" << std::endl;

    float times[4];
    auto count = (int64_t*) malloc(N * sizeof(int64_t));
    auto M = ((n_density-1) >> 10) + 1; // (n_density + 1023) / 1024 = (n_density-1)/ 2^(10)
    cudaMemcpy(d_density, density, n_density * sizeof (complex_t), cudaMemcpyHostToDevice);

    PRINT("Arranging particles: ");
    for(int16_t i=0; i<30; i++){  // Iterating to convergence
        cudaMemcpy(d_sites, sites, N * sizeof (complex_t), cudaMemcpyHostToDevice);
        compute_nearest<<<M, 1024>>>(d_density, n_density, d_sites, N, d_nearest);
        cudaMemcpy(nearest, d_nearest, n_density * sizeof (uint32_t), cudaMemcpyDeviceToHost);
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

__global__ void scale_complex(double real, double imag, complex_t offset, complex_t * data, uint64_t N) {
    auto index = threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
    if(index >= N) return;
    auto z = data[index];
    z.real(z.real() * real);
    z.imag(z.imag() * imag);
    z += offset;
    data[index] = z;
}

/**
 * Updates the position of the sites. To be called after density points are sorted by nearest site
 * @param density_points
 * @param N_density
 * @param sites
 * @param N_sites
 * @param nearest
 */
__global__ void update_sites(
        complex_t * density_points, uint32_t N_density,
        complex_t * sites, uint32_t N_sites,
        uint32_t * nearest
) {
    auto site_index = threadIdx.x + blockIdx.x * blockDim.x;
    if(site_index >= N_sites) return;

    int64_t count = 0;
    complex_t sum = 0.0;
    uint32_t dpoint_index = lower_bound(site_index, nearest, N_density);
    while(nearest[dpoint_index] == site_index) {
        sum += density_points[dpoint_index];
        dpoint_index++;
        count++;
    }
    if(count > 0) sites[site_index] = sum / (double) count;
}

complex_t* particles_gpu(complex_t z1, complex_t z2, uint32_t N){
    int64_t n_density = 128*N;
    auto M = ((N-1) >> 10) + 1; // (n_density + 1023) / 1024 = (n_density-1)/ 2^(10)
    auto D = ((n_density-1) >> 10) + 1;

    complex_t *d_density, *d_sites;
    uint32_t *d_nearest;
    cudaMalloc((void **)&d_density, n_density * sizeof (complex_t));
    cudaMalloc((void **)&d_sites, N * sizeof (complex_t));
    cudaMalloc((void **)&d_nearest, n_density * sizeof (uint32_t));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    auto deltaReal = z2.real()-z1.real();
    auto deltaImag = z2.imag()-z1.imag();

    curandGenerateUniformDouble(gen, (double*) d_sites, N*2);
    scale_complex<<<M, 1024>>>(deltaReal, deltaImag, z1, d_sites, N);
    curandGenerateUniformDouble(gen, (double*) d_density, n_density*2);
    scale_complex<<<D, 1024>>>(deltaReal, deltaImag, z1, d_density, N);

    PRINT("Arranging particles: ");
    for(int16_t i=0; i<20; i++){  // Iterating to convergence
        compute_nearest<<<D, 1024>>>(d_density, n_density, d_sites, N, d_nearest);
        thrust::sort_by_key(thrust::device, d_nearest, d_nearest + n_density, d_density);
        update_sites<<<M, 1024>>>(d_density, n_density, d_sites, N, d_nearest);
        PRINT(' ' << i+1)
    }
    PRINTLN(' ');

    auto sites = (complex_t*) malloc(N * sizeof(complex_t));
    cudaMemcpy(sites, d_sites, N * sizeof (complex_t), cudaMemcpyDeviceToHost);

    cudaFree(d_sites);
    cudaFree(d_density);
    cudaFree(d_nearest);

    return sites;
}