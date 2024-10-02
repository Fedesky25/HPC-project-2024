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

#define LLOYD_ITERATIONS 12

#define SETUP_CPU \
    auto sites = (complex_t*) malloc(N * sizeof(complex_t));           \
    auto n_density = 128*(int64_t)N;                                   \
    auto density = (complex_t*) malloc(n_density * sizeof(complex_t)); \
    auto nearest = (uint32_t*) malloc(n_density * sizeof(uint32_t));   \
    auto count = (uint64_t*) malloc(N * sizeof(uint64_t));

#define PRINT_INITIAL std::cout << "Random initial numbers (" << N << " sites, " << n_density << " density points)";

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
    SETUP_CPU PRINT_INITIAL timers(3) tick(0)
    rand_complex(z1, z2, sites, N);
    rand_complex(z1, z2, density, n_density);
    tock_ms(0) std::cout << " generated in " << t_elapsed << "ms" << std::endl;

    float times[2];
    std::cout << "Lloyd's algorithm:  i | t  (s) | n. c. | s. u." << std::endl << std::fixed;
    tick(0)
    for(uint16_t i=0; i<LLOYD_ITERATIONS; i++){  // Iterating to convergence
        tick(1) tick(2)
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
        tock_s(2) times[0] = t_elapsed; tick(2)
        for(int64_t k=0; k<N; k++) {
            sites[k] = 0;
            count[k] = 0;
        }
        for(int64_t j=0; j<n_density; j++){ // Iterating on nearest
            sites[nearest[j]] += density[j];
            count[nearest[j]] ++;
        }
        for (int64_t k = 0; k < N; k++) {
            if (count[k] == 0) rand_complex(z1, z2, sites + k, 1);
            else sites[k] /= (double) count[k];
        }
        tock_s(2) times[1] = t_elapsed; tock_s(1)
        float m = 100.0f / t_elapsed;
        std::cout << "                   " << std::setw(2) << i+1
                  << " | " << std::setw(6) << std::setprecision(3) << t_elapsed
                  << " | " << std::setw(5) << std::setprecision(2) << times[0]*m
                  << " | " << std::setw(5) << std::setprecision(2) << times[1]*m << std::endl;
//        for(uint64_t k=0; k<N; k++){ // Iterating on sites
//            double ctr = 0;
//            sites[k] = 0;
//            for(uint64_t j=0; j<n_density; j++){ // Iterating on nearest
//                if(nearest[j] == k){  // Finding density points associated to the k-th site
//                    sites[k] += density[j];
//                    ctr++;
//                }
//            }
//            if(ctr != 0) sites[k] /= ctr;
//        }
    }
    tock_s(0) std::cout << "  :: total " << std::setprecision(3) << t_elapsed << 's' << std::endl;
    free(density);
    free(nearest);
    free(count);
    return sites;
}


void rand_complex_omp(
        complex_t min, complex_t max,
        complex_t * sites, uint32_t N_sites,
        complex_t * density, uint64_t N_density
) {
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    #pragma omp parallel
    {
        std::default_random_engine generator(seed + omp_get_thread_num());
        std::uniform_real_distribution<double> dist_real(min.real(), max.real());
        std::uniform_real_distribution<double> dist_imag(min.imag(), max.imag());
        #pragma omp for schedule(static)
        for(int32_t i=0; i<N_sites; i++){
            sites[i].real(dist_real(generator));
            sites[i].imag(dist_imag(generator));
        }
        #pragma omp for schedule(static)
        for(int64_t i=0; i<N_density; i++){
            density[i].real(dist_real(generator));
            density[i].imag(dist_imag(generator));
        }
    };
}


complex_t* particles_omp(complex_t z1, complex_t z2, uint32_t N){
    SETUP_CPU PRINT_INITIAL timers(3) tick(0)
    rand_complex_omp(z1, z2, sites, N, density, n_density);
    tock_ms(0) std::cout << " generated in " << t_elapsed << "ms" << std::endl;

    float times[2];
    std::cout << "Lloyd's algorithm:  i | t  (s) | n. c. | s. u.    using " << std::endl << std::fixed;
    tick(0)
    for(int16_t i=0; i<LLOYD_ITERATIONS; i++){  // Iterating to convergence
        tick(1) tick(2)
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
        tock_s(2) times[0] = t_elapsed; tick(2)
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
        tock_s(2) times[1] = t_elapsed; tock_s(1)
        float m = 100.0f / t_elapsed;
        std::cout << "                   " << std::setw(2) << i+1
                  << " | " << std::setw(6) << std::setprecision(3) << t_elapsed
                  << " | " << std::setw(5) << std::setprecision(2) << times[0]*m
                  << " | " << std::setw(5) << std::setprecision(2) << times[1]*m << std::endl;
    }
    free(density);
    free(nearest);
    free(count);
    tock_s(0)
    std::cout << "  :: total " << std::setprecision(3) << t_elapsed << 's' << std::endl;
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

    PRINT_INITIAL timers(3) tick(0)
    rand_complex_omp(z1, z2, sites, N, density, n_density);
    tock_ms(0) std::cout << " generated in " << t_elapsed << "ms" << std::endl;

    float times[4];
    auto M = ((n_density-1) >> 10) + 1; // (n_density + 1023) / 1024 = (n_density-1)/ 2^(10)
    cudaMemcpy(d_density, density, n_density * sizeof (complex_t), cudaMemcpyHostToDevice);

    std::cout << "Lloyd's algorithm:  i | t (ms) | s. -> | n. c. | n. <- | s. u." << std::endl << std::fixed;
    tick(0)

    for(int16_t i=0; i<LLOYD_ITERATIONS; i++){  // Iterating to convergence
        tick(1) tick(2)
        cudaMemcpy(d_sites, sites, N * sizeof (complex_t), cudaMemcpyHostToDevice);
        tock_ms(2) times[0] = t_elapsed; tick(2)
        compute_nearest<<<M, 1024>>>(d_density, n_density, d_sites, N, d_nearest);
        tock_ms(2) times[1] = t_elapsed; tick(2)
        cudaMemcpy(nearest, d_nearest, n_density * sizeof (uint32_t), cudaMemcpyDeviceToHost);
        tock_ms(2) times[2] = t_elapsed; tick(2)
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
        tock_ms(2) times[3] = t_elapsed;
        tock_ms(1)
        float m = 100.0f / t_elapsed;
        std::cout << "                   " << std::setw(2) << i+1
                  << " | " << std::setw(6) << std::setprecision(1) << t_elapsed
                  << " | " << std::setw(5) << std::setprecision(2) << times[0]*m
                  << " | " << std::setw(5) << std::setprecision(2) << times[1]*m
                  << " | " << std::setw(5) << std::setprecision(2) << times[2]*m
                  << " | " << std::setw(5) << std::setprecision(2) << times[3]*m << std::endl;
    }
    cudaFree(d_density);
    cudaFree(d_nearest);
    free(sites);
    free(density);
    free(nearest);
    free(count);
    tock_s(0)
    std::cout << "  :: total " << std::setprecision(3) << t_elapsed << 's' << std::endl;
    return d_sites;
}

__global__ void scale_complex(double real, double imag, complex_t offset, complex_t * data, uint64_t N) {
    auto increment = (uint64_t) blockDim.x * gridDim.x;
    for(uint64_t i=threadIdx.x+blockIdx.x*blockDim.x; i<N; i+=increment) {
        data[i].real(data[i].real() * real);
        data[i].imag(data[i].imag() * imag);
        data[i] += offset;
    }
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

    PRINT_INITIAL timers(3) tick(0)
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    auto deltaReal = z2.real()-z1.real();
    auto deltaImag = z2.imag()-z1.imag();

    curandGenerateUniformDouble(gen, (double*) d_sites, N*2);
    scale_complex<<<128, 1024>>>(deltaReal, deltaImag, z1, d_sites, N);
    curandGenerateUniformDouble(gen, (double*) d_density, n_density*2);
    scale_complex<<<128, 1024>>>(deltaReal, deltaImag, z1, d_density, n_density);
    cudaDeviceSynchronize();
    tock_ms(0) std::cout << " generated in " << t_elapsed << "ms" << std::endl;

    float times[3];
    std::cout << "Lloyd's algorithm:  i | t (ms) | n. c. | sortk | s. u." << std::endl << std::fixed;
    tick(0)
    for(int16_t i=0; i<20; i++){  // Iterating to convergence
        tick(1) tick(2)
        compute_nearest<<<D, 1024>>>(d_density, n_density, d_sites, N, d_nearest);
        cudaDeviceSynchronize();
        tock_ms(2) times[0] = t_elapsed; tick(2)
        thrust::sort_by_key(thrust::device, d_nearest, d_nearest + n_density, d_density);
        tock_ms(2) times[1] = t_elapsed; tick(2)
        update_sites<<<M, 1024>>>(d_density, n_density, d_sites, N, d_nearest);
        cudaDeviceSynchronize();
        tock_ms(2) times[2] = t_elapsed; tock_ms(1)
        float m = 100.0f / t_elapsed;
        std::cout << "                   " << std::setw(2) << i+1
                  << " | " << std::setw(6) << std::setprecision(1) << t_elapsed
                  << " | " << std::setw(5) << std::setprecision(2) << times[0]*m
                  << " | " << std::setw(5) << std::setprecision(2) << times[1]*m
                  << " | " << std::setw(5) << std::setprecision(2) << times[2]*m << std::endl;
    }
    tock_s(0) std::cout << "  :: total " << std::setprecision(3) << t_elapsed << 's' << std::endl;

    cudaFree(d_density);
    cudaFree(d_nearest);
    return d_sites;
}