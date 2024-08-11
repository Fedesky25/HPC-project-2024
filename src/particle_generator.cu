//
// Created by Sofyh02 on 10/08/2024.
// SERIAL VERSION

#include "particle_generator.cuh"
#include <ctime>
#include <cstdlib>
#include <cmath>


/**
 * Generates M random complex numbers in the rectangle
 * @param z1 lower-left vertex
 * @param z2 upper-right vertex
 * @param rdm array of points
 * @param M number of sites
 */
void rand_complex(complex_t z1, complex_t z2, complex_t * rdm, uint64_t M) {
    srand(time(NULL));
    for(uint64_t i=0; i<M; i++){
        rdm[i].real(real(z1) + ((double)rand()/(double)RAND_MAX)*(real(z2)-real(z1)));
        rdm[i].imag(imag(z1) + ((double)rand()/(double)RAND_MAX)*(imag(z2)-imag(z1)));
    }
}

void particles(complex_t z1, complex_t z2, complex_t sites[], uint64_t N){

    uint64_t n_density = 128*N;
    auto density = (complex_t*) malloc(n_density * sizeof(complex_t));
    rand_complex(z1, z2, density, n_density); // Random complex density points

    rand_complex(z1, z2, sites, N); // Random complex sites

    // Moving sites
    auto nearest = (uint64_t*) malloc(n_density * sizeof(uint64_t));// To save nearest sites
    for(uint16_t i=0; i<50; i++){  // Iterating to convergence
        for(uint64_t j=0; j<n_density; j++){ // Iterating on density points
            double current, min = INFINITY;
            for(uint64_t k=0; k<N; k++){  // Iterating on sites to save the nearest site to each density point
                current = std::norm(density[j]-sites[k]);
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
}