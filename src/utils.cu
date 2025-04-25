//
// Created by feder on 25/04/2025.
//

#include "utils.cuh"

unsigned KernelSizes::SM_count;

void KernelSizes::cover(unsigned N) {
    grid = SM_count;
    block = 1 + (N - 1)/grid;
    if(block > 1024) {
        block = 1024;
        grid = 1 + ((N-1) >> 10);
    }
}

void KernelSizes::warp_cover(unsigned N) {
    grid = SM_count;
    block = 1 + (N - 1)/(32 * grid);
    if(block > 32) {
        block = 32;
        grid = 1 + ((N-1) >> 10);
    }
    block <<= 5;
}

void KernelSizes::set_SM() {
    int v;
    CATCH_CUDA_ERROR(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0));
    SM_count = v;
}
