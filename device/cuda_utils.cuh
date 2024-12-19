#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA error checking macro
#define CHECK_CUDA_CALL(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n", \
                    __FILE__, __LINE__, error, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    }

// Timer structure
struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer();
    ~GpuTimer();
    void Start();
    void Stop();
    float Elapsed();
}; 