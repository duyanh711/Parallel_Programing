#pragma once

#include <cuda_runtime.h>
#include "neural_network.cuh"

// Forward declarations of CUDA kernels
__global__ void softmax_kernel(float *x, int batch_size, int size);

__global__ void forward_layer_kernel(float *input, float *weights, float *bias,
                                     float *output, int input_size, int output_size,
                                     int batch_size, bool use_relu);

__global__ void compute_gradients_kernel(float *input, float *delta,
                                         float *grad_weights, float *grad_bias,
                                         int batch_size, int input_size, int output_size);

__global__ void compute_delta_relu_kernel(float *relu_del_out, float *weights,
                                          float *input_layer, float *relu_del,
                                          int batch_size, int output_size, int input_size);

__global__ void update_weights_kernel(float *weights, float *grad_weights,
                                      float *bias, float *grad_bias,
                                      int output_size, int input_size);