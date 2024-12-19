#include "kernels.cuh"
#include <cmath>

__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x;
    if (b < batch_size) {
        // Find max value for numerical stability
        float max_val = x[b * size];
        for (int i = 1; i < size; i++) {
            max_val = fmaxf(max_val, x[b * size + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        // Normalize
        for (int i = 0; i < size; i++) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

__global__ void forwardLayerKernel(float *input, float *weights, float *bias, 
                                  float *output, int input_size, int output_size, 
                                  int batch_size, bool use_relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_size) {
        int b = idx / output_size;
        int j = idx % output_size;
        output[idx] = 0.0f;
        for (int k = 0; k < input_size; k++) {
            output[idx] += input[b * input_size + k] * weights[k * output_size + j];
        }
        output[idx] = output[idx] + bias[j];
        if (use_relu) {
            output[idx] = fmaxf(0.0f, output[idx]);
        }
    }
}

__global__ void compute_gradients_kernel(float *input, float *delta, 
                                       float *grad_weights, float *grad_bias,
                                       int batch_size, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size * output_size) {
        int i = idx / output_size;
        int j = idx % output_size;

        float grad_w = 0.0f;
        if (i == 0)
            grad_bias[j] = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_w += input[b * input_size + i] * delta[b * output_size + j];
            if (i == 0)
                grad_bias[j] += delta[b * output_size + j];
        }
        grad_weights[idx] = grad_w;
    }
}

__global__ void compute_delta_relu_kernel(float *relu_del_out, float *weights,
                                        float *input_layer, float *relu_del,
                                        int batch_size, int output_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * input_size) {
        int i = idx / input_size;
        int j = idx % input_size;

        relu_del[idx] = 0.0f;
        for (int k = 0; k < output_size; k++) {
            relu_del[idx] += relu_del_out[i * output_size + k] * weights[j * output_size + k];
        }
        relu_del[idx] *= (input_layer[idx] > 0.0f);
    }
}

__global__ void update_weights_kernel(float *weights, float *grad_weights,
                                    float *bias, float *grad_bias,
                                    int output_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_size * output_size) {
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
    }

    if (idx < output_size) {
        bias[idx] -= LEARNING_RATE * grad_bias[idx];
    }
}
