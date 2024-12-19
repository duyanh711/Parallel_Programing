#include "kernels.cuh"
#include <cmath>

__global__ void softmax_kernel(float *data, int batch_count, int feature_count)
{
    int batch_idx = blockIdx.x;
    if (batch_idx < batch_count)
    {
        // Find maximum value in the batch for numerical stability
        float max_val = data[batch_idx * feature_count];
        for (int i = 1; i < feature_count; i++)
        {
            max_val = fmaxf(max_val, data[batch_idx * feature_count + i]);
        }

        // Compute exponentials and their sum
        float exp_sum = 0.0f;
        for (int i = 0; i < feature_count; i++)
        {
            data[batch_idx * feature_count + i] = expf(data[batch_idx * feature_count + i] - max_val);
            exp_sum += data[batch_idx * feature_count + i];
        }

        // Normalize to compute softmax
        for (int i = 0; i < feature_count; i++)
        {
            data[batch_idx * feature_count + i] = fmaxf(data[batch_idx * feature_count + i] / exp_sum, 1e-7f);
        }
    }
}

__global__ void forward_layer_kernel(float *inputs, float *weights, float *biases,
                                     float *outputs, int input_dim, int output_dim,
                                     int batch_count, bool apply_relu)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < batch_count * output_dim)
    {
        int batch_idx = global_idx / output_dim;
        int output_idx = global_idx % output_dim;

        float activation = 0.0f;

        for (int i = 0; i < input_dim; i++)
        {
            activation += inputs[batch_idx * input_dim + i] * weights[i * output_dim + output_idx];
        }

        activation += biases[output_idx];
        outputs[global_idx] = apply_relu ? fmaxf(0.0f, activation) : activation;
    }
}

__global__ void compute_gradients_kernel(float *inputs, float *output_deltas,
                                         float *weight_gradients, float *bias_gradients,
                                         int batch_count, int input_dim, int output_dim)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx < input_dim * output_dim)
    {
        int input_idx = global_idx / output_dim;
        int output_idx = global_idx % output_dim;

        float weight_grad = 0.0f;
        if (input_idx == 0)
        {
            bias_gradients[output_idx] = 0.0f;
        }

        for (int batch_idx = 0; batch_idx < batch_count; batch_idx++)
        {
            weight_grad += inputs[batch_idx * input_dim + input_idx] * output_deltas[batch_idx * output_dim + output_idx];
            if (input_idx == 0)
            {
                bias_gradients[output_idx] += output_deltas[batch_idx * output_dim + output_idx];
            }
        }
        weight_gradients[global_idx] = weight_grad;
    }
}

__global__ void compute_delta_relu_kernel(float *output_deltas, float *weights,
                                          float *input_layer, float *input_deltas,
                                          int batch_count, int output_dim, int input_dim)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx < batch_count * input_dim)
    {
        int batch_idx = global_idx / input_dim;
        int input_idx = global_idx % input_dim;

        input_deltas[global_idx] = 0.0f;
        for (int output_idx = 0; output_idx < output_dim; output_idx++)
        {
            input_deltas[global_idx] += output_deltas[batch_idx * output_dim + output_idx] * weights[input_idx * output_dim + output_idx];
        }
        input_deltas[global_idx] *= (input_layer[global_idx] > 0.0f);
    }
}

__global__ void update_weights_kernel(float *weights, float *weight_gradients,
                                      float *biases, float *bias_gradients,
                                      int output_dim, int input_dim)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < input_dim * output_dim)
    {
        weights[global_idx] -= LEARNING_RATE * weight_gradients[global_idx];
    }

    if (global_idx < output_dim)
    {
        biases[global_idx] -= LEARNING_RATE * bias_gradients[global_idx];
    }
}
