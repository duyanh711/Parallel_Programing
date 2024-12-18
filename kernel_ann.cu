#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>

// Constants for neural network structure and training
constexpr int INPUT_SIZE = 784;
constexpr int HIDDEN1_SIZE = 128;
constexpr int HIDDEN2_SIZE = 128;
constexpr int OUTPUT_SIZE = 10;
constexpr int TRAIN_DATA_SIZE = 10000;
constexpr int TEST_DATA_SIZE = 1000;
constexpr int BATCH_SIZE = 4;
constexpr int EPOCHS = 10;
constexpr float LEARNING_RATE = 0.01f;

// Neural Network class definition
class NeuralNetwork
{
public:
    float *weightsInputHidden1, *weightsHidden1Hidden2, *weightsHidden2Output;
    float *biasHidden1, *biasHidden2, *biasOutput;
    float *gradWeightsInputHidden1, *gradWeightsHidden1Hidden2, *gradWeightsHidden2Output;
    float *gradBiasHidden1, *gradBiasHidden2, *gradBiasOutput;
};

// Helper functions
int validatePredictions(const float *output, const int *labels, int startIdx, int batchSize, int outputSize)
{
    int correct = 0;
    for (int i = 0; i < batchSize; i++)
    {
        int predicted = 0;
        for (int j = 1; j < outputSize; j++)
        {
            if (output[i * outputSize + j] > output[i * outputSize + predicted])
            {
                predicted = j;
            }
        }
        if (predicted == labels[startIdx + i])
        {
            correct++;
        }
    }
    return correct;
}

float compute_loss(const float *output, const int *labels, int batch_size)
{
    float loss = 0.0f;
    for (int i = 0; i < batch_size; i++)
    {
        int label = labels[i];
        loss -= logf(output[i * OUTPUT_SIZE + label] + 1e-7f);
    }
    return loss / batch_size;
}

// CUDA error checking macro
#define CHECK_CUDA_CALL(call)                                              \
    {                                                                      \
        const cudaError_t error = call;                                    \
        if (error != cudaSuccess)                                          \
        {                                                                  \
            fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n",   \
                    __FILE__, __LINE__, error, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

// Timer structure
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Initialize weights with Xavier initialization
void initWeights(float *weights, int size)
{
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++)
    {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

// Initialize biases to zero
void initBias(float *bias, int size)
{
    for (int i = 0; i < size; i++)
    {
        bias[i] = 0.0f;
    }
}

// Allocate and initialize the neural network
void initializeNetwork(NeuralNetwork *nn)
{
    // Allocate memory on the device
    CHECK_CUDA_CALL(cudaMalloc(&nn->weightsInputHidden1, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&nn->weightsHidden1Hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&nn->weightsHidden2Output, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&nn->biasHidden1, HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&nn->biasHidden2, HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&nn->biasOutput, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA_CALL(cudaMalloc(&nn->gradWeightsInputHidden1, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&nn->gradWeightsHidden1Hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&nn->gradWeightsHidden2Output, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&nn->gradBiasHidden1, HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&nn->gradBiasHidden2, HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&nn->gradBiasOutput, OUTPUT_SIZE * sizeof(float)));

    // Host memory for initialization
    std::vector<float> hostWeightsInputHidden1(HIDDEN1_SIZE * INPUT_SIZE);
    std::vector<float> hostWeightsHidden1Hidden2(HIDDEN2_SIZE * HIDDEN1_SIZE);
    std::vector<float> hostWeightsHidden2Output(OUTPUT_SIZE * HIDDEN2_SIZE);
    std::vector<float> hostBiasHidden1(HIDDEN1_SIZE);
    std::vector<float> hostBiasHidden2(HIDDEN2_SIZE);
    std::vector<float> hostBiasOutput(OUTPUT_SIZE);

    initWeights(hostWeightsInputHidden1.data(), HIDDEN1_SIZE * INPUT_SIZE);
    initWeights(hostWeightsHidden1Hidden2.data(), HIDDEN2_SIZE * HIDDEN1_SIZE);
    initWeights(hostWeightsHidden2Output.data(), OUTPUT_SIZE * HIDDEN2_SIZE);
    initBias(hostBiasHidden1.data(), HIDDEN1_SIZE);
    initBias(hostBiasHidden2.data(), HIDDEN2_SIZE);
    initBias(hostBiasOutput.data(), OUTPUT_SIZE);

    CHECK_CUDA_CALL(cudaMemcpy(nn->weightsInputHidden1, hostWeightsInputHidden1.data(), HIDDEN1_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(nn->weightsHidden1Hidden2, hostWeightsHidden1Hidden2.data(), HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(nn->weightsHidden2Output, hostWeightsHidden2Output.data(), OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(nn->biasHidden1, hostBiasHidden1.data(), HIDDEN1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(nn->biasHidden2, hostBiasHidden2.data(), HIDDEN2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(nn->biasOutput, hostBiasOutput.data(), OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
}

__global__ void softmax_kernel(float *x, int batch_size, int size)
{
    int b = blockIdx.x;
    if (b < batch_size)
    {
        // Find max value for numerical stability
        float max_val = x[b * size];
        for (int i = 1; i < size; i++)
        {
            max_val = fmaxf(max_val, x[b * size + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        // Normalize
        for (int i = 0; i < size; i++)
        {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}
__global__ void compute_gradients_kernel(float *input, float *delta, float *grad_weights, float *grad_bias, int batch_size, int input_size, int output_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size * output_size)
    {
        int i = idx / output_size;
        int j = idx % output_size;

        float grad_w = 0.0f;
        if (i == 0)
            grad_bias[j] = 0.0f;
        for (int b = 0; b < batch_size; b++)
        {
            grad_w += input[b * input_size + i] * delta[b * output_size + j];
            if (i == 0)
                grad_bias[j] += delta[b * output_size + j];
        }
        grad_weights[idx] = grad_w;
    }
}

__global__ void compute_delta_relu_kernel(float *relu_del_out, float *weights, float *input_layer, float *relu_del, int batch_size, int output_size, int input_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * input_size)
    {
        int i = idx / input_size;
        int j = idx % input_size;

        relu_del[idx] = 0.0f;
        for (int k = 0; k < output_size; k++)
        {
            relu_del[idx] += relu_del_out[i * output_size + k] * weights[j * output_size + k];
        }

        relu_del[idx] *= (input_layer[idx] > 0.0f);
    }
}

__global__ void forwardLayerKernel(float *input, float *weights, float *bias, float *output, int input_size, int output_size, int batch_size, bool use_relu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_size)
    {
        int b = idx / output_size;
        int j = idx % output_size;
        output[idx] = 0.0f;
        for (int k = 0; k < input_size; k++)
        {
            output[idx] += input[b * input_size + k] * weights[k * output_size + j];
        }
        output[idx] = output[idx] + bias[j];
        if (use_relu)
        {
            output[idx] = fmaxf(0.0f, output[idx]);
        }
    }
}

__global__ void update_weights_kernel(float *weights, float *grad_weights, float *bias, float *grad_bias, int output_size, int input_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_size * output_size)
    {
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
    }

    if (idx < output_size)
    {
        bias[idx] -= LEARNING_RATE * grad_bias[idx];
    }
}

void load_data(const char *filename, float *data, int size)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        printf("Không thể mở file\n");
        return;
    }
    // Tạo bộ nhớ để lưu dữ liệu
    size_t elements_read = fread(data, sizeof(float), size, file);

    // Kiểm tra số lượng phần tử đọc được
    if (elements_read != size)
    {
        printf("Số phần tử đọc được không khớp\n");
    }
    // Đóng file
    fclose(file);
}

void load_labels(const char *filename, int *labels, int size)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        printf("Không thể mở file\n");
        return;
    }
    // Tạo bộ nhớ để lưu dữ liệu
    size_t elements_read = fread(labels, sizeof(int), size, file);

    // Kiểm tra số lượng phần tử đọc được
    if (elements_read != size)
    {
        printf("Số phần tử đọc được không khớp\n");
    }
    // Đóng file
    fclose(file);
}

// Thêm hàm forward pass riêng
void forward_pass(float *d_input, float *d_hidden1, float *d_hidden2, float *d_output,
                  NeuralNetwork *nn, int batch_size, int start_idx)
{
    // Layer 1
    forwardLayerKernel<<<(batch_size * HIDDEN1_SIZE + 255) / 256, 256>>>(
        d_input + start_idx * INPUT_SIZE,
        nn->weightsInputHidden1,
        nn->biasHidden1,
        d_hidden1,
        INPUT_SIZE, HIDDEN1_SIZE,
        batch_size, true);

    // Layer 2
    forwardLayerKernel<<<(batch_size * HIDDEN2_SIZE + 255) / 256, 256>>>(
        d_hidden1,
        nn->weightsHidden1Hidden2,
        nn->biasHidden2,
        d_hidden2,
        HIDDEN1_SIZE, HIDDEN2_SIZE,
        batch_size, true);

    // Output layer
    forwardLayerKernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(
        d_hidden2,
        nn->weightsHidden2Output,
        nn->biasOutput,
        d_output,
        HIDDEN2_SIZE, OUTPUT_SIZE,
        batch_size, false);

    // Softmax
    softmax_kernel<<<batch_size, 1>>>(d_output, batch_size, OUTPUT_SIZE);
}

void backward_pass(float *d_input, float *d_hidden1, float *d_hidden2,
                   float *d_output, float *d_del_output,
                   float *d_d_ReLU_out1, float *d_d_ReLU_out2,
                   NeuralNetwork *nn, int batch_size, int start_idx)
{
    // Compute gradients for output layer
    compute_gradients_kernel<<<(HIDDEN2_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(
        d_hidden2, d_del_output,
        nn->gradWeightsHidden2Output,
        nn->gradBiasOutput,
        batch_size, HIDDEN2_SIZE, OUTPUT_SIZE);

    // Compute gradients for hidden layer 2
    compute_delta_relu_kernel<<<(batch_size * HIDDEN2_SIZE + 255) / 256, 256>>>(
        d_del_output, nn->weightsHidden2Output,
        d_hidden2, d_d_ReLU_out2,
        batch_size, OUTPUT_SIZE, HIDDEN2_SIZE);
    compute_gradients_kernel<<<(HIDDEN1_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(
        d_hidden1, d_d_ReLU_out2,
        nn->gradWeightsHidden1Hidden2,
        nn->gradBiasHidden2,
        batch_size, HIDDEN1_SIZE, HIDDEN2_SIZE);

    // Compute gradients for hidden layer 1
    compute_delta_relu_kernel<<<(batch_size * HIDDEN1_SIZE + 255) / 256, 256>>>(
        d_d_ReLU_out2, nn->weightsHidden1Hidden2,
        d_hidden1, d_d_ReLU_out1,
        batch_size, HIDDEN2_SIZE, HIDDEN1_SIZE);
    compute_gradients_kernel<<<(INPUT_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(
        d_input + start_idx * INPUT_SIZE, d_d_ReLU_out1,
        nn->gradWeightsInputHidden1,
        nn->gradBiasHidden1,
        batch_size, INPUT_SIZE, HIDDEN1_SIZE);
}

void update_weights(NeuralNetwork *nn)
{
    update_weights_kernel<<<(INPUT_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(
        nn->weightsInputHidden1, nn->gradWeightsInputHidden1,
        nn->biasHidden1, nn->gradBiasHidden1,
        HIDDEN1_SIZE, INPUT_SIZE);

    update_weights_kernel<<<(HIDDEN1_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(
        nn->weightsHidden1Hidden2, nn->gradWeightsHidden1Hidden2,
        nn->biasHidden2, nn->gradBiasHidden2,
        HIDDEN2_SIZE, HIDDEN1_SIZE);

    update_weights_kernel<<<(HIDDEN2_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(
        nn->weightsHidden2Output, nn->gradWeightsHidden2Output,
        nn->biasOutput, nn->gradBiasOutput,
        OUTPUT_SIZE, HIDDEN2_SIZE);
}

void train(NeuralNetwork *nn, float *X_train, int *y_train)
{
    // Allocate device memory
    float *d_X_train, *d_hidden1, *d_hidden2, *d_output;
    float *d_del_output, *d_d_ReLU_out2, *d_d_ReLU_out1;
    int *d_y_train;

    // Allocate and copy data
    CHECK_CUDA_CALL(cudaMalloc(&d_X_train, TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_hidden1, BATCH_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_hidden2, BATCH_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_del_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_d_ReLU_out2, BATCH_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_d_ReLU_out1, BATCH_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_y_train, TRAIN_DATA_SIZE * sizeof(int)));

    CHECK_CUDA_CALL(cudaMemcpy(d_X_train, X_train, TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_y_train, y_train, TRAIN_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    int num_batches = TRAIN_DATA_SIZE / BATCH_SIZE;

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        GpuTimer timer;
        timer.Start();

        float total_loss = 0.0f;
        int correct = 0;

        for (int batch = 0; batch < num_batches; batch++)
        {
            int start_idx = batch * BATCH_SIZE;

            // Forward pass
            forward_pass(d_X_train, d_hidden1, d_hidden2, d_output, nn, BATCH_SIZE, start_idx);

            // Get results and compute metrics
            float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            CHECK_CUDA_CALL(cudaMemcpy(output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            total_loss += compute_loss(output, &y_train[start_idx], BATCH_SIZE);
            correct += validatePredictions(output, y_train, start_idx, BATCH_SIZE, OUTPUT_SIZE);

            // Prepare for backprop
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                for (int i = 0; i < OUTPUT_SIZE; i++)
                {
                    output[b * OUTPUT_SIZE + i] -= (i == y_train[start_idx + b]) ? 1.0f : 0.0f;
                }
            }
            CHECK_CUDA_CALL(cudaMemcpy(d_del_output, output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            // Backward pass
            backward_pass(d_X_train, d_hidden1, d_hidden2, d_output, d_del_output,
                          d_d_ReLU_out1, d_d_ReLU_out2, nn, BATCH_SIZE, start_idx);

            // Update weights
            update_weights(nn);

            free(output);
        }

        timer.Stop();
        float elapsed = timer.Elapsed();

        printf("Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%, Time: %.2f ms\n",
               epoch + 1, EPOCHS,
               total_loss / num_batches,
               100.0f * correct / TRAIN_DATA_SIZE,
               elapsed);
    }

    // Free device memory
    cudaFree(d_X_train);
    cudaFree(d_hidden1);
    cudaFree(d_hidden2);
    cudaFree(d_output);
    cudaFree(d_del_output);
    cudaFree(d_d_ReLU_out2);
    cudaFree(d_d_ReLU_out1);
    cudaFree(d_y_train);
}

void test(NeuralNetwork *nn, float *X_test, int *y_test)
{
    // Allocate device memory
    float *d_X_test, *d_hidden1, *d_hidden2, *d_output;
    CHECK_CUDA_CALL(cudaMalloc(&d_X_test, TEST_DATA_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_hidden1, BATCH_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_hidden2, BATCH_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA_CALL(cudaMemcpy(d_X_test, X_test, TEST_DATA_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int num_batches = TEST_DATA_SIZE / BATCH_SIZE;
    int correct = 0;

    for (int batch = 0; batch < num_batches; batch++)
    {
        int start_idx = batch * BATCH_SIZE;

        // Forward pass
        forward_pass(d_X_test, d_hidden1, d_hidden2, d_output, nn, BATCH_SIZE, start_idx);

        // Get results and compute accuracy
        float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
        CHECK_CUDA_CALL(cudaMemcpy(output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        correct += validatePredictions(output, y_test, start_idx, BATCH_SIZE, OUTPUT_SIZE);
        free(output);
    }

    printf("Test Accuracy: %.2f%%\n", 100.0f * correct / TEST_DATA_SIZE);

    // Free device memory
    cudaFree(d_X_test);
    cudaFree(d_hidden1);
    cudaFree(d_hidden2);
    cudaFree(d_output);
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    NeuralNetwork nn;
    initializeNetwork(&nn);

    float *X_train = (float *)malloc(TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int *)malloc(TRAIN_DATA_SIZE * sizeof(int));

    float *X_test = (float *)malloc(TEST_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int *)malloc(TEST_DATA_SIZE * sizeof(int));

    load_data("x_train.bin", X_train, TRAIN_DATA_SIZE * INPUT_SIZE);
    load_labels("y_train.bin", y_train, TRAIN_DATA_SIZE);
    load_data("x_test.bin", X_test, TEST_DATA_SIZE * INPUT_SIZE);
    load_labels("y_test.bin", y_test, TEST_DATA_SIZE);

    // Training
    train(&nn, X_train, y_train);

    // Testing
    test(&nn, X_test, y_test);

    CHECK_CUDA_CALL(cudaFree(nn.weightsInputHidden1));
    CHECK_CUDA_CALL(cudaFree(nn.weightsHidden1Hidden2));
    CHECK_CUDA_CALL(cudaFree(nn.weightsHidden2Output));
    CHECK_CUDA_CALL(cudaFree(nn.biasHidden1));
    CHECK_CUDA_CALL(cudaFree(nn.biasHidden2));
    CHECK_CUDA_CALL(cudaFree(nn.biasOutput));
    CHECK_CUDA_CALL(cudaFree(nn.gradWeightsInputHidden1));
    CHECK_CUDA_CALL(cudaFree(nn.gradWeightsHidden1Hidden2));
    CHECK_CUDA_CALL(cudaFree(nn.gradWeightsHidden2Output));
    CHECK_CUDA_CALL(cudaFree(nn.gradBiasHidden1));
    CHECK_CUDA_CALL(cudaFree(nn.gradBiasHidden2));
    CHECK_CUDA_CALL(cudaFree(nn.gradBiasOutput));

    return 0;
}
