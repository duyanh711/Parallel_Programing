#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <vector>

const int INPUT_SIZE = 784;
const int HIDDEN1_SIZE = 128;
const int HIDDEN2_SIZE = 128;
const int OUTPUT_SIZE = 10;
const int TRAIN_DATA_SIZE = 10000;
const int TEST_DATA_SIZE = 1000;
const int BATCH_SIZE = 4;
const int EPOCHS = 10;
const float LEARNING_RATE = 0.01f;

class NeuralNetwork
{
public:
    float *weightsInputHidden1, *weightsHidden1Hidden2, *weightsHidden2Output;
    float *biasHidden1, *biasHidden2, *biasOutput;
    float *gradWeightsInputHidden1, *gradWeightsHidden1Hidden2, *gradWeightsHidden2Output;
    float *gradBiasHidden1, *gradBiasHidden2, *gradBiasOutput;
};

void initialize_weights(float *weights, int size)
{
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++)
    {
        float random = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Uniform [-1, 1]
        weights[i] = random * scale;                             // Áp dụng scale He
    }
}

void initialize_bias(float *bias, int size)
{
    for (int i = 0; i < size; i++)
    {
        bias[i] = 0.0f;
    }
}

void initializeNetwork(NeuralNetwork *nn)
{
    nn->weightsInputHidden1 = (float *)malloc(HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    nn->weightsHidden1Hidden2 = (float *)malloc(HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    nn->weightsHidden2Output = (float *)malloc(OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));

    nn->biasHidden1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    nn->biasHidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    nn->biasOutput = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    nn->gradWeightsInputHidden1 = (float *)malloc(HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    nn->gradWeightsHidden1Hidden2 = (float *)malloc(HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    nn->gradWeightsHidden2Output = (float *)malloc(OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));

    nn->gradBiasHidden1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    nn->gradBiasHidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    nn->gradBiasOutput = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(nn->weightsInputHidden1, HIDDEN1_SIZE * INPUT_SIZE);
    initialize_weights(nn->weightsHidden1Hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE);
    initialize_weights(nn->weightsHidden2Output, OUTPUT_SIZE * HIDDEN2_SIZE);

    initialize_bias(nn->biasHidden1, HIDDEN1_SIZE);
    initialize_bias(nn->biasHidden2, HIDDEN2_SIZE);
    initialize_bias(nn->biasOutput, OUTPUT_SIZE);
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




void matrix_multiplication(float *A, float *B, float *C, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
          for (int col = 0; col < k; ++col) {
              float value = 0;
              for (int e = 0; e < n; ++e) {
                  value += A[row * n + e] * B[e * k + col];
              }
              C[row * k + col] = value;
          }
      }
}
// Add bias
void bias_forward(float *x, float *bias, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            x[b * size + i] += bias[i];
        }
    }
}

void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

void forwardLayer(float *input, float *weights, float *bias, float *output, int input_size,
                   int output_size, int batch_size, bool use_relu) { 
  
  matrix_multiplication(input, weights, output, batch_size, input_size, output_size);    
  // Add bias1
  bias_forward(output, bias, batch_size, output_size);
  if (use_relu) {
      relu(output, batch_size*output_size); 
  }
}

void softmax_cpu(float *x, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
      int idx = b*size;
        float max_val = x[idx];
        for (int i = 1; i < size; i++) {
            max_val = fmaxf(max_val, x[idx + i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[idx + i] = expf(x[idx + i] - max_val);
            sum += x[idx + i];
        }
        for (int i = 0; i < size; i++) {
            x[idx + i] = fmaxf(x[idx + i] / sum, 1e-7f);
        }
    }
}

void forward_pass(float* input, float* hidden1, float* hidden2,
                  float* output, NeuralNetwork &nn,
                  int batch_size, int start_idx)
{
    // Layer 1
    forwardLayer(input, nn.weightsInputHidden1, nn.biasHidden1,
                 hidden1, INPUT_SIZE, HIDDEN1_SIZE, batch_size, true);

    // Layer 2
    forwardLayer(hidden1, nn.weightsHidden1Hidden2, nn.biasHidden2,
                 hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE, batch_size, true);

    // Output layer
    forwardLayer(hidden2, nn.weightsHidden2Output, nn.biasOutput,
                 output, HIDDEN2_SIZE, OUTPUT_SIZE, batch_size, false);

    // Softmax
    softmax_cpu(output, batch_size, OUTPUT_SIZE);
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


void train(NeuralNetwork *nn, float *X_train, int *y_train)
{
    // Allocate device memory
    float *d_X_train, *d_hidden1, *d_hidden2, *d_output;
    float *d_del_output, *d_d_ReLU_out2, *d_d_ReLU_out1;
    int *d_y_train;

    float *hidden1 = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TRAIN_DATA_SIZE / BATCH_SIZE;

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {

        float total_loss = 0.0f;
        int correct = 0;

        for (int batch = 0; batch < num_batches; batch++)
        {
            int start_idx = batch * BATCH_SIZE;

            // Forward pass
            forward_pass(
                X_train + start_idx * INPUT_SIZE, hidden1, hidden2, output, nn, BATCH_SIZE, start_idx
            );
            // Get results and compute metrics
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

int main()
{
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
}