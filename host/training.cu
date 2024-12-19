#include "training.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctime>

void train(NeuralNetwork *nn, float *X_train, int *y_train)
{
    float *hidden1 = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    float *del_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    float *d_ReLU_out1 = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *d_ReLU_out2 = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));

    int num_batches = TRAIN_DATA_SIZE / BATCH_SIZE;

    clock_t start = clock(), end;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        float total_loss = 0.0f;
        int correct = 0;

        for (int batch = 0; batch < num_batches; batch++)
        {
            int start_idx = batch * BATCH_SIZE;

            forward_pass(X_train + start_idx * INPUT_SIZE, hidden1, hidden2,
                         output, nn, BATCH_SIZE, start_idx);

            total_loss += compute_loss(output, &y_train[start_idx], BATCH_SIZE);
            correct += validatePredictions(output, y_train, start_idx, BATCH_SIZE, OUTPUT_SIZE);

            for (int b = 0; b < BATCH_SIZE; b++)
            {
                for (int i = 0; i < OUTPUT_SIZE; i++)
                {
                    del_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i] -
                                                      ((i == y_train[start_idx + b]) ? 1.0f : 0.0f);
                }
            }

            backward_pass(X_train + start_idx * INPUT_SIZE, hidden1, hidden2, output,
                          del_output, d_ReLU_out1, d_ReLU_out2, nn, BATCH_SIZE, start_idx);

            updateWeights(nn->weightsInputHidden1, nn->gradWeightsInputHidden1,
                          nn->biasHidden1, nn->gradBiasHidden1,
                          HIDDEN1_SIZE, INPUT_SIZE, LEARNING_RATE);
            updateWeights(nn->weightsHidden1Hidden2, nn->gradWeightsHidden1Hidden2,
                          nn->biasHidden2, nn->gradBiasHidden2,
                          HIDDEN2_SIZE, HIDDEN1_SIZE, LEARNING_RATE);
            updateWeights(nn->weightsHidden2Output, nn->gradWeightsHidden2Output,
                          nn->biasOutput, nn->gradBiasOutput,
                          OUTPUT_SIZE, HIDDEN2_SIZE, LEARNING_RATE);
        }
        end = clock();
        printf("Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%, Time: %.2f seconds\n",
               epoch + 1, EPOCHS,
               total_loss / num_batches,
               100.0f * correct / TRAIN_DATA_SIZE,
               (float)(end - start) / CLOCKS_PER_SEC);
        start = clock();
    }

    free(hidden1);
    free(hidden2);
    free(output);
    free(del_output);
    free(d_ReLU_out1);
    free(d_ReLU_out2);
}

void test(NeuralNetwork *nn, float *X_test, int *y_test)
{
    float *hidden1 = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TEST_DATA_SIZE / BATCH_SIZE;
    int correct = 0;

    for (int batch = 0; batch < num_batches; batch++)
    {
        int start_idx = batch * BATCH_SIZE;
        forward_pass(X_test + start_idx * INPUT_SIZE, hidden1, hidden2,
                     output, nn, BATCH_SIZE, start_idx);
        correct += validatePredictions(output, y_test, start_idx, BATCH_SIZE, OUTPUT_SIZE);
    }

    printf("Test Accuracy: %.2f%%\n", 100.0f * correct / TEST_DATA_SIZE);

    free(hidden1);
    free(hidden2);
    free(output);
}


void forward_pass(float *input, float *hidden1, float *hidden2, float *output,
                  NeuralNetwork *nn, int batch_size, int start_idx)
{
    forwardLayer(input, nn->weightsInputHidden1, nn->biasHidden1,
                 hidden1, INPUT_SIZE, HIDDEN1_SIZE, batch_size, true);

    forwardLayer(hidden1, nn->weightsHidden1Hidden2, nn->biasHidden2,
                 hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE, batch_size, true);

    forwardLayer(hidden2, nn->weightsHidden2Output, nn->biasOutput,
                 output, HIDDEN2_SIZE, OUTPUT_SIZE, batch_size, false);

    softmax_cpu(output, batch_size, OUTPUT_SIZE);
}

void backward_pass(float *input, float *hidden1, float *hidden2, float *output,
                   float *del_output, float *d_ReLU_out1, float *d_ReLU_out2,
                   NeuralNetwork *nn, int batch_size, int start_idx)
{
    computeGradients(hidden2, del_output, nn->gradWeightsHidden2Output,
                     nn->gradBiasOutput, batch_size, HIDDEN2_SIZE, OUTPUT_SIZE);

    computeDeltaReLU(del_output, nn->weightsHidden2Output, hidden2,
                     d_ReLU_out2, batch_size, OUTPUT_SIZE, HIDDEN2_SIZE);

    computeGradients(hidden1, d_ReLU_out2, nn->gradWeightsHidden1Hidden2,
                     nn->gradBiasHidden2, batch_size, HIDDEN1_SIZE, HIDDEN2_SIZE);

    computeDeltaReLU(d_ReLU_out2, nn->weightsHidden1Hidden2, hidden1,
                     d_ReLU_out1, batch_size, HIDDEN2_SIZE, HIDDEN1_SIZE);

    computeGradients(input, d_ReLU_out1, nn->gradWeightsInputHidden1,
                     nn->gradBiasHidden1, batch_size, INPUT_SIZE, HIDDEN1_SIZE);
}

void updateWeights(float *weights, const float *grad_weights,
                   float *bias, const float *grad_bias,
                   int output_size, int input_size, float learning_rate)
{
    for (int i = 0; i < input_size * output_size; i++)
    {
        weights[i] -= learning_rate * grad_weights[i];
    }
    for (int i = 0; i < output_size; i++)
    {
        bias[i] -= learning_rate * grad_bias[i];
    }
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

void forwardLayer(float *input, float *weights, float *bias, float *output,
                  int input_size, int output_size, int batch_size, bool use_relu)
{
    for (int b = 0; b < batch_size; b++)
    {
        for (int j = 0; j < output_size; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < input_size; k++)
            {
                sum += input[b * input_size + k] * weights[k * output_size + j];
            }
            sum += bias[j];
            output[b * output_size + j] = use_relu ? fmaxf(0.0f, sum) : sum;
        }
    }
}

void softmax_cpu(float *x, int batch_size, int size)
{
    for (int b = 0; b < batch_size; b++)
    {
        float max_val = x[b * size];
        for (int i = 1; i < size; i++)
        {
            max_val = fmaxf(max_val, x[b * size + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        for (int i = 0; i < size; i++)
        {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

// Backward propagation functions
void computeGradients(float *input, float *delta, float *grad_weights,
                      float *grad_bias, int batch_size, int input_size, int output_size)
{
    memset(grad_bias, 0, output_size * sizeof(float));
    memset(grad_weights, 0, input_size * output_size * sizeof(float));

    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < input_size; i++)
        {
            for (int j = 0; j < output_size; j++)
            {
                grad_weights[i * output_size + j] += input[b * input_size + i] * delta[b * output_size + j];
                if (i == 0)
                {
                    grad_bias[j] += delta[b * output_size + j];
                }
            }
        }
    }
}

void computeDeltaReLU(float *relu_del_out, float *weights, float *input_layer,
                      float *relu_del, int batch_size, int output_size, int input_size)
{
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < input_size; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < output_size; j++)
            {
                sum += relu_del_out[b * output_size + j] * weights[i * output_size + j];
            }
            relu_del[b * input_size + i] = sum * (input_layer[b * input_size + i] > 0.0f);
        }
    }
}

