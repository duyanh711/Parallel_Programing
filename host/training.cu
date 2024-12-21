#include "training.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctime>

void Train::train(NeuralNetwork *network, float *training_data, int *labels)
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

            forward_pass(training_data + start_idx * INPUT_SIZE, hidden1, hidden2,
                         output, network, BATCH_SIZE, start_idx);

            total_loss += compute_loss(output, &labels[start_idx], BATCH_SIZE);
            correct += validatePredictions(output, labels, start_idx, BATCH_SIZE, OUTPUT_SIZE);

            compute_output_error(output, &labels[start_idx], del_output, BATCH_SIZE);

            backward_pass(training_data + start_idx * INPUT_SIZE, hidden1, hidden2, output,
                          del_output, d_ReLU_out1, d_ReLU_out2, network, BATCH_SIZE, start_idx);

            update_weights(network);
        }

        end = clock();
        float elapsed = (float)(end - start) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds
        print_training_progress(epoch, total_loss / num_batches,
                                100.0f * correct / TRAIN_DATA_SIZE, elapsed);
        start = clock();
    }

    free(hidden1);
    free(hidden2);
    free(output);
    free(del_output);
    free(d_ReLU_out1);
    free(d_ReLU_out2);
}

void Train::test(NeuralNetwork *network, float *test_data, int *test_labels)
{
    float *hidden1 = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TEST_DATA_SIZE / BATCH_SIZE;
    int correct = 0;

    for (int batch = 0; batch < num_batches; batch++)
    {
        int start_idx = batch * BATCH_SIZE;
        forward_pass(test_data + start_idx * INPUT_SIZE, hidden1, hidden2,
                     output, network, BATCH_SIZE, start_idx);
        correct += validatePredictions(output, test_labels, start_idx, BATCH_SIZE, OUTPUT_SIZE);
    }

    printf("Test Accuracy: %.2f%%\n", 100.0f * correct / TEST_DATA_SIZE);

    free(hidden1);
    free(hidden2);
    free(output);
}

void Train::forward_pass(float *input, float *hidden1, float *hidden2, float *output,
                         NeuralNetwork *network, int batch_size, int batch_offset)
{
    // Input -> Hidden1
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < HIDDEN1_SIZE; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                sum += input[b * INPUT_SIZE + j] * network->weightsInputHidden1[j * HIDDEN1_SIZE + i];
            }
            sum += network->biasHidden1[i];
            hidden1[b * HIDDEN1_SIZE + i] = fmaxf(0.0f, sum); // ReLU
        }
    }

    // Hidden1 -> Hidden2
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < HIDDEN2_SIZE; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < HIDDEN1_SIZE; j++)
            {
                sum += hidden1[b * HIDDEN1_SIZE + j] * network->weightsHidden1Hidden2[j * HIDDEN2_SIZE + i];
            }
            sum += network->biasHidden2[i];
            hidden2[b * HIDDEN2_SIZE + i] = fmaxf(0.0f, sum); // ReLU
        }
    }

    // Hidden2 -> Output
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < HIDDEN2_SIZE; j++)
            {
                sum += hidden2[b * HIDDEN2_SIZE + j] * network->weightsHidden2Output[j * OUTPUT_SIZE + i];
            }
            sum += network->biasOutput[i];
            output[b * OUTPUT_SIZE + i] = sum;
        }
    }

    // Apply softmax
    for (int b = 0; b < batch_size; b++)
    {
        float max_val = output[b * OUTPUT_SIZE];
        for (int i = 1; i < OUTPUT_SIZE; i++)
        {
            max_val = fmaxf(max_val, output[b * OUTPUT_SIZE + i]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            output[b * OUTPUT_SIZE + i] = expf(output[b * OUTPUT_SIZE + i] - max_val);
            sum_exp += output[b * OUTPUT_SIZE + i];
        }

        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            output[b * OUTPUT_SIZE + i] = fmaxf(output[b * OUTPUT_SIZE + i] / sum_exp, 1e-7f);
        }
    }
}

void Train::compute_layer_activations(float *input, float *weights, float *bias, float *output,
                                      int input_size, int output_size, int batch_size, bool use_relu)
{
    for (int b = 0; b < batch_size; b++)
    {
        for (int j = 0; j < output_size; j++)
        {
            float sum = compute_neuron_activation(input + b * input_size,
                                                  weights + j * input_size,
                                                  bias[j], input_size);
            output[b * output_size + j] = use_relu ? fmaxf(0.0f, sum) : sum;
        }
    }
}

float Train::compute_neuron_activation(float *input, float *weights, float bias, int input_size)
{
    float sum = 0.0f;
    for (int i = 0; i < input_size; i++)
    {
        sum += input[i] * weights[i];
    }
    return sum + bias;
}

void Train::apply_softmax(float *layer_output, int batch_size, int num_classes)
{
    for (int b = 0; b < batch_size; b++)
    {
        float *current_output = layer_output + b * num_classes;

        // Find max for numerical stability
        float max_val = current_output[0];
        for (int i = 1; i < num_classes; i++)
        {
            max_val = fmaxf(max_val, current_output[i]);
        }

        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++)
        {
            current_output[i] = expf(current_output[i] - max_val);
            sum_exp += current_output[i];
        }

        // Normalize
        for (int i = 0; i < num_classes; i++)
        {
            current_output[i] = fmaxf(current_output[i] / sum_exp, 1e-7f);
        }
    }
}

void Train::backward_pass(float *input, float *hidden1, float *hidden2, float *output,
                          float *del_output, float *d_ReLU_out1, float *d_ReLU_out2,
                          NeuralNetwork *network, int batch_size, int batch_offset)
{
    computeGradients(hidden2, del_output, network->gradWeightsHidden2Output,
                     network->gradBiasOutput, batch_size, HIDDEN2_SIZE, OUTPUT_SIZE);

    computeDeltaReLU(del_output, network->weightsHidden2Output, hidden2,
                     d_ReLU_out2, batch_size, OUTPUT_SIZE, HIDDEN2_SIZE);

    computeGradients(hidden1, d_ReLU_out2, network->gradWeightsHidden1Hidden2,
                     network->gradBiasHidden2, batch_size, HIDDEN1_SIZE, HIDDEN2_SIZE);

    computeDeltaReLU(d_ReLU_out2, network->weightsHidden1Hidden2, hidden1,
                     d_ReLU_out1, batch_size, HIDDEN2_SIZE, HIDDEN1_SIZE);

    computeGradients(input, d_ReLU_out1, network->gradWeightsInputHidden1,
                     network->gradBiasHidden1, batch_size, INPUT_SIZE, HIDDEN1_SIZE);
}

void Train::update_weights(NeuralNetwork *network)
{
    updateWeights(network->weightsInputHidden1, network->gradWeightsInputHidden1,
                  network->biasHidden1, network->gradBiasHidden1,
                  HIDDEN1_SIZE, INPUT_SIZE, LEARNING_RATE);
    updateWeights(network->weightsHidden1Hidden2, network->gradWeightsHidden1Hidden2,
                  network->biasHidden2, network->gradBiasHidden2,
                  HIDDEN2_SIZE, HIDDEN1_SIZE, LEARNING_RATE);
    updateWeights(network->weightsHidden2Output, network->gradWeightsHidden2Output,
                  network->biasOutput, network->gradBiasOutput,
                  OUTPUT_SIZE, HIDDEN2_SIZE, LEARNING_RATE);
}

float Train::compute_loss(const float *output, const int *labels, int batch_size)
{
    float loss = 0.0f;
    for (int i = 0; i < batch_size; i++)
    {
        int label = labels[i];
        loss -= logf(output[i * OUTPUT_SIZE + label] + 1e-7f);
    }
    return loss / batch_size;
}

int Train::validatePredictions(const float *output, const int *labels, int startIdx, int batchSize, int outputSize)
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

void Train::print_training_progress(int epoch, float loss, float accuracy, float time_ms)
{
    printf("Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%, Time: %.2f ms\n",
           epoch + 1, EPOCHS, loss, accuracy, time_ms);
}

void Train::computeGradients(float *input, float *delta, float *grad_weights,
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

void Train::computeDeltaReLU(float *relu_del_out, float *weights, float *input_layer,
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

void Train::updateWeights(float *weights, const float *grad_weights,
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

void Train::compute_output_error(float *output_layer, int *true_labels, float *error_out, int batch_size)
{
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            error_out[b * OUTPUT_SIZE + i] = output_layer[b * OUTPUT_SIZE + i] -
                                             (i == true_labels[b] ? 1.0f : 0.0f);
        }
    }
}
