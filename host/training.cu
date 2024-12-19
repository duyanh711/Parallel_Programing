#include "training.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void Train::train(NeuralNetwork *network, float *training_data, int *labels)
{
    // Allocate memory for intermediate layers
    float *hidden1 = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    float *output_error = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    float *hidden1_error = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *hidden2_error = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));

    int num_batches = TRAIN_DATA_SIZE / BATCH_SIZE;

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        float total_loss = 0.0f;
        int total_correct = 0;

        // Process each batch
        for (int batch = 0; batch < num_batches; batch++)
        {
            int start_idx = batch * BATCH_SIZE;
            float *batch_data = training_data + start_idx * INPUT_SIZE;
            int *batch_labels = labels + start_idx;

            // Forward propagation
            forward_pass(batch_data, hidden1, hidden2, output,
                         network, BATCH_SIZE, start_idx);

            // Calculate loss and accuracy
            float batch_loss = compute_loss(output, batch_labels, BATCH_SIZE);
            total_loss += batch_loss;
            total_correct += validatePredictions(output, labels, start_idx, BATCH_SIZE, OUTPUT_SIZE);

            // Calculate output layer error
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                for (int i = 0; i < OUTPUT_SIZE; i++)
                {
                    output_error[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i] -
                                                        (i == batch_labels[b] ? 1.0f : 0.0f);
                }
            }

            // Backpropagation
            backward_pass(batch_data, hidden1, hidden2, output,
                          output_error, hidden1_error, hidden2_error,
                          network, BATCH_SIZE, start_idx);

            // Update weights
            update_weights(network);
        }

        float avg_loss = total_loss / num_batches;
        float accuracy = (float)total_correct / TRAIN_DATA_SIZE * 100.0f;
        printf("Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%\n",
               epoch + 1, EPOCHS, avg_loss, accuracy);
    }

    // Cleanup
    free(hidden1);
    free(hidden2);
    free(output);
    free(output_error);
    free(hidden1_error);
    free(hidden2_error);
}

void Train::test(NeuralNetwork *network, float *test_data, int *test_labels)
{
    float *hidden1 = (float *)malloc(BATCH_SIZE * HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(BATCH_SIZE * HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TEST_DATA_SIZE / BATCH_SIZE;
    int total_correct = 0;
    float total_loss = 0.0f;

    for (int batch = 0; batch < num_batches; batch++)
    {
        int start_idx = batch * BATCH_SIZE;
        float *batch_data = test_data + start_idx * INPUT_SIZE;
        int *batch_labels = test_labels + start_idx;

        forward_pass(batch_data, hidden1, hidden2, output,
                     network, BATCH_SIZE, start_idx);

        float batch_loss = compute_loss(output, batch_labels, BATCH_SIZE);
        total_loss += batch_loss;
        total_correct += validatePredictions(output, test_labels, start_idx, BATCH_SIZE, OUTPUT_SIZE);
    }

    float avg_loss = total_loss / num_batches;
    float accuracy = (float)total_correct / TEST_DATA_SIZE * 100.0f;
    printf("Test - Loss: %.4f, Accuracy: %.2f%%\n", avg_loss, accuracy);

    free(hidden1);
    free(hidden2);
    free(output);
}

void Train::forward_pass(float *input_data, float *hidden1, float *hidden2, float *output,
                         NeuralNetwork *network, int batch_size, int batch_offset)
{
    // Input -> Hidden1 (với ReLU)
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < HIDDEN1_SIZE; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < INPUT_SIZE; j++)
            {
                sum += input_data[b * INPUT_SIZE + j] * network->weightsInputHidden1[j * HIDDEN1_SIZE + i];
            }
            sum += network->biasHidden1[i];
            hidden1[b * HIDDEN1_SIZE + i] = fmaxf(0.0f, sum); // ReLU
        }
    }

    // Hidden1 -> Hidden2 (với ReLU)
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

    // Hidden2 -> Output (không có ReLU)
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

    // Apply softmax to output
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
            output[b * OUTPUT_SIZE + i] /= sum_exp;
        }
    }
}

void Train::backward_pass(float *input, float *hidden1, float *hidden2, float *output,
                          float *output_error, float *hidden1_error, float *hidden2_error,
                          NeuralNetwork *network, int batch_size, int batch_offset)
{
    // Reset gradients
    memset(network->gradWeightsHidden2Output, 0, HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float));
    memset(network->gradWeightsHidden1Hidden2, 0, HIDDEN1_SIZE * HIDDEN2_SIZE * sizeof(float));
    memset(network->gradWeightsInputHidden1, 0, INPUT_SIZE * HIDDEN1_SIZE * sizeof(float));
    memset(network->gradBiasOutput, 0, OUTPUT_SIZE * sizeof(float));
    memset(network->gradBiasHidden2, 0, HIDDEN2_SIZE * sizeof(float));
    memset(network->gradBiasHidden1, 0, HIDDEN1_SIZE * sizeof(float));

    // Compute output layer gradients
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < HIDDEN2_SIZE; i++)
        {
            for (int j = 0; j < OUTPUT_SIZE; j++)
            {
                network->gradWeightsHidden2Output[i * OUTPUT_SIZE + j] +=
                    hidden2[b * HIDDEN2_SIZE + i] * output_error[b * OUTPUT_SIZE + j];
                if (i == 0)
                {
                    network->gradBiasOutput[j] += output_error[b * OUTPUT_SIZE + j];
                }
            }
        }
    }

    // Compute hidden2 error
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < HIDDEN2_SIZE; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < OUTPUT_SIZE; j++)
            {
                sum += output_error[b * OUTPUT_SIZE + j] * network->weightsHidden2Output[i * OUTPUT_SIZE + j];
            }
            hidden2_error[b * HIDDEN2_SIZE + i] = sum * (hidden2[b * HIDDEN2_SIZE + i] > 0.0f);
        }
    }

    // Compute hidden2 layer gradients
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < HIDDEN1_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN2_SIZE; j++)
            {
                network->gradWeightsHidden1Hidden2[i * HIDDEN2_SIZE + j] +=
                    hidden1[b * HIDDEN1_SIZE + i] * hidden2_error[b * HIDDEN2_SIZE + j];
                if (i == 0)
                {
                    network->gradBiasHidden2[j] += hidden2_error[b * HIDDEN2_SIZE + j];
                }
            }
        }
    }

    // Compute hidden1 error
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < HIDDEN1_SIZE; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < HIDDEN2_SIZE; j++)
            {
                sum += hidden2_error[b * HIDDEN2_SIZE + j] * network->weightsHidden1Hidden2[i * HIDDEN2_SIZE + j];
            }
            hidden1_error[b * HIDDEN1_SIZE + i] = sum * (hidden1[b * HIDDEN1_SIZE + i] > 0.0f);
        }
    }

    // Compute hidden1 layer gradients
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            for (int j = 0; j < HIDDEN1_SIZE; j++)
            {
                network->gradWeightsInputHidden1[i * HIDDEN1_SIZE + j] +=
                    input[b * INPUT_SIZE + i] * hidden1_error[b * HIDDEN1_SIZE + j];
                if (i == 0)
                {
                    network->gradBiasHidden1[j] += hidden1_error[b * HIDDEN1_SIZE + j];
                }
            }
        }
    }
}

void Train::update_weights(NeuralNetwork *network)
{
    // Update weights
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN1_SIZE; j++)
        {
            network->weightsInputHidden1[i * HIDDEN1_SIZE + j] -=
                LEARNING_RATE * network->gradWeightsInputHidden1[i * HIDDEN1_SIZE + j] / BATCH_SIZE;
        }
    }

    for (int i = 0; i < HIDDEN1_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN2_SIZE; j++)
        {
            network->weightsHidden1Hidden2[i * HIDDEN2_SIZE + j] -=
                LEARNING_RATE * network->gradWeightsHidden1Hidden2[i * HIDDEN2_SIZE + j] / BATCH_SIZE;
        }
    }

    for (int i = 0; i < HIDDEN2_SIZE; i++)
    {
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            network->weightsHidden2Output[i * OUTPUT_SIZE + j] -=
                LEARNING_RATE * network->gradWeightsHidden2Output[i * OUTPUT_SIZE + j] / BATCH_SIZE;
        }
    }

    // Update biases
    for (int i = 0; i < HIDDEN1_SIZE; i++)
    {
        network->biasHidden1[i] -= LEARNING_RATE * network->gradBiasHidden1[i] / BATCH_SIZE;
    }

    for (int i = 0; i < HIDDEN2_SIZE; i++)
    {
        network->biasHidden2[i] -= LEARNING_RATE * network->gradBiasHidden2[i] / BATCH_SIZE;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        network->biasOutput[i] -= LEARNING_RATE * network->gradBiasOutput[i] / BATCH_SIZE;
    }
}

float Train::compute_loss(const float *network_output, const int *true_labels, int batch_size)
{
    float batch_loss = 0.0f;
    for (int sample = 0; sample < batch_size; sample++)
    {
        int true_class = true_labels[sample];
        float predicted_probability = network_output[sample * OUTPUT_SIZE + true_class];
        batch_loss -= logf(predicted_probability + 1e-7f); // Add small epsilon for numerical stability
    }
    return batch_loss / batch_size;
}

int Train::validatePredictions(const float *network_output, const int *true_labels,
                               int batch_offset, int batch_size, int num_classes)
{
    int correct_count = 0;

    for (int sample = 0; sample < batch_size; sample++)
    {
        const float *sample_output = network_output + sample * num_classes;

        // Find the predicted class (maximum output)
        int predicted_class = 0;
        float max_probability = sample_output[0];

        for (int class_idx = 1; class_idx < num_classes; class_idx++)
        {
            if (sample_output[class_idx] > max_probability)
            {
                max_probability = sample_output[class_idx];
                predicted_class = class_idx;
            }
        }

        // Kiểm tra lại index của true_labels
        if (predicted_class == true_labels[batch_offset + sample])
        {
            correct_count++;
        }
    }

    return correct_count;
}

void Train::print_training_progress(int epoch, float loss, float accuracy)
{
    printf("Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%\n",
           epoch + 1, EPOCHS, loss, accuracy);
}
