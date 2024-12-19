#pragma once

#include "neural_network.cuh"

class Train
{
public:
    // Main training functions
    static void train(NeuralNetwork *network, float *training_data, int *labels);
    static void test(NeuralNetwork *network, float *test_data, int *test_labels);

    // Forward propagation
    static void forward_pass(float *input_data, float *hidden1, float *hidden2, float *output,
                             NeuralNetwork *network, int batch_size, int batch_offset);

    // Backpropagation
    static void backward_pass(float *input, float *hidden1, float *hidden2, float *output,
                              float *output_error, float *hidden1_error, float *hidden2_error,
                              NeuralNetwork *network, int batch_size, int batch_offset);

    // Weight updates
    static void update_weights(NeuralNetwork *network);

    // Metrics and utilities
    static float compute_loss(const float *network_output, const int *true_labels, int batch_size);

    static int validatePredictions(const float *network_output, const int *true_labels,
                                   int batch_offset, int batch_size, int num_classes);

    static void print_training_progress(int epoch, float loss, float accuracy);
};