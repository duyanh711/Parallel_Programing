#pragma once

#include "neural_network.cuh"

class Train
{
public:
    // Main training functions
    static void train(NeuralNetwork *network, float *training_data, int *labels);
    static void test(NeuralNetwork *network, float *test_data, int *test_labels);

private:
    // Forward propagation
    static void forward_pass(float *input_data, float *hidden1, float *hidden2, float *output,
                             NeuralNetwork *network, int batch_size, int batch_offset);

    static void compute_layer_activations(float *input, float *weights, float *bias, float *output,
                                          int input_size, int output_size, int batch_size, bool use_relu);

    static float compute_neuron_activation(float *input, float *weights, float bias, int input_size);

    static void apply_softmax(float *layer_output, int batch_size, int num_classes);

    // Backpropagation
    static void backward_pass(float *input, float *hidden1, float *hidden2, float *output,
                              float *del_output, float *d_ReLU_out1, float *d_ReLU_out2,
                              NeuralNetwork *network, int batch_size, int batch_offset);

    static void computeGradients(float *input, float *delta, float *grad_weights,
                                 float *grad_bias, int batch_size, int input_size, int output_size);

    static void computeDeltaReLU(float *relu_del_out, float *weights, float *input_layer,
                                 float *relu_del, int batch_size, int output_size, int input_size);

    // Weight updates
    static void update_weights(NeuralNetwork *network);

    static void updateWeights(float *weights, const float *grad_weights,
                              float *bias, const float *grad_bias,
                              int output_size, int input_size, float learning_rate);

    // Metrics and utilities
    static float compute_loss(const float *network_output, const int *true_labels, int batch_size);

    static void compute_output_error(float *output_layer, int *true_labels, float *error_out, int batch_size);

    static int validatePredictions(const float *network_output, const int *true_labels,
                                   int batch_offset, int batch_size, int num_classes);

    static void print_training_progress(int epoch, float loss, float accuracy, float time_ms);
};

// Constants
#define TRAIN_DATA_SIZE 10000
#define TEST_DATA_SIZE 1000
#define BATCH_SIZE 4
#define EPOCHS 10
#define LEARNING_RATE 0.01f
