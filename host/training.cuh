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
                              float *output_error, float *hidden1_error, float *hidden2_error,
                              NeuralNetwork *network, int batch_size, int batch_offset);

    static void compute_output_error(float *output_layer, int *true_labels,
                                     float *error_out, int batch_size);

    static void compute_layer_gradients(float *layer_input, float *layer_error,
                                        float *weight_gradients, float *bias_gradients,
                                        int batch_size, int input_size, int output_size);

    static void compute_layer_error(float *next_layer_error, float *weights,
                                    float *current_layer_output, float *current_layer_error,
                                    int batch_size, int next_size, int current_size);

    // Weight updates
    static void update_weights(NeuralNetwork *network);

    static void update_layer_parameters(float *weights, float *weight_gradients,
                                        float *biases, float *bias_gradients,
                                        int input_size, int output_size);

    // Metrics and utilities
    static float compute_loss(const float *network_output, const int *true_labels, int batch_size);

    static int validatePredictions(const float *network_output, const int *true_labels,
                                   int batch_offset, int batch_size, int num_classes);

    static void print_training_progress(int epoch, float loss, float accuracy);
};