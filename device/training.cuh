#pragma once

#include "neural_network.cuh"

class Train {
public:
    static void train(NeuralNetwork *nn, float *X_train, int *y_train);
    static void test(NeuralNetwork *nn, float *X_test, int *y_test);

private:
    static void forward_pass(float *d_input, float *d_hidden1, float *d_hidden2,
                           float *d_output, NeuralNetwork *nn, int batch_size, int start_idx);
    
    static void backward_pass(float *d_input, float *d_hidden1, float *d_hidden2,
                            float *d_output, float *d_del_output,
                            float *d_d_ReLU_out1, float *d_d_ReLU_out2,
                            NeuralNetwork *nn, int batch_size, int start_idx);
    
    static void update_weights(NeuralNetwork *nn);
    static int validatePredictions(const float *output, const int *labels,
                                 int startIdx, int batchSize, int outputSize);
    static float compute_loss(const float *output, const int *labels, int batch_size);
}; 