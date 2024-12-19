#pragma once

#include "neural_network.cuh"

void train(NeuralNetwork *nn, float *X_train, int *y_train);
void test(NeuralNetwork *nn, float *X_test, int *y_test);
void forward_pass(float *input, float *hidden1, float *hidden2, float *output,
                  NeuralNetwork *nn, int batch_size, int start_idx);
void backward_pass(float *input, float *hidden1, float *hidden2, float *output,
                   float *del_output, float *d_ReLU_out1, float *d_ReLU_out2,
                   NeuralNetwork *nn, int batch_size, int start_idx);

void updateWeights(float *weights, const float *grad_weights,
                   float *bias, const float *grad_bias,
                   int output_size, int input_size, float learning_rate);
float compute_loss(const float *output, const int *labels, int batch_size);
int validatePredictions(const float *output, const int *labels, int startIdx, int batchSize, int outputSize);
void forwardLayer(float *input, float *weights, float *bias, float *output,
                  int input_size, int output_size, int batch_size, bool use_relu);
void softmax_cpu(float *x, int batch_size, int size);
void computeGradients(float *input, float *delta, float *grad_weights,
                      float *grad_bias, int batch_size, int input_size, int output_size);

void computeDeltaReLU(float *relu_del_out, float *weights, float *input_layer,
                      float *relu_del, int batch_size, int output_size, int input_size);
