#pragma once

// Constants for neural network structure and training
#define INPUT_SIZE 784
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 128
#define OUTPUT_SIZE 10
#define TRAIN_DATA_SIZE 10000
#define TEST_DATA_SIZE 1000
#define BATCH_SIZE 4
#define EPOCHS 10
#define LEARNING_RATE 0.01f

typedef struct
{
    float *weightsInputHidden1, *weightsHidden1Hidden2, *weightsHidden2Output;
    float *biasHidden1, *biasHidden2, *biasOutput;
    float *gradWeightsInputHidden1, *gradWeightsHidden1Hidden2, *gradWeightsHidden2Output;
    float *gradBiasHidden1, *gradBiasHidden2, *gradBiasOutput;
} NeuralNetwork;

void initWeights(float *weights, int size);
void initBias(float *bias, int size);
void initializeNetwork(NeuralNetwork *nn);