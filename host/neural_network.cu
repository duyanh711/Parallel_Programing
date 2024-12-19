#include "neural_network.cuh"
#include <cmath>
#include <vector>



void initWeights(float *weights, int size)
{
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++)
    {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

void initBias(float *bias, int size)
{
    for (int i = 0; i < size; i++)
    {
        bias[i] = 0.0f;
    }
}

void initializeNetwork(NeuralNetwork *nn)
{
    // Allocate memory
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

    // Initialize weights and biases
    initWeights(nn->weightsInputHidden1, HIDDEN1_SIZE * INPUT_SIZE);
    initWeights(nn->weightsHidden1Hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE);
    initWeights(nn->weightsHidden2Output, OUTPUT_SIZE * HIDDEN2_SIZE);

    initBias(nn->biasHidden1, HIDDEN1_SIZE);
    initBias(nn->biasHidden2, HIDDEN2_SIZE);
    initBias(nn->biasOutput, OUTPUT_SIZE);
}

