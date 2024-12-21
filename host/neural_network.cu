#include "neural_network.cuh"
#include <cmath>
#include <cstdlib>

void NeuralNetwork::initWeights(float *weights, int size)
{
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++)
    {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

void NeuralNetwork::initBias(float *bias, int size)
{
    for (int i = 0; i < size; i++)
    {
        bias[i] = 0.0f;
    }
}

NeuralNetwork::NeuralNetwork()
{
    // Allocate memory
    weightsInputHidden1 = (float *)malloc(HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    weightsHidden1Hidden2 = (float *)malloc(HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    weightsHidden2Output = (float *)malloc(OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));

    biasHidden1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    biasHidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    biasOutput = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    gradWeightsInputHidden1 = (float *)malloc(HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    gradWeightsHidden1Hidden2 = (float *)malloc(HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    gradWeightsHidden2Output = (float *)malloc(OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));

    gradBiasHidden1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    gradBiasHidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    gradBiasOutput = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    // Initialize weights and biases
    initWeights(weightsInputHidden1, HIDDEN1_SIZE * INPUT_SIZE);
    initWeights(weightsHidden1Hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE);
    initWeights(weightsHidden2Output, OUTPUT_SIZE * HIDDEN2_SIZE);

    initBias(biasHidden1, HIDDEN1_SIZE);
    initBias(biasHidden2, HIDDEN2_SIZE);
    initBias(biasOutput, OUTPUT_SIZE);
}

NeuralNetwork::~NeuralNetwork()
{
    // Free all allocated memory
    free(weightsInputHidden1);
    free(weightsHidden1Hidden2);
    free(weightsHidden2Output);
    free(biasHidden1);
    free(biasHidden2);
    free(biasOutput);
    free(gradWeightsInputHidden1);
    free(gradWeightsHidden1Hidden2);
    free(gradWeightsHidden2Output);
    free(gradBiasHidden1);
    free(gradBiasHidden2);
    free(gradBiasOutput);
}
