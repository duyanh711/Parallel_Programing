#include "neural_network.cuh"
#include <cmath>
#include <vector>

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
    this->weightsInputHidden1 = (float *)malloc(HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    this->weightsHidden1Hidden2 = (float *)malloc(HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    this->weightsHidden2Output = (float *)malloc(OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));

    this->biasHidden1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    this->biasHidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    this->biasOutput = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    this->gradWeightsInputHidden1 = (float *)malloc(HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    this->gradWeightsHidden1Hidden2 = (float *)malloc(HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    this->gradWeightsHidden2Output = (float *)malloc(OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));

    this->gradBiasHidden1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    this->gradBiasHidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    this->gradBiasOutput = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    // Initialize weights and biases
    initWeights(this->weightsInputHidden1, HIDDEN1_SIZE * INPUT_SIZE);
    initWeights(this->weightsHidden1Hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE);
    initWeights(this->weightsHidden2Output, OUTPUT_SIZE * HIDDEN2_SIZE);

    initBias(this->biasHidden1, HIDDEN1_SIZE);
    initBias(this->biasHidden2, HIDDEN2_SIZE);
    initBias(this->biasOutput, OUTPUT_SIZE);
}

NeuralNetwork::~NeuralNetwork()
{
    // Free weights
    delete[] weightsInputHidden1;
    delete[] weightsHidden1Hidden2;
    delete[] weightsHidden2Output;

    // Free biases
    delete[] biasHidden1;
    delete[] biasHidden2;
    delete[] biasOutput;

    // Free gradients
    delete[] gradWeightsInputHidden1;
    delete[] gradWeightsHidden1Hidden2;
    delete[] gradWeightsHidden2Output;
    delete[] gradBiasHidden1;
    delete[] gradBiasHidden2;
    delete[] gradBiasOutput;
}