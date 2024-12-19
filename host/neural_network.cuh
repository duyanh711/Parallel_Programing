#pragma once

// Constants for neural network structure and training
constexpr int INPUT_SIZE = 784;
constexpr int HIDDEN1_SIZE = 128;
constexpr int HIDDEN2_SIZE = 128;
constexpr int OUTPUT_SIZE = 10;
constexpr int TRAIN_DATA_SIZE = 10000;
constexpr int TEST_DATA_SIZE = 1000;
constexpr int BATCH_SIZE = 4;
constexpr int EPOCHS = 10;
constexpr float LEARNING_RATE = 0.01f;

// Neural Network class definition
class NeuralNetwork
{
public:
    // Weights
    float *weightsInputHidden1, *weightsHidden1Hidden2, *weightsHidden2Output;
    // Biases
    float *biasHidden1, *biasHidden2, *biasOutput;
    // Gradients
    float *gradWeightsInputHidden1, *gradWeightsHidden1Hidden2, *gradWeightsHidden2Output;
    float *gradBiasHidden1, *gradBiasHidden2, *gradBiasOutput;

    // Constructor and destructor
    NeuralNetwork();
    ~NeuralNetwork();

private:
    void initWeights(float *weights, int size);
    void initBias(float *bias, int size);
};