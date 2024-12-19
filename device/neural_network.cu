#include "neural_network.cuh"
#include "cuda_utils.cuh"
#include <cmath>
#include <vector>

void NeuralNetwork::initWeights(float *weights, int size) {
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

void NeuralNetwork::initBias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

NeuralNetwork::NeuralNetwork() {
    // Allocate memory on the device
    CHECK_CUDA_CALL(cudaMalloc(&weightsInputHidden1, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&weightsHidden1Hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&weightsHidden2Output, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&biasHidden1, HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&biasHidden2, HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&biasOutput, OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA_CALL(cudaMalloc(&gradWeightsInputHidden1, HIDDEN1_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&gradWeightsHidden1Hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&gradWeightsHidden2Output, OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&gradBiasHidden1, HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&gradBiasHidden2, HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&gradBiasOutput, OUTPUT_SIZE * sizeof(float)));

    // Host memory for initialization
    std::vector<float> hostWeightsInputHidden1(HIDDEN1_SIZE * INPUT_SIZE);
    std::vector<float> hostWeightsHidden1Hidden2(HIDDEN2_SIZE * HIDDEN1_SIZE);
    std::vector<float> hostWeightsHidden2Output(OUTPUT_SIZE * HIDDEN2_SIZE);
    std::vector<float> hostBiasHidden1(HIDDEN1_SIZE);
    std::vector<float> hostBiasHidden2(HIDDEN2_SIZE);
    std::vector<float> hostBiasOutput(OUTPUT_SIZE);

    initWeights(hostWeightsInputHidden1.data(), HIDDEN1_SIZE * INPUT_SIZE);
    initWeights(hostWeightsHidden1Hidden2.data(), HIDDEN2_SIZE * HIDDEN1_SIZE);
    initWeights(hostWeightsHidden2Output.data(), OUTPUT_SIZE * HIDDEN2_SIZE);
    initBias(hostBiasHidden1.data(), HIDDEN1_SIZE);
    initBias(hostBiasHidden2.data(), HIDDEN2_SIZE);
    initBias(hostBiasOutput.data(), OUTPUT_SIZE);

    CHECK_CUDA_CALL(cudaMemcpy(weightsInputHidden1, hostWeightsInputHidden1.data(), 
                              HIDDEN1_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(weightsHidden1Hidden2, hostWeightsHidden1Hidden2.data(), 
                              HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(weightsHidden2Output, hostWeightsHidden2Output.data(), 
                              OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(biasHidden1, hostBiasHidden1.data(), 
                              HIDDEN1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(biasHidden2, hostBiasHidden2.data(), 
                              HIDDEN2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(biasOutput, hostBiasOutput.data(), 
                              OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
}

NeuralNetwork::~NeuralNetwork() {
    cudaFree(weightsInputHidden1);
    cudaFree(weightsHidden1Hidden2);
    cudaFree(weightsHidden2Output);
    cudaFree(biasHidden1);
    cudaFree(biasHidden2);
    cudaFree(biasOutput);
    cudaFree(gradWeightsInputHidden1);
    cudaFree(gradWeightsHidden1Hidden2);
    cudaFree(gradWeightsHidden2Output);
    cudaFree(gradBiasHidden1);
    cudaFree(gradBiasHidden2);
    cudaFree(gradBiasOutput);
} 