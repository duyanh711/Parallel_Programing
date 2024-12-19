#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <cstdio>

#include "neural_network.cuh"
#include "data_loader.cuh"
#include "training.cuh"

int main(int argc, char **argv) {
    srand(time(NULL));

    // Initialize neural network
    NeuralNetwork nn;

    // Allocate memory for training and test data
    float *X_train = (float *)malloc(TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int *)malloc(TRAIN_DATA_SIZE * sizeof(int));
    float *X_test = (float *)malloc(TEST_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int *)malloc(TEST_DATA_SIZE * sizeof(int));

    // Load data
    DataLoader::load_data("x_train.bin", X_train, TRAIN_DATA_SIZE * INPUT_SIZE);
    DataLoader::load_labels("y_train.bin", y_train, TRAIN_DATA_SIZE);
    DataLoader::load_data("x_test.bin", X_test, TEST_DATA_SIZE * INPUT_SIZE);
    DataLoader::load_labels("y_test.bin", y_test, TEST_DATA_SIZE);

    // Train and test
    Train::train(&nn, X_train, y_train);
    Train::test(&nn, X_test, y_test);

    // Free memory
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
} 