#include "neural_network.cuh"
#include "training.cuh"
#include "data_loader.cuh"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    // Initialize random seed
    srand(time(NULL));

    // Initialize neural network
    NeuralNetwork network;

    // Allocate memory for training data
    float *training_data = (float *)malloc(TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *training_labels = (int *)malloc(TRAIN_DATA_SIZE * sizeof(int));
    float *test_data = (float *)malloc(TEST_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *test_labels = (int *)malloc(TEST_DATA_SIZE * sizeof(int));

    if (!training_data || !training_labels || !test_data || !test_labels)
    {
        printf("Error: Failed to allocate memory for dataset\n");
        return -1;
    }

    float *X_train = (float *)malloc(TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int *)malloc(TRAIN_DATA_SIZE * sizeof(int));
    float *X_test = (float *)malloc(TEST_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int *)malloc(TEST_DATA_SIZE * sizeof(int));
    // Load training and test data
    DataLoader::load_data("x_train.bin", X_train, TRAIN_DATA_SIZE * INPUT_SIZE);
    DataLoader::load_labels("y_train.bin", y_train, TRAIN_DATA_SIZE);
    DataLoader::load_data("x_test.bin", X_test, TEST_DATA_SIZE * INPUT_SIZE);
    DataLoader::load_labels("y_test.bin", y_test, TEST_DATA_SIZE);

    // Train the network
    Train::train(&network, training_data, training_labels);

    // Evaluate on test set
    Train::test(&network, test_data, test_labels);

    delete[] training_data;
    delete[] training_labels;
    delete[] test_data;
    delete[] test_labels;

    return 0;
}
