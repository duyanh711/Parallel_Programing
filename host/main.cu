#include "neural_network.cuh"
#include "training.cuh"
#include "data_loader.cuh"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    srand(time(NULL));

    // Create neural network object (constructor will initialize everything)
    NeuralNetwork network;

    float *X_train = (float *)malloc(TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int *)malloc(TRAIN_DATA_SIZE * sizeof(int));
    float *X_test = (float *)malloc(TEST_DATA_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int *)malloc(TEST_DATA_SIZE * sizeof(int));

    load_data("x_train.bin", X_train, TRAIN_DATA_SIZE * INPUT_SIZE);
    load_labels("y_train.bin", y_train, TRAIN_DATA_SIZE);
    load_data("x_test.bin", X_test, TEST_DATA_SIZE * INPUT_SIZE);
    load_labels("y_test.bin", y_test, TEST_DATA_SIZE);

    Train trainer;
    trainer.train(&network, X_train, y_train);
    trainer.test(&network, X_test, y_test);

    // No need to manually free network memory - destructor will handle it
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}
