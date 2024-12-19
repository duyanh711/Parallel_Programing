#include "training.cuh"
#include "kernels.cuh"
#include "cuda_utils.cuh"
#include <cmath>

void Train::train(NeuralNetwork *nn, float *X_train, int *y_train)
{
    // Allocate device memory
    float *d_X_train, *d_hidden1, *d_hidden2, *d_output;
    float *d_del_output, *d_d_ReLU_out2, *d_d_ReLU_out1;
    int *d_y_train;

    // Allocate and copy data
    CHECK_CUDA_CALL(cudaMalloc(&d_X_train, TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_hidden1, BATCH_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_hidden2, BATCH_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_del_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_d_ReLU_out2, BATCH_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_d_ReLU_out1, BATCH_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_y_train, TRAIN_DATA_SIZE * sizeof(int)));

    CHECK_CUDA_CALL(cudaMemcpy(d_X_train, X_train, TRAIN_DATA_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_y_train, y_train, TRAIN_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    int num_batches = TRAIN_DATA_SIZE / BATCH_SIZE;

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        GpuTimer timer;
        timer.Start();

        float total_loss = 0.0f;
        int correct = 0;

        for (int batch = 0; batch < num_batches; batch++)
        {
            int start_idx = batch * BATCH_SIZE;

            // Forward pass
            forward_pass(d_X_train, d_hidden1, d_hidden2, d_output, nn, BATCH_SIZE, start_idx);

            // Get results and compute metrics
            float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            CHECK_CUDA_CALL(cudaMemcpy(output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            total_loss += compute_loss(output, &y_train[start_idx], BATCH_SIZE);
            correct += validate_predictions(output, y_train, start_idx, BATCH_SIZE, OUTPUT_SIZE);

            // Prepare for backprop
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                for (int i = 0; i < OUTPUT_SIZE; i++)
                {
                    output[b * OUTPUT_SIZE + i] -= (i == y_train[start_idx + b]) ? 1.0f : 0.0f;
                }
            }
            CHECK_CUDA_CALL(cudaMemcpy(d_del_output, output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            // Backward pass
            backward_pass(d_X_train, d_hidden1, d_hidden2, d_output, d_del_output,
                          d_d_ReLU_out1, d_d_ReLU_out2, nn, BATCH_SIZE, start_idx);

            // Update weights
            update_weights(nn);

            free(output);
        }

        timer.Stop();
        float elapsed = timer.Elapsed();

        printf("Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%, Time: %.2f ms\n",
               epoch + 1, EPOCHS,
               total_loss / num_batches,
               100.0f * correct / TRAIN_DATA_SIZE,
               elapsed);
    }

    // Free device memory
    cudaFree(d_X_train);
    cudaFree(d_hidden1);
    cudaFree(d_hidden2);
    cudaFree(d_output);
    cudaFree(d_del_output);
    cudaFree(d_d_ReLU_out2);
    cudaFree(d_d_ReLU_out1);
    cudaFree(d_y_train);
}

void Train::test(NeuralNetwork *nn, float *X_test, int *y_test)
{
    // Allocate device memory
    float *d_X_test, *d_hidden1, *d_hidden2, *d_output;
    CHECK_CUDA_CALL(cudaMalloc(&d_X_test, TEST_DATA_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_hidden1, BATCH_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_hidden2, BATCH_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA_CALL(cudaMemcpy(d_X_test, X_test, TEST_DATA_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int num_batches = TEST_DATA_SIZE / BATCH_SIZE;
    int correct = 0;

    for (int batch = 0; batch < num_batches; batch++)
    {
        int start_idx = batch * BATCH_SIZE;

        // Forward pass
        forward_pass(d_X_test, d_hidden1, d_hidden2, d_output, nn, BATCH_SIZE, start_idx);

        // Get results and compute accuracy
        float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
        CHECK_CUDA_CALL(cudaMemcpy(output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        correct += validate_predictions(output, y_test, start_idx, BATCH_SIZE, OUTPUT_SIZE);
        free(output);
    }

    printf("Test Accuracy: %.2f%%\n", 100.0f * correct / TEST_DATA_SIZE);

    // Free device memory
    cudaFree(d_X_test);
    cudaFree(d_hidden1);
    cudaFree(d_hidden2);
    cudaFree(d_output);
}

void Train::forward_pass(float *d_input, float *d_hidden1, float *d_hidden2,
                         float *d_output, NeuralNetwork *nn, int batch_size, int start_idx)
{
    // Layer 1
    forward_layer_kernel<<<(batch_size * HIDDEN1_SIZE + 255) / 256, 256>>>(
        d_input + start_idx * INPUT_SIZE,
        nn->weightsInputHidden1,
        nn->biasHidden1,
        d_hidden1,
        INPUT_SIZE, HIDDEN1_SIZE,
        batch_size, true);

    // Layer 2
    forward_layer_kernel<<<(batch_size * HIDDEN2_SIZE + 255) / 256, 256>>>(
        d_hidden1,
        nn->weightsHidden1Hidden2,
        nn->biasHidden2,
        d_hidden2,
        HIDDEN1_SIZE, HIDDEN2_SIZE,
        batch_size, true);

    // Output layer
    forward_layer_kernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(
        d_hidden2,
        nn->weightsHidden2Output,
        nn->biasOutput,
        d_output,
        HIDDEN2_SIZE, OUTPUT_SIZE,
        batch_size, false);

    // Softmax
    softmax_kernel<<<batch_size, 1>>>(d_output, batch_size, OUTPUT_SIZE);
}

void Train::backward_pass(float *d_input, float *d_hidden1, float *d_hidden2,
                          float *d_output, float *d_del_output,
                          float *d_d_ReLU_out1, float *d_d_ReLU_out2,
                          NeuralNetwork *nn, int batch_size, int start_idx)
{
    // Compute gradients for output layer
    compute_gradients_kernel<<<(HIDDEN2_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(
        d_hidden2, d_del_output,
        nn->gradWeightsHidden2Output,
        nn->gradBiasOutput,
        batch_size, HIDDEN2_SIZE, OUTPUT_SIZE);

    // Compute gradients for hidden layer 2
    compute_delta_relu_kernel<<<(batch_size * HIDDEN2_SIZE + 255) / 256, 256>>>(
        d_del_output, nn->weightsHidden2Output,
        d_hidden2, d_d_ReLU_out2,
        batch_size, OUTPUT_SIZE, HIDDEN2_SIZE);
    compute_gradients_kernel<<<(HIDDEN1_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(
        d_hidden1, d_d_ReLU_out2,
        nn->gradWeightsHidden1Hidden2,
        nn->gradBiasHidden2,
        batch_size, HIDDEN1_SIZE, HIDDEN2_SIZE);

    // Compute gradients for hidden layer 1
    compute_delta_relu_kernel<<<(batch_size * HIDDEN1_SIZE + 255) / 256, 256>>>(
        d_d_ReLU_out2, nn->weightsHidden1Hidden2,
        d_hidden1, d_d_ReLU_out1,
        batch_size, HIDDEN2_SIZE, HIDDEN1_SIZE);
    compute_gradients_kernel<<<(INPUT_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(
        d_input + start_idx * INPUT_SIZE, d_d_ReLU_out1,
        nn->gradWeightsInputHidden1,
        nn->gradBiasHidden1,
        batch_size, INPUT_SIZE, HIDDEN1_SIZE);
}

void Train::update_weights(NeuralNetwork *nn)
{
    update_weights_kernel<<<(INPUT_SIZE * HIDDEN1_SIZE + 255) / 256, 256>>>(
        nn->weightsInputHidden1, nn->gradWeightsInputHidden1,
        nn->biasHidden1, nn->gradBiasHidden1,
        HIDDEN1_SIZE, INPUT_SIZE);

    update_weights_kernel<<<(HIDDEN1_SIZE * HIDDEN2_SIZE + 255) / 256, 256>>>(
        nn->weightsHidden1Hidden2, nn->gradWeightsHidden1Hidden2,
        nn->biasHidden2, nn->gradBiasHidden2,
        HIDDEN2_SIZE, HIDDEN1_SIZE);

    update_weights_kernel<<<(HIDDEN2_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(
        nn->weightsHidden2Output, nn->gradWeightsHidden2Output,
        nn->biasOutput, nn->gradBiasOutput,
        OUTPUT_SIZE, HIDDEN2_SIZE);
}

int Train::validate_predictions(const float *output, const int *labels, int startIdx, int batchSize, int outputSize)
{
    int correct = 0;
    for (int b = 0; b < batchSize; b++)
    {
        const float *currentOutput = output + b * outputSize;
        float maxVal = currentOutput[0];
        int predicted = 0;

// Find maximum value and its index using single pass
#pragma unroll
        for (int j = 1; j < outputSize; j++)
        {
            if (currentOutput[j] > maxVal)
            {
                maxVal = currentOutput[j];
                predicted = j;
            }
        }

        correct += (predicted == labels[startIdx + b]);
    }
    return correct;
}

float Train::compute_loss(const float *output, const int *labels, int batch_size)
{
    float loss = 0.0f;
    for (int i = 0; i < batch_size; i++)
    {
        int label = labels[i];
        loss -= logf(output[i * OUTPUT_SIZE + label] + 1e-7f);
    }
    return loss / batch_size;
}
