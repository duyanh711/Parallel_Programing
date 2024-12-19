#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>

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

class NeuralNetwork
{
public:
    float *weightsInputHidden1, *weightsHidden1Hidden2, *weightsHidden2Output;
    float *biasHidden1, *biasHidden2, *biasOutput;
    float *gradWeightsInputHidden1, *gradWeightsHidden1Hidden2, *gradWeightsHidden2Output;
    float *gradBiasHidden1, *gradBiasHidden2, *gradBiasOutput;
};

// Helper functions
int validatePredictions(const float *output, const int *labels, int startIdx, int batchSize, int outputSize)
{
    int correct = 0;
    for (int i = 0; i < batchSize; i++)
    {
        int predicted = 0;
        for (int j = 1; j < outputSize; j++)
        {
            if (output[i * outputSize + j] > output[i * outputSize + predicted])
            {
                predicted = j;
            }
        }
        if (predicted == labels[startIdx + i])
        {
            correct++;
        }
    }
    return correct;
}

float compute_loss(const float *output, const int *labels, int batch_size)
{
    float loss = 0.0f;
    for (int i = 0; i < batch_size; i++)
    {
        int label = labels[i];
        loss -= logf(output[i * OUTPUT_SIZE + label] + 1e-7f);
    }
    return loss / batch_size;
}



// Initialize weights with Xavier initialization
void initWeights(float *weights, int size)
{
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++)
    {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

// Initialize biases to zero
void initBias(float *bias, int size)
{
    for (int i = 0; i < size; i++)
    {
        bias[i] = 0.0f;
    }
}

void initializeNetwork(NeuralNetwork *nn)
{
    nn->weightsInputHidden1 = new float[HIDDEN1_SIZE * INPUT_SIZE];
    nn->weightsHidden1Hidden2 = new float[HIDDEN2_SIZE * HIDDEN1_SIZE];
    nn->weightsHidden2Output = new float[OUTPUT_SIZE * HIDDEN2_SIZE];

    nn->biasHidden1 = new float[HIDDEN1_SIZE];
    nn->biasHidden2 = new float[HIDDEN2_SIZE];
    nn->biasOutput = new float[OUTPUT_SIZE];

    nn->gradWeightsInputHidden1 = new float[HIDDEN1_SIZE * INPUT_SIZE];
    nn->gradWeightsHidden1Hidden2 = new float[HIDDEN2_SIZE * HIDDEN1_SIZE];
    nn->gradWeightsHidden2Output = new float[OUTPUT_SIZE * HIDDEN2_SIZE];

    nn->gradBiasHidden1 = new float[HIDDEN1_SIZE];
    nn->gradBiasHidden2 = new float[HIDDEN2_SIZE];
    nn->gradBiasOutput = new float[OUTPUT_SIZE];

    initWeights(nn->weightsInputHidden1, HIDDEN1_SIZE * INPUT_SIZE);
    initWeights(nn->weightsHidden1Hidden2, HIDDEN2_SIZE * HIDDEN1_SIZE);
    initWeights(nn->weightsHidden2Output, OUTPUT_SIZE * HIDDEN2_SIZE);

    initBias(nn->biasHidden1, HIDDEN1_SIZE);
    initBias(nn->biasHidden2, HIDDEN2_SIZE);
    initBias(nn->biasOutput, OUTPUT_SIZE);
}

// void softmax_cpu(float* x, int batch_size, int size)
// {
//     for (int b = 0; b < batch_size; ++b)
//     {
//         // Tìm giá trị max cho tính ổn định số học
//         float max_val = x[b * size];
//         for (int i = 1; i < size; i++)
//         {
//             max_val = std::max(max_val, x[b * size + i]);
//         }

//         // Tính exp và tổng
//         float sum = 0.0f;
//         for (int i = 0; i < size; i++)
//         {
//             x[b * size + i] = std::exp(x[b * size + i] - max_val);
//             sum += x[b * size + i];
//         }

//         // Chuẩn hóa
//         for (int i = 0; i < size; i++)
//         {
//             x[b * size + i] = std::max(x[b * size + i] / sum, 1e-7f);
//         }
//     }
// }

void softmax_cpu(float* x, int batch_size, int size)
{
    for (int b = 0; b < batch_size; ++b)
    {
        float max_val = x[b * size];
        for (int i = 1; i < size; i++)
        {
            max_val = std::max(max_val, x[b * size + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            x[b * size + i] = std::exp(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        for (int i = 0; i < size; i++)
        {
            x[b * size + i] /= sum;
        }
    }
}
void computeGradients(float* input, float* delta, float* grad_weights, 
                      float* grad_bias, int batch_size, int input_size, int output_size) 
{
    // Khởi tạo gradient bias về 0 trước
    std::fill(grad_bias, grad_bias + output_size, 0.0f);

    // Lặp qua từng phần tử của grad_weights và grad_bias
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float grad_w = 0.0f;

            // Tính toán gradient cho weights
            for (int b = 0; b < batch_size; ++b) {
                grad_w += input[b * input_size + i] * delta[b * output_size + j];
            }

            // Cập nhật gradient weights
            grad_weights[i * output_size + j] = grad_w;

            // Tính toán gradient cho bias
            grad_bias[j] = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                grad_bias[j] += delta[b * output_size + j];
            }
        }
    }
}


void computeDeltaReLU(float* relu_del_out, float* weights, float* input_layer, 
                      float* relu_del, int batch_size, int output_size, int input_size) 
{
    // Lặp qua từng phần tử trong batch và input_size
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            // Tính giá trị relu_del cho phần tử (i, j)
            float grad_sum = 0.0f;

            // Lặp qua các output để tính tổng gradient
            for (int k = 0; k < output_size; ++k) {
                grad_sum += relu_del_out[i * output_size + k] * weights[j * output_size + k];
            }

            // Lưu giá trị gradient vào relu_del
            relu_del[i * input_size + j] = grad_sum * (input_layer[i * input_size + j] > 0.0f); // Derivative of ReLU (ReLU'(x) = 1 if x > 0, otherwise 0)
        }
    }
}


void forwardLayer(float* input, float* weights, float* bias, float* output, 
                  int input_size, int output_size, int batch_size, bool use_relu)
{
    // Lặp qua từng phần tử trong batch
    for (int b = 0; b < batch_size; ++b)
    {
        for (int j = 0; j < output_size; ++j)
        {
            // Tính giá trị đầu ra cho từng phần tử (b, j)
            float sum = 0.0f;
            
            // Lặp qua tất cả các input_size để tính tích vô hướng
            for (int k = 0; k < input_size; ++k)
            {
                sum += input[b * input_size + k] * weights[k * output_size + j];
            }

            // Cộng bias vào giá trị tính được
            sum += bias[j];

            // Áp dụng ReLU nếu cần
            if (use_relu)
            {
                sum = fmaxf(0.0f, sum); // ReLU activation
            }

            // Lưu giá trị vào output
            output[b * output_size + j] = sum;
        }
    }
}


void updateWeights(float* weights, const float* grad_weights, 
                   float* bias, const float* grad_bias, 
                   int output_size, int input_size, float learning_rate)
{
    // Cập nhật trọng số
    for (int i = 0; i < input_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            int idx = i * output_size + j;
            weights[idx] -= learning_rate * grad_weights[idx];
        }
    }

    // Cập nhật bias
    for (int j = 0; j < output_size; ++j)
    {
        bias[j] -= learning_rate * grad_bias[j];
    }
}


void load_data(const char *filename, float* data, int size)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        printf("Không thể mở file\n");
        return;
    }
    
    file.read(reinterpret_cast<char*>(data), size * sizeof(float));

    // Kiểm tra số lượng phần tử đọc được
    if (file.gcount() != size * sizeof(float))
    {
        printf("Số phần tử đọc được không khớp\n");
    }
    file.close();
}


void load_labels(const char *filename, int* labels, int size)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        printf("Không thể mở file\n");
        return;
    }
    
    file.read(reinterpret_cast<char*>(labels), size * sizeof(int));

    // Kiểm tra số lượng phần tử đọc được
    if (file.gcount() != size * sizeof(int))
    {
        printf("Số phần tử đọc được không khớp\n");
    }
    file.close();
}


// Thêm hàm forward pass riêng
void forward_pass(float* input, float* hidden1, float* hidden2, 
                  float* output, NeuralNetwork &nn, 
                  int batch_size, int start_idx)
{
    // Layer 1
    forwardLayer(input, nn.weightsInputHidden1, nn.biasHidden1, 
                 hidden1, INPUT_SIZE, HIDDEN1_SIZE, batch_size, true);

    // Layer 2
    forwardLayer(hidden1, nn.weightsHidden1Hidden2, nn.biasHidden2, 
                 hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE, batch_size, true);

    // Output layer
    forwardLayer(hidden2, nn.weightsHidden2Output, nn.biasOutput, 
                 output, HIDDEN2_SIZE, OUTPUT_SIZE, batch_size, false);

    // Softmax
    softmax_cpu(output, batch_size, OUTPUT_SIZE);
}



void backward_pass(float* input, float* hidden1,float* hidden2,float* output,float* del_output,
                   float* d_ReLU_out1,float* d_ReLU_out2,NeuralNetwork &nn,int batch_size, int start_idx)
{
    // Compute gradients for output layer
    computeGradients(hidden2, del_output, nn.gradWeightsHidden2Output,
                     nn.gradBiasOutput, batch_size, HIDDEN2_SIZE, OUTPUT_SIZE);

    // Compute delta for hidden layer 2
    computeDeltaReLU(del_output, nn.weightsHidden2Output, hidden1, d_ReLU_out2, batch_size, OUTPUT_SIZE, HIDDEN2_SIZE);

    // Compute gradients for hidden layer 1
    computeGradients(hidden1, d_ReLU_out2, nn.gradWeightsHidden1Hidden2,
                     nn.gradBiasHidden2, batch_size, HIDDEN1_SIZE, HIDDEN2_SIZE);

    // Compute delta for hidden layer 1
    computeDeltaReLU(d_ReLU_out2, nn.weightsHidden1Hidden2, input, d_ReLU_out1, batch_size, HIDDEN2_SIZE, HIDDEN1_SIZE);

    // Compute gradients for input layer
    computeGradients(input, d_ReLU_out1, nn.gradWeightsInputHidden1,
                     nn.gradBiasHidden1, batch_size, INPUT_SIZE, HIDDEN1_SIZE);
}


void train(NeuralNetwork &nn, float* X_train, int* y_train) {
    // Khởi tạo bộ nhớ cho các tầng mạng
    float* hidden1 = new float[BATCH_SIZE * HIDDEN1_SIZE];
    float* hidden2 = new float[BATCH_SIZE * HIDDEN2_SIZE];
    float* output = new float[BATCH_SIZE * OUTPUT_SIZE]();
       

    float* del_output = new float[BATCH_SIZE * OUTPUT_SIZE];
    float* d_ReLU_out1 = new float[BATCH_SIZE * HIDDEN1_SIZE];
    float* d_ReLU_out2 = new float[BATCH_SIZE * HIDDEN2_SIZE];

    int num_batches = TRAIN_DATA_SIZE / BATCH_SIZE;

    // Vòng lặp huấn luyện
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;

        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;

            // Forward pass
            forward_pass(
                X_train + start_idx * INPUT_SIZE, hidden1, hidden2, output, nn, BATCH_SIZE, start_idx
            );

            // Tính toán loss và độ chính xác
            total_loss += compute_loss(output, &y_train[start_idx], BATCH_SIZE);
            correct += validatePredictions(output, y_train, start_idx, BATCH_SIZE, OUTPUT_SIZE);

            // Chuẩn bị cho backpropagation
            for (int b = 0; b < BATCH_SIZE; b++) {
                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    del_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i] - ((i == y_train[start_idx + b]) ? 1.0f : 0.0f);
                }
            }

            // Backward pass
            backward_pass(
                X_train, hidden1, hidden2, output, del_output,
                d_ReLU_out1, d_ReLU_out2, nn, BATCH_SIZE, start_idx
            );

            // Cập nhật weights
            updateWeights(nn.weightsInputHidden1, nn.gradWeightsInputHidden1, nn.biasHidden1, nn.gradBiasHidden1, HIDDEN1_SIZE, INPUT_SIZE, LEARNING_RATE);
            updateWeights(nn.weightsHidden1Hidden2, nn.gradWeightsHidden1Hidden2, nn.biasHidden2, nn.gradBiasHidden2, HIDDEN2_SIZE, HIDDEN1_SIZE, LEARNING_RATE);
            updateWeights(nn.weightsHidden2Output, nn.gradWeightsHidden2Output, nn.biasOutput, nn.gradBiasOutput, OUTPUT_SIZE, HIDDEN2_SIZE, LEARNING_RATE);
        }

        // In thông tin mỗi epoch
        printf("Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%\n",
               epoch + 1, EPOCHS,
               total_loss / num_batches,
               100.0f * correct / TRAIN_DATA_SIZE);
    }

    // Giải phóng bộ nhớ đã cấp phát
    delete[] hidden1;
    delete[] hidden2;
    delete[] output;
    delete[] del_output;
    delete[] d_ReLU_out1;
    delete[] d_ReLU_out2;
}


void test(NeuralNetwork &nn, float* X_test, int* y_test)
{
    // Biến lưu trữ kết quả
    int num_batches = TEST_DATA_SIZE / BATCH_SIZE;
    int correct = 0;

    // Các mảng để lưu trữ trung gian
    float* hidden1 = new float[BATCH_SIZE * HIDDEN1_SIZE]();
    float* hidden2 = new float[BATCH_SIZE * HIDDEN2_SIZE]();
    float* output = new float[BATCH_SIZE * OUTPUT_SIZE]();

    for (int batch = 0; batch < num_batches; batch++)
    {
        int start_idx = batch * BATCH_SIZE;

        // Chuẩn bị input batch
        float* input = X_test + start_idx * INPUT_SIZE;

        // Forward pass
        forward_pass(input, hidden1, hidden2, output, nn, BATCH_SIZE, start_idx);

        // Kiểm tra và tính độ chính xác
        correct += validatePredictions(output, y_test, start_idx, BATCH_SIZE, OUTPUT_SIZE);
    }

    printf("Test Accuracy: %.2f%%\n", 100.0f * correct / TEST_DATA_SIZE);

    // Giải phóng bộ nhớ
    delete[] hidden1;
    delete[] hidden2;
    delete[] output;
}


int main(int argc, char **argv)
{
    srand(time(NULL));

    // Khởi tạo mạng neural
    NeuralNetwork nn;
    initializeNetwork(&nn);

    // Tạo và khởi tạo dữ liệu
    float* X_train = new float[TRAIN_DATA_SIZE * INPUT_SIZE]();  // Khởi tạo mảng dữ liệu huấn luyện
    int* y_train = new int[TRAIN_DATA_SIZE]();                    // Khởi tạo mảng nhãn huấn luyện

    float* X_test = new float[TEST_DATA_SIZE * INPUT_SIZE]();    // Khởi tạo mảng dữ liệu kiểm thử
    int* y_test = new int[TEST_DATA_SIZE]();                      // Khởi tạo mảng nhãn kiểm thử

    // Load dữ liệu từ file
    load_data("x_train.bin", X_train, TRAIN_DATA_SIZE * INPUT_SIZE);
    load_labels("y_train.bin", y_train, TRAIN_DATA_SIZE);
    load_data("x_test.bin", X_test, TEST_DATA_SIZE * INPUT_SIZE);
    load_labels("y_test.bin", y_test, TEST_DATA_SIZE);

    // Huấn luyện
    train(nn, X_train, y_train);

    // Kiểm thử
    test(nn, X_test, y_test);

    // Giải phóng bộ nhớ
    delete[] X_train;
    delete[] y_train;
    delete[] X_test;
    delete[] y_test;

    return 0;
}
