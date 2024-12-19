# Neural Network Implementation with CUDA

## Team Members
- 21120409 - Nguyễn Đức Duy Anh
- 21120167 - Cao Thị Yến Vy
- 21120451 - Lê Bảo Hiếu

## Project Description
This project implements a neural network using CUDA for parallel processing. The neural network consists of:
- Input layer: 784 neurons
- Two hidden layers: 128 neurons each
- Output layer: 10 neurons

The implementation includes:
- Forward and backward propagation
- Batch processing
- ReLU activation function
- Softmax output layer
- Cross-entropy loss function

## Requirements
- CUDA Toolkit (>= 10.0)
- GPU with compute capability >= 3.0
- GCC/G++ compiler
- Make build system

## Project Structure
```
Parallel_Programing/
├── device/
│ ├── neural_network.cuh
│ ├── cuda_utils.cuh
│ ├── data_loader.cuh
│ ├── kernels.cuh
│ ├── training.cuh
│ ├── neural_network.cu
│ ├── cuda_utils.cu
│ ├── data_loader.cu
│ ├── kernels.cu
│ ├── training.cu
│ └── main.cu
├── Makefile
└── README.md
```
## Building and Running

### Local Machine
1. Make sure CUDA toolkit is installed:
```bash
nvcc --version
```
2. Clone the repository:
```bash
git clone https://github.com/duyanh711/Parallel_Programing.git
cd Parallel_Programing
```
3. Build the project:
```bash
make
```
4. Run the project:
```bash
./main
```


### Google Colab
1. Create a new notebook and select GPU runtime
2. Mount your Google Drive (optional)
3. Clone the repository or upload project files
4. Install required dependencies:
```bash
!nvcc --version
```
5. Build and run:
```bash
!make
!./main
```
