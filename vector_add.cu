#include <iostream>
#include <vector>

__global__ void vecAdd(float* A, float* B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int n = 100000;
    size_t bytes = n * sizeof(float);

    // Allocate memory on the host
    std::vector<float> h_A(n, 1.1f);
    std::vector<float> h_B(n, 2.2f);
    std::vector<float> h_C(n);

    // Allocate memory on the device
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from the host to the device
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // Set up the kernel launch parameters
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // Copy data from the device to the host
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print the results
    for (int i = 0; i < n; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
