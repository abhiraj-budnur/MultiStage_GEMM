#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>

__global__ void matMul(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < K && row < M)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
        {
            sum += A[row * K + i] * B[col * N + i];
        }
        C[row * N + col] = sum;
    }
}

int main()
{

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    int M = 4096;
    int N = 4096;
    int K = 4096;

    // size_t elements = M * N;

    auto startCPU = std::chrono::steady_clock::now();
    std::vector<float> A(M * N);
    std::vector<float> B(N * K);
    std::vector<float> C(M * K);

    for (auto &s : A)
        s = dist(rng);
    for (auto &s : B)
        s = dist(rng);

    auto endCPU = std::chrono::steady_clock::now();
    double endTime = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();

    std::cout << "\nTime taken to initialize and fill matrices: " << endTime << std::endl;

    // Device Pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory on GPU Buffers
    cudaMalloc((void **)&d_A, sizeof(float) * (M * N));
    cudaMalloc((void **)&d_B, sizeof(float) * (N * K));
    cudaMalloc((void **)&d_C, sizeof(float) * (M * K));

    // Copy data to device
    cudaMemcpy(d_A, A.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeof(float) * N * K, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_C, C, sizeof(float) * M * K, cudaMemcpyHostToDevice);

    dim3 threadsBlock(32, 32);
    dim3 numBlocks(
        (K + threadsBlock.x - 1) / threadsBlock.x,
        (M + threadsBlock.y - 1) / threadsBlock.y);
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Launch Kernel

    cudaEventRecord(start);
    matMul<<<numBlocks, threadsBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
    // Wait for Device
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;

    cudaEventElapsedTime(&ms, start, stop);
    // Copy from device to host
    cudaMemcpy(C.data(), d_C, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

    std::cout<<"\nFirst 6 Elements of the C matrix:"<<std::endl;
    for (int i = 0; i <= 5; i++)
    {
        std::cout << C[i] << std::endl;
    }

    double seconds = ms / 1000.0;
    double flops = 2.0 * M * N * K;
    double gflops = flops / seconds / 1e9;
    double bytes = sizeof(float) * (M * N + N * K + K * M);
    double bandwidth = bytes / ms * (int)pow(10, 6);
    double arith = flops / bytes;
    std::cout << "\n\nSeconds required for kernel: " << seconds << std::endl;
    std::cout << "\nFLOPS: " << flops << std::endl;
    std::cout << "\nGFLOPS: " << gflops << std::endl;
    std::cout << "\nBytes: " << bytes << std::endl;
    std::cout << "\nBandWidth in GB/s: " << bandwidth << std::endl;
    std::cout << "\nArithmetic Intensity: " << arith << std::endl;

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}