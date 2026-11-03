
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

#define TILE_SIZE 16

// Shared Memory Tiled FP32 GEMM
__global__ void matMulTiled(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N)
{
    // Shared memory tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Global row and column
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Number of tiles along K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++)
    {
        // Global memory indices
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // Load A tile into shared memory
        if (row < M && aCol < K)
            tileA[threadIdx.y][threadIdx.x] =
                A[row * K + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile into shared memory
        if (bRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] =
                B[bRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Ensure all threads finish loading
        __syncthreads();

        // Compute partial matrix multiplication
        for (int i = 0; i < TILE_SIZE; i++)
        {
            sum += tileA[threadIdx.y][i] *
                   tileB[i][threadIdx.x];
        }

        // Ensure all threads finish using current tiles
        __syncthreads();
    }

    // Store final result
    if (row < M && col < N)
    {
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

    // Initialize matrices
    auto startCPU = std::chrono::steady_clock::now();

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    for (auto &s : A)
        s = dist(rng);

    for (auto &s : B)
        s = dist(rng);

    auto endCPU = std::chrono::steady_clock::now();

    double endTime =
        std::chrono::duration<double, std::milli>(
            endCPU - startCPU).count();

    std::cout << "\nTime taken to initialize and fill matrices: "
              << endTime << " ms" << std::endl;

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate GPU memory
    cudaMalloc((void **)&d_A, sizeof(float) * M * K);
    cudaMalloc((void **)&d_B, sizeof(float) * K * N);
    cudaMalloc((void **)&d_C, sizeof(float) * M * N);

    // Copy data to GPU
    cudaMemcpy(d_A, A.data(),
               sizeof(float) * M * K,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, B.data(),
               sizeof(float) * K * N,
               cudaMemcpyHostToDevice);

    // Thread configuration
    dim3 threadsBlock(TILE_SIZE, TILE_SIZE);

    dim3 numBlocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch tiled GEMM kernel
    cudaEventRecord(start);

    matMulTiled<<<numBlocks, threadsBlock>>>(
        d_A, d_B, d_C, M, K, N
    );

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: "
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;

    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back to CPU
    cudaMemcpy(C.data(), d_C,
               sizeof(float) * M * N,
               cudaMemcpyDeviceToHost);

    // Print first six output elements
    std::cout<<"\nFirst 6 Elements of the C matrix:"<<std::endl;
    for (int i = 0; i < 6; i++)
    {
        std::cout << C[i] << std::endl;
    }

    // Performance calculations
    double seconds = ms / 1000.0;

    double flops = 2.0 * M * N * K;

    double gflops = flops / seconds / 1e9;

    double bytes =
        sizeof(float) * (M * K + K * N + M * N);

    double bandwidth =
        bytes / (ms / 1000.0) / 1e9;

    double arith = flops / bytes;

    std::cout << "\n\nSeconds required for kernel: "
              << seconds << std::endl;

    std::cout << "\nFLOPS: "
              << flops << std::endl;

    std::cout << "\nGFLOPS: "
              << gflops << std::endl;

    std::cout << "\nBytes: "
              << bytes << std::endl;

    std::cout << "\nBandWidth in GB/s: "
              << bandwidth << std::endl;

    std::cout << "\nArithmetic Intensity: "
              << arith << std::endl;

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}