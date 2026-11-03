
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

#define BM 64
#define BN 64
#define BK 16

#define TM 4
#define TN 4

#define THREADS_X (BN / TN)
#define THREADS_Y (BM / TM)

// Register-blocked FP32 GEMM
__global__ void matMulRegisterTiled(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N)
{
    // Shared memory tiles
    __shared__ float tileA[BM][BK];
    __shared__ float tileB[BK][BN];

    // Thread coordinates inside block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Starting output coordinates for this thread
    int row = blockIdx.y * BM + ty * TM;
    int col = blockIdx.x * BN + tx * TN;

    // Register accumulators
    float accum[TM][TN] = {0.0f};

    // Number of K tiles
    int numTiles = (K + BK - 1) / BK;

    for (int t = 0; t < numTiles; t++)
    {
        // Cooperatively load A tile
        for (int idx = ty * THREADS_X + tx;
             idx < BM * BK;
             idx += THREADS_X * THREADS_Y)
        {
            int r = idx / BK;
            int c = idx % BK;

            int globalRow = blockIdx.y * BM + r;
            int globalCol = t * BK + c;

            if (globalRow < M && globalCol < K)
                tileA[r][c] = A[globalRow * K + globalCol];
            else
                tileA[r][c] = 0.0f;
        }

        // Cooperatively load B tile
        for (int idx = ty * THREADS_X + tx;
             idx < BK * BN;
             idx += THREADS_X * THREADS_Y)
        {
            int r = idx / BN;
            int c = idx % BN;

            int globalRow = t * BK + r;
            int globalCol = blockIdx.x * BN + c;

            if (globalRow < K && globalCol < N)
                tileB[r][c] = B[globalRow * N + globalCol];
            else
                tileB[r][c] = 0.0f;
        }

        __syncthreads();

        // Compute register tile
        for (int k = 0; k < BK; k++)
        {
            float aReg[TM];
            float bReg[TN];

            // Load A values into registers
            #pragma unroll
            for (int i = 0; i < TM; i++)
            {
                aReg[i] = tileA[ty * TM + i][k];
            }

            // Load B values into registers
            #pragma unroll
            for (int j = 0; j < TN; j++)
            {
                bReg[j] = tileB[k][tx * TN + j];
            }

            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < TM; i++)
            {
                #pragma unroll
                for (int j = 0; j < TN; j++)
                {
                    accum[i][j] += aReg[i] * bReg[j];
                }
            }
        }

        __syncthreads();
    }

    // Write output tile
    #pragma unroll
    for (int i = 0; i < TM; i++)
    {
        #pragma unroll
        for (int j = 0; j < TN; j++)
        {
            int globalRow = row + i;
            int globalCol = col + j;

            if (globalRow < M && globalCol < N)
            {
                C[globalRow * N + globalCol] = accum[i][j];
            }
        }
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

    cudaMalloc((void **)&d_A, sizeof(float) * M * K);
    cudaMalloc((void **)&d_B, sizeof(float) * K * N);
    cudaMalloc((void **)&d_C, sizeof(float) * M * N);

    // Copy matrices to device
    cudaMemcpy(d_A, A.data(),
               sizeof(float) * M * K,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, B.data(),
               sizeof(float) * K * N,
               cudaMemcpyHostToDevice);

    // Thread configuration
    dim3 threadsBlock(THREADS_X, THREADS_Y);

    dim3 numBlocks(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    cudaEventRecord(start);

    matMulRegisterTiled<<<numBlocks, threadsBlock>>>(
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

    // Copy output back to host
    cudaMemcpy(C.data(), d_C,
               sizeof(float) * M * N,
               cudaMemcpyDeviceToHost);

    // Print first six elements
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