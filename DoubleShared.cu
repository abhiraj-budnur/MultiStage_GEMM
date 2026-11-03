
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

#define BM 64
#define BN 64
#define BK 16

#define THREADS 256

namespace cg = cooperative_groups;

// Double-buffered shared-memory FP32 GEMM
__global__ void matMulDoubleBuffered(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N)
{
    // Two shared-memory buffers
    __shared__ float tileA[2][BM][BK];
    __shared__ float tileB[2][BK][BN];

    // Block-wide asynchronous pipeline
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block, 2> pipelineState;

    auto block = cg::this_thread_block();

    auto pipe = cuda::make_pipeline(block, &pipelineState);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Thread coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Register accumulators
    float accum[4][4] = {0.0f};

    int row = blockIdx.y * BM + ty * 4;
    int col = blockIdx.x * BN + tx * 4;

    int numTiles = (K + BK - 1) / BK;

    // Cooperatively load one tile into a buffer
    auto loadTile = [&](int buffer, int tile)
    {
        pipe.producer_acquire();

        // Load A tile
        for (int idx = tid; idx < BM * BK; idx += THREADS)
        {
            int r = idx / BK;
            int c = idx % BK;

            int globalRow = blockIdx.y * BM + r;
            int globalCol = tile * BK + c;

            float *dst = &tileA[buffer][r][c];

            if (globalRow < M && globalCol < K)
            {
                cuda::memcpy_async(
                    dst,
                    &A[globalRow * K + globalCol],
                    sizeof(float),
                    pipe
                );
            }
            else
            {
                *dst = 0.0f;
            }
        }

        // Load B tile
        for (int idx = tid; idx < BK * BN; idx += THREADS)
        {
            int r = idx / BN;
            int c = idx % BN;

            int globalRow = tile * BK + r;
            int globalCol = blockIdx.x * BN + c;

            float *dst = &tileB[buffer][r][c];

            if (globalRow < K && globalCol < N)
            {
                cuda::memcpy_async(
                    dst,
                    &B[globalRow * N + globalCol],
                    sizeof(float),
                    pipe
                );
            }
            else
            {
                *dst = 0.0f;
            }
        }

        pipe.producer_commit();
    };

    // Initially load the first tile
    loadTile(0, 0);

    // Prefetch second tile into alternate buffer
    if (numTiles > 1)
        loadTile(1, 1);

    // Process all K tiles
    for (int t = 0; t < numTiles; t++)
    {
        int buffer = t % 2;

        // Wait for current tile to finish loading
        pipe.consumer_wait();

        // Compute using shared memory
        #pragma unroll
        for (int k = 0; k < BK; k++)
        {
            float aReg[4];
            float bReg[4];

            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                aReg[i] =
                    tileA[buffer][ty * 4 + i][k];
            }

            #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                bReg[j] =
                    tileB[buffer][k][tx * 4 + j];
            }

            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    accum[i][j] += aReg[i] * bReg[j];
                }
            }
        }

        // Release the consumed buffer
        pipe.consumer_release();

        // Prefetch a future tile into the freed buffer
        if (t + 2 < numTiles)
        {
            loadTile(buffer, t + 2);
        }
    }

    // Store output
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
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
    dim3 threadsBlock(16, 16);

    dim3 numBlocks(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    cudaEventRecord(start);

    matMulDoubleBuffered<<<numBlocks, threadsBlock>>>(
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

    // Copy result back
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
