
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void matMulCoalesced(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N)
{
   // Global row and column
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if (row < M && col < N)
   {
      float sum = 0.0f;

      for (int i = 0; i < K; i++)
      {
         // Coalesced access to B
         float a = A[row * K + i];
         float b = B[i * N + col];

         sum += a * b;
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

   // Host matrices
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
           endCPU - startCPU)
           .count();

   std::cout << "\nTime taken to initialize and fill matrices: "
             << endTime << " ms" << std::endl;

   // Device pointers
   float *d_A, *d_B, *d_C;

   // Allocate GPU memory
   cudaMalloc((void **)&d_A, sizeof(float) * M * K);
   cudaMalloc((void **)&d_B, sizeof(float) * K * N);
   cudaMalloc((void **)&d_C, sizeof(float) * M * N);

   // Copy matrices to GPU
   cudaMemcpy(d_A, A.data(),
              sizeof(float) * M * K,
              cudaMemcpyHostToDevice);

   cudaMemcpy(d_B, B.data(),
              sizeof(float) * K * N,
              cudaMemcpyHostToDevice);

   // CUDA thread configuration
   dim3 threadsBlock(32, 8);

   dim3 numBlocks(
       (N + threadsBlock.x - 1) / threadsBlock.x,
       (M + threadsBlock.y - 1) / threadsBlock.y);

   cudaEvent_t start, stop;

   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   // Launch kernel
   cudaEventRecord(start);

   matMulCoalesced<<<numBlocks, threadsBlock>>>(
       d_A, d_B, d_C, M, K, N);

   cudaError_t err = cudaGetLastError();

   if (err != cudaSuccess)
   {
      std::cerr << cudaGetErrorString(err) << std::endl;
   }

   cudaEventRecord(stop);
   cudaEventSynchronize(stop);

   float ms;

   cudaEventElapsedTime(&ms, start, stop);

   // Copy output back to CPU
   cudaMemcpy(C.data(), d_C,
              sizeof(float) * M * N,
              cudaMemcpyDeviceToHost);

   // Print first 6 elements
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

   double bandwidth = bytes / (ms / 1000.0) / 1e9;

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