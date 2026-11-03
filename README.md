MultiStage GEMM — CUDA Matrix Multiplication Optimization

A CUDA-based exploration of General Matrix Multiplication (GEMM), demonstrating how GPU kernel optimization techniques improve matrix multiplication performance.

This project implements and compares multiple CUDA approaches, progressing from a basic matrix multiplication kernel to shared-memory tiling, register blocking, memory coalescing, and asynchronous double buffering.

The objective is to understand how GPU memory hierarchy, thread organization, data reuse, and instruction-level optimization influence GEMM performance.

Overview

Matrix multiplication is a fundamental operation in high-performance computing, machine learning, and scientific applications.

Given two matrices:

𝐴
 of dimensions 
𝑀
×
𝐾
𝐵
 of dimensions 
𝐾
×
𝑁

The resulting matrix 
𝐶
 has dimensions 
𝑀
×
𝑁
:

𝐶
𝑖
𝑗
=
∑
𝑘
=
0
𝐾
−
1
𝐴
𝑖
𝑘
𝐵
𝑘
𝑗

This repository explores different CUDA implementations of this operation and evaluates their performance on a large FP32 matrix multiplication workload.

Implementations

The repository contains five CUDA implementations, each exploring a different optimization strategy.

File	Implementation	Optimization
MatMul.cu	Naive GEMM	Basic CUDA kernel
Coal.cu	Coalesced GEMM	Improved global memory access
tiledGEMM.cu	Shared-memory tiled GEMM	Data reuse through shared memory
RegisterTiled.cu	Register-tiled GEMM	Multiple output accumulators per thread
DoubleShared.cu	Double-buffered GEMM	Asynchronous data movement and pipelining
1. Naive Matrix Multiplication — MatMul.cu

The baseline implementation assigns individual output elements to CUDA threads.

Each thread computes a dot product by repeatedly accessing matrix elements from global memory.

This implementation provides a starting point for understanding the performance limitations of a straightforward GPU implementation.

2. Coalesced Memory Access — Coal.cu

This implementation explores memory coalescing by organizing threads to access consecutive elements of the second input matrix.

Coalesced global memory accesses help reduce the number of memory transactions and improve memory access efficiency.

3. Shared Memory Tiling — tiledGEMM.cu

This implementation uses shared memory to improve data reuse.

Instead of repeatedly fetching matrix elements from global memory, CUDA thread blocks cooperatively load smaller tiles of the input matrices into shared memory.

Each thread then performs partial multiplication using these cached tiles.

Key techniques:

Shared-memory tiling.
Cooperative loading of matrix tiles.
Synchronization using __syncthreads().
Reduced redundant global memory accesses.
4. Register Tiling — RegisterTiled.cu

This implementation extends shared-memory tiling by allowing each thread to compute multiple output elements.

The kernel uses:

A 64 × 64 output tile.
A reduction tile size of 16.
A 4 × 4 register accumulator per thread.

Instead of computing a single output element, each thread maintains multiple accumulators in registers.

This increases data reuse and reduces the overhead associated with repeatedly loading values from shared memory.

Key techniques:

Register blocking.
Shared-memory reuse.
Outer-product accumulation.
Loop unrolling.
Multiple output elements per thread.
5. Double-Buffered Shared Memory — DoubleShared.cu

The final implementation explores double buffering using CUDA asynchronous pipelines.

Two shared-memory buffers are maintained for input tiles. While one buffer is being processed, the next tile can be prefetched into the alternate buffer.

This helps overlap memory movement with computation.

Key techniques:

Double-buffered shared memory.
Asynchronous memory copies using cuda::memcpy_async.
CUDA pipeline synchronization.
Cooperative Groups.
Register-based accumulation.

This implementation explores a more advanced approach to hiding memory latency and improving GPU utilization.

Benchmark Results

The following results are taken from outputs.txt.

Experimental Configuration
Parameter	Value
Matrix A	4096 × 4096
Matrix B	4096 × 4096
Output matrix C	4096 × 4096
Data type	FP32 (float)
Input initialization	Random values in [-1, 1]
Random seed	42
Total floating-point operations	~137.44 billion
Performance Comparison
Implementation	Kernel Execution Time	Performance
Naive GEMM	71.1126 s	1.9327 GFLOPS
Coalesced GEMM	7.24481 s	18.9707 GFLOPS
Shared Memory Tiled	1.46799 s	93.624 GFLOPS
Register Tiled	0.273339 s	502.816 GFLOPS
Double Buffered	Not available	Not available
Performance Visualization

The reported GFLOPS demonstrate the improvement obtained through successive optimization techniques.

Implementation	Relative Performance
Naive	▏ 1.93
Coalesced	█ 18.97
Shared Memory Tiled	█████ 93.62
Register Tiled	█████████████████████████ 502.82

The visualization is illustrative; values are the reported GFLOPS.

Key Observations
Memory coalescing: The coalesced implementation achieves approximately 9.8× the reported throughput of the naive implementation.
Shared memory: Tiling improves performance by enabling reuse of input matrix elements and reducing redundant global memory accesses.
Register blocking: Register tiling delivers the largest measured improvement, reaching 502.816 GFLOPS.
Double buffering: The double-buffered implementation introduces asynchronous data movement to explore overlapping memory transfers and computation. Its performance has not yet been recorded in outputs.txt.

Overall, the measured register-tiled implementation achieves approximately 260× the reported throughput of the naive implementation.

Note: These are recorded measurements, not a controlled cross-platform comparison. Kernel execution times and reported GFLOPS are reproduced from outputs.txt. The benchmark output does not establish numerical correctness across all implementations, and the naive kernel uses a different B-matrix indexing expression. Correctness should be validated against a CPU reference before treating the timings as directly comparable.
Project Structure
text
MultiStage_GEMM/
│
├── MatMul.cu          # Naive matrix multiplication
├── Coal.cu            # Coalesced memory access
├── tiledGEMM.cu       # Shared-memory tiled GEMM
├── RegisterTiled.cu   # Register-tiled GEMM
├── DoubleShared.cu    # Double-buffered shared-memory GEMM
│
├── outputs.txt        # Recorded benchmark results
└── README.md          # Project documentation

Requirements

To compile and execute the implementations, you need:

NVIDIA GPU with CUDA support.
NVIDIA CUDA Toolkit.
CUDA-compatible C++ compiler (nvcc).
NVIDIA GPU driver compatible with your CUDA Toolkit.

The double-buffered implementation additionally requires CUDA pipeline and asynchronous copy support.

Building and Running

Clone the repository:

bash
git clone https://github.com/abhiraj-budnur/MultiStage_GEMM.git

cd MultiStage_GEMM


Compile each implementation independently using NVIDIA's CUDA compiler.

Naive GEMM
bash
nvcc MatMul.cu -o matmul
./matmul

Coalesced GEMM
bash
nvcc Coal.cu -o coal
./coal

Shared Memory Tiled GEMM
bash
nvcc tiledGEMM.cu -o tiled
./tiled

Register Tiled GEMM
bash
nvcc RegisterTiled.cu -o register_tiled
./register_tiled

Double-Buffered GEMM
bash
nvcc DoubleShared.cu -o double_shared
./double_shared


For a specific GPU architecture, you can also specify the appropriate compute capability using -arch, for example:

bash
nvcc -arch=sm_80 RegisterTiled.cu -o register_tiled


Replace sm_80 with the architecture supported by your GPU.

Future Improvements

Potential directions for further optimization include:

Benchmarking double-buffered GEMM against register-tiled GEMM.
Implementing multi-stage asynchronous pipelines.
Exploring warp-level matrix operations using Tensor Cores.
Experimenting with different tile dimensions and thread-block configurations.
Profiling kernels using NVIDIA Nsight Compute.
Comparing performance across different GPU architectures.
Adding CPU reference implementations for numerical correctness validation.
Measuring effective memory bandwidth and hardware utilization.
Learning Outcomes

This project demonstrates several important GPU optimization concepts:

CUDA thread and block organization.
Global memory coalescing.
Shared-memory tiling and data reuse.
Register blocking and accumulation.
Synchronization and race-condition avoidance.
Asynchronous memory operations.
Double buffering and latency hiding.
GPU performance measurement using GFLOPS.
Author

Abhiraj Budnur

GitHub: @abhiraj-budnur