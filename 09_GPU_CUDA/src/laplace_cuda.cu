#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Custom version of atomicMax for float, since Nvidia does not support an
// official "atomicMax" function for floats
static inline __device__ float atomicMax(float *addr, float val) {
  unsigned int old = __float_as_uint(*addr), assumed;
  do {
    assumed = old;
    if (__uint_as_float(old) >= val)
      break;

    old = atomicCAS((unsigned int *)addr, assumed, __float_as_uint(val));
  } while (assumed != old);

  return __uint_as_float(old);
}

__global__ void dev_laplace_error(float *A, float *Anew, float *error_dev, int n, int m) {
  // Set indices - add 1 because computation starts at index 1 (avoiding boundaries)
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  // Boundary check: ensure we don't exceed the computation region
  if (i >= m-1 || j >= n-1) return;

  // Calculate linear index for 1D array access
  int idx = j * m + i;

  // 5-point stencil computation: average of 4 neighbors
  Anew[idx] = (A[idx-1] + A[idx+1] + A[idx-m] + A[idx+m]) * 0.25f;

  // Compute local error and update global maximum error using atomic operation
  float local_error = sqrtf(fabsf(A[idx] - Anew[idx]));
  atomicMax(error_dev, local_error);
}

void laplace_init(float *in, int n, int m) {
  int i, j;
  const float pi = 2.0f * asinf(1.0f);
  memset(in, 0, n * m * sizeof(float));
  
  // Initialize boundaries
  for (i = 0; i < m; i++) {
    in[i] = 0.f;                    // Top boundary
    in[(n - 1) * m + i] = 0.f;     // Bottom boundary
  }
  
  for (j = 0; j < n; j++) {
    in[j * m] = sinf(pi * j / (n - 1));                    // Left boundary
    in[j * m + m - 1] = sinf(pi * j / (n - 1)) * expf(-pi); // Right boundary
  }
}

int main(int argc, char **argv) {
  int n = 4096, m = 4096;
  int iter_max = 100, THREADS_BLOCK = 16;
  float *A;

  const float tol = 1.0e-8f;
  float error = 1.0f;

  // Parse command line arguments
  if (argc > 1) n = atoi(argv[1]);
  if (argc > 2) m = atoi(argv[2]);
  if (argc > 3) iter_max = atoi(argv[3]);
  if (argc > 4) THREADS_BLOCK = atoi(argv[4]);

  // Allocate host memory
  A = (float *)malloc(n * m * sizeof(float));

  // Initialize the problem
  laplace_init(A, n, m);
  A[(n / 128) * m + m / 128] = 1.0f; // Set singular point

  printf("Jacobi relaxation Calculation: %d x %d mesh, "
         "maximum of %d iterations. Threads per block= %d\n",
         n, m, iter_max, THREADS_BLOCK);

  // Allocate GPU memory
  float *A_dev, *Anew_dev, *error_dev;
  cudaMalloc(&A_dev, n * m * sizeof(float));
  cudaMalloc(&Anew_dev, n * m * sizeof(float));
  cudaMalloc(&error_dev, sizeof(float));

  // Copy initial data to GPU
  cudaMemcpy(A_dev, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Anew_dev, A, n * m * sizeof(float), cudaMemcpyHostToDevice);

  // Configure GPU execution parameters
  // We compute interior points only (exclude boundaries), so n-2 by m-2 region
  int n_matrix_to_compute = n - 2;
  int m_matrix_to_compute = m - 2;
  
  dim3 gridDim((m_matrix_to_compute + THREADS_BLOCK - 1) / THREADS_BLOCK,
               (n_matrix_to_compute + THREADS_BLOCK - 1) / THREADS_BLOCK);
  dim3 blockDim(THREADS_BLOCK, THREADS_BLOCK);

  printf("Grid dimensions: %d x %d, Block dimensions: %d x %d\n",
         gridDim.x, gridDim.y, blockDim.x, blockDim.y);

  int iter = 0;
  while (error > tol && iter < iter_max) {
    iter++;

    // Reset error for this iteration
    cudaMemset(error_dev, 0, sizeof(float));
    
    // Launch CUDA kernel
    dev_laplace_error<<<gridDim, blockDim>>>(A_dev, Anew_dev, error_dev, n, m);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
      break;
    }

    // Copy error back to host
    cudaMemcpy(&error, error_dev, sizeof(float), cudaMemcpyDeviceToHost);

    // Swap pointers for next iteration
    float *swap = A_dev;
    A_dev = Anew_dev;
    Anew_dev = swap;

    // Print progress
    if (iter % (iter_max / 10) == 0 || iter == 1)
      printf("%5d, %0.6f\n", iter, error);
  }

  // Copy final result back to host
  cudaMemcpy(A, A_dev, n * m * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
  printf("A[%d][%d]= %0.6f\n", n / 128, m / 128, A[(n / 128) * m + m / 128]);

  // Cleanup
  cudaFree(A_dev);
  cudaFree(Anew_dev);
  cudaFree(error_dev);
  free(A);

  return 0;
}