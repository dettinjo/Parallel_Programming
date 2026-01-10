#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Optimized atomicMax for float
__device__ float atomicMax(float *addr, float val) {
  unsigned int old = __float_as_uint(*addr), assumed;
  do {
    assumed = old;
    if (__uint_as_float(old) >= val) break;
    old = atomicCAS((unsigned int *)addr, assumed, __float_as_uint(val));
  } while (assumed != old);
  return __uint_as_float(old);
}

// Optimized kernel with better memory access patterns and occupancy
__global__ void dev_laplace_error_v3(float *A, float *Anew, float *error_dev, 
                                     int n, int m, int compute_error) {
  // Use 1D indexing for better memory coalescing
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  
  // Process multiple points per thread to improve arithmetic intensity
  int total_points = (n-2) * (m-2);
  
  for (int linear_idx = tid; linear_idx < total_points; linear_idx += stride) {
    // Convert back to 2D coordinates
    int j = (linear_idx / (m-2)) + 1;  // Row
    int i = (linear_idx % (m-2)) + 1;  // Column
    
    int idx = j * m + i;
    
    // 5-point stencil with optimized memory access
    float center = A[idx];
    float left   = A[idx - 1];
    float right  = A[idx + 1];
    float top    = A[idx - m];
    float bottom = A[idx + m];
    
    // Compute new value
    float new_val = 0.25f * (left + right + top + bottom);
    Anew[idx] = new_val;
    
    // Only compute error when needed (not every iteration)
    if (compute_error) {
      float local_error = fabsf(center - new_val);
      if (local_error > 0.0f) {
        atomicMax(error_dev, local_error);
      }
    }
  }
}

void laplace_init(float *in, int n, int m) {
  int i, j;
  const float pi = 2.0f * asinf(1.0f);
  memset(in, 0, n * m * sizeof(float));
  
  for (i = 0; i < m; i++) {
    in[i] = 0.f;
    in[(n - 1) * m + i] = 0.f;
  }
  
  for (j = 0; j < n; j++) {
    in[j * m] = sinf(pi * j / (n - 1));
    in[j * m + m - 1] = sinf(pi * j / (n - 1)) * expf(-pi);
  }
}

int main(int argc, char **argv) {
  int n = 4096, m = 4096;
  int iter_max = 100, THREADS_BLOCK = 256;  // Larger block size for better occupancy
  float *A;

  const float tol = 1.0e-8f;
  float error = 1.0f;

  if (argc > 1) n = atoi(argv[1]);
  if (argc > 2) m = atoi(argv[2]);
  if (argc > 3) iter_max = atoi(argv[3]);
  if (argc > 4) THREADS_BLOCK = atoi(argv[4]);

  A = (float *)malloc(n * m * sizeof(float));
  laplace_init(A, n, m);
  A[(n / 128) * m + m / 128] = 1.0f;

  printf("Jacobi relaxation Calculation: %d x %d mesh, "
         "maximum of %d iterations. Threads per block= %d\n",
         n, m, iter_max, THREADS_BLOCK);

  // GPU memory allocation
  float *A_dev, *Anew_dev, *error_dev;
  cudaMalloc(&A_dev, n * m * sizeof(float));
  cudaMalloc(&Anew_dev, n * m * sizeof(float));
  cudaMalloc(&error_dev, sizeof(float));

  // Single transfer to GPU
  cudaMemcpy(A_dev, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Anew_dev, A, n * m * sizeof(float), cudaMemcpyHostToDevice);

  // Grid configuration: 1D grid for better flexibility
  int total_points = (n-2) * (m-2);
  int num_blocks = (total_points + THREADS_BLOCK - 1) / THREADS_BLOCK;
  // Limit to reasonable number of blocks
  num_blocks = min(num_blocks, 2048);
  
  dim3 gridDim(num_blocks, 1, 1);
  dim3 blockDim(THREADS_BLOCK, 1, 1);

  printf("Grid: %d blocks, Block: %d threads, Total threads: %d\n", 
         num_blocks, THREADS_BLOCK, num_blocks * THREADS_BLOCK);

  // Aggressive batching to minimize CPU-GPU sync
  int check_interval = 50;  // Check convergence every 50 iterations
  int iter = 0;
  
  printf("Using batched convergence checking (every %d iterations)\n", check_interval);

  while (iter < iter_max) {
    // Determine batch size
    int remaining = iter_max - iter;
    int batch_size = (remaining < check_interval) ? remaining : check_interval;
    int batch_end = iter + batch_size;
    
    // Run batch of iterations without error checking
    for (int i = iter; i < batch_end - 1; i++) {
      dev_laplace_error_v3<<<gridDim, blockDim>>>(A_dev, Anew_dev, error_dev, n, m, 0);
      
      // Swap arrays
      float *temp = A_dev;
      A_dev = Anew_dev;
      Anew_dev = temp;
    }
    
    // Final iteration of batch WITH error checking
    cudaMemset(error_dev, 0, sizeof(float));
    dev_laplace_error_v3<<<gridDim, blockDim>>>(A_dev, Anew_dev, error_dev, n, m, 1);
    
    // Only NOW do we transfer error back to CPU
    cudaMemcpy(&error, error_dev, sizeof(float), cudaMemcpyDeviceToHost);
    error = sqrtf(error);
    
    float *temp = A_dev;
    A_dev = Anew_dev;
    Anew_dev = temp;
    
    iter = batch_end;
    printf("%5d, %0.6f\n", iter, error);
    
    // Check convergence
    if (error <= tol) {
      printf("Converged at iteration %d\n", iter);
      break;
    }
  }

  // Final result transfer
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