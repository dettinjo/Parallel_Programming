#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ float atomicMax(float *addr, float val) {
  unsigned int old = __float_as_uint(*addr), assumed;
  do {
    assumed = old;
    if (__uint_as_float(old) >= val) break;
    old = atomicCAS((unsigned int *)addr, assumed, __float_as_uint(val));
  } while (assumed != old);
  return __uint_as_float(old);
}

// Correct single-iteration kernel maintaining Jacobi semantics
__global__ void dev_laplace_single(float *A, float *Anew, float *error_dev, 
                                   int n, int m, int compute_error) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_points = (n-2) * (m-2);
  
  if (tid >= total_points) return;
  
  int j = (tid / (m-2)) + 1;
  int i = (tid % (m-2)) + 1;
  int idx = j * m + i;
  
  // 5-point stencil using current A values
  float new_val = 0.25f * (A[idx-1] + A[idx+1] + A[idx-m] + A[idx+m]);
  Anew[idx] = new_val;
  
  // Compute error if needed
  if (compute_error) {
    float local_error = fabsf(A[idx] - new_val);
    if (local_error > 0.0f) {
      atomicMax(error_dev, local_error);
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
  int iter_max = 100, THREADS_BLOCK = 256;
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

  float *A_dev, *Anew_dev, *error_dev;
  cudaMalloc(&A_dev, n * m * sizeof(float));
  cudaMalloc(&Anew_dev, n * m * sizeof(float));
  cudaMalloc(&error_dev, sizeof(float));

  cudaMemcpy(A_dev, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Anew_dev, A, n * m * sizeof(float), cudaMemcpyHostToDevice);

  int total_points = (n-2) * (m-2);
  int num_blocks = (total_points + THREADS_BLOCK - 1) / THREADS_BLOCK;
  num_blocks = min(num_blocks, 2048);
  
  printf("Grid: %d blocks × %d threads\n", num_blocks, THREADS_BLOCK);

  // Batching approach: run many iterations between error checks
  int check_interval = 50;
  int iter = 0;

  while (iter < iter_max) {
    int remaining = iter_max - iter;
    int batch_size = (remaining < check_interval) ? remaining : check_interval;
    
    // Run batch_size iterations
    for (int i = 0; i < batch_size; i++) {
      if (i == batch_size - 1) {
        // Last iteration: compute error
        cudaMemset(error_dev, 0, sizeof(float));
        dev_laplace_single<<<num_blocks, THREADS_BLOCK>>>(A_dev, Anew_dev, error_dev, n, m, 1);
      } else {
        // Regular iteration: no error
        dev_laplace_single<<<num_blocks, THREADS_BLOCK>>>(A_dev, Anew_dev, error_dev, n, m, 0);
      }
      
      // Swap arrays (maintains Jacobi semantics)
      float *temp = A_dev;
      A_dev = Anew_dev;
      Anew_dev = temp;
    }
    
    // Check convergence
    cudaMemcpy(&error, error_dev, sizeof(float), cudaMemcpyDeviceToHost);
    error = sqrtf(error);
    
    iter += batch_size;
    printf("%5d, %0.6f\n", iter, error);
    
    if (error <= tol) {
      printf("Converged at iteration %d\n", iter);
      break;
    }
  }

  cudaMemcpy(A, A_dev, n * m * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
  printf("A[%d][%d]= %0.6f\n", n / 128, m / 128, A[(n / 128) * m + m / 128]);

  cudaFree(A_dev);
  cudaFree(Anew_dev);
  cudaFree(error_dev);
  free(A);

  return 0;
}