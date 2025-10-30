// OpenMP parallelized Laplace solver - Version 2
// Optimized to reduce OpenMP overhead by using single parallel region
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define PI (3.1415926535897932384626)

int main(int argc, char** argv) {
  int i, j, iter = 0;

  // Initialize with defaults
  int n = 100, m = 100;
  if (argc > 1) n = atoi(argv[1]);
  if (argc > 2) m = atoi(argv[2]);

  // The data is dynamically allocated
  float *A[n], *Anew[n];
  for (i = 0; i < n; i++) {
    A[i] = (float*)malloc(m * sizeof(float));
    Anew[i] = (float*)malloc(m * sizeof(float));
  }

  // All the interior points in the 2D matrix are zero
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) A[i][j] = 0;

  // set boundary conditions
  for (j = 0; j < m; j++) {
    A[0][j] = 0.f;
    A[n - 1][j] = 0.f;
  }
  for (i = 0; i < n; i++) {
    A[i][0] = sinf(PI * i / (n - 1));
    A[i][m - 1] = sinf(PI * i / (n - 1)) * expf(-PI);
  }

  const float tol = 1.0e-3f;
  float error = 1.0f;
  int iter_max = 100;
  
  double start_time = omp_get_wtime();
  
  // OPTIMIZATION: Single parallel region for entire main loop
  // This eliminates the overhead of creating/destroying parallel regions
  #pragma omp parallel
  {
    float local_error;
    
    while (error > tol && iter < iter_max) {
      
      // Stencil computation - work sharing without parallel overhead
      #pragma omp for collapse(2) nowait
      for (i = 1; i < n - 1; i++)
        for (j = 1; j < m - 1; j++)
          Anew[i][j] = (A[i][j + 1] + A[i][j - 1] + A[i - 1][j] + A[i + 1][j]) / 4;

      // Barrier to ensure stencil computation is complete before error calculation
      #pragma omp barrier
      
      // Reset error for this iteration
      #pragma omp single
      error = 0.0f;
      
      // Error calculation with proper reduction
      local_error = 0.0f;
      #pragma omp for collapse(2)
      for (i = 1; i < n - 1; i++)
        for (j = 1; j < m - 1; j++) {
          float diff = sqrtf(fabsf(Anew[i][j] - A[i][j]));
          if (diff > local_error) local_error = diff;
        }
      
      // Global reduction - find maximum across all threads
      #pragma omp critical
      {
        if (local_error > error) error = local_error;
      }
      
      // Wait for all threads to complete reduction
      #pragma omp barrier
      
      // Matrix update
      #pragma omp for collapse(2) nowait
      for (i = 1; i < n - 1; i++)
        for (j = 1; j < m - 1; j++) A[i][j] = Anew[i][j];

      // Only master thread handles iteration counting and printing
      #pragma omp master
      {
        iter++;
        if (iter % 10 == 0) printf("%5d, %0.6f\n", iter, error);
      }
      
      // Synchronize before next iteration
      #pragma omp barrier
    }
  } // End of parallel region
  
  double end_time = omp_get_wtime();
  printf("Execution time: %.6f seconds\n", end_time - start_time);
  printf("Threads used: %d\n", omp_get_max_threads());
  
  // Cleanup
  for (i = 0; i < n; i++) {
    free(A[i]);
    free(Anew[i]);
  }
  
  return 0;
}