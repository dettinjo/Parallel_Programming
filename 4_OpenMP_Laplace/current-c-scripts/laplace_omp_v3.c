// OpenMP parallelized Laplace solver - Version 3
// Flattened array with pointer swapping and optimizations
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define PI (3.1415926535897932384626)
#define INDEX(i, j, M) ((i) * (M) + (j))

int main(int argc, char** argv) {
  int i, j, iter = 0;

  // Initialize with defaults
  int n = 100, m = 100;
  if (argc > 1) n = atoi(argv[1]);
  if (argc > 2) m = atoi(argv[2]);

  // OPTIMIZATION 1: Flattened arrays for better memory locality
  // Single contiguous allocation instead of array of pointers
  float * restrict A = (float*)malloc(n * m * sizeof(float));
  float * restrict Anew = (float*)malloc(n * m * sizeof(float));
  
  if (A == NULL || Anew == NULL) {
    printf("Error: Memory allocation failed!\n");
    if (A) free(A);
    if (Anew) free(Anew);
    return 1;
  }

  // Initialize all interior points to zero
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) 
      A[INDEX(i, j, m)] = 0.0f;

  // Set boundary conditions
  for (j = 0; j < m; j++) {
    A[INDEX(0, j, m)] = 0.0f;
    A[INDEX(n-1, j, m)] = 0.0f;
  }
  for (i = 0; i < n; i++) {
    A[INDEX(i, 0, m)] = sinf(PI * i / (n - 1));
    A[INDEX(i, m-1, m)] = sinf(PI * i / (n - 1)) * expf(-PI);
  }

  const float tol = 1.0e-3f;
  float error = 1.0f;
  int iter_max = 100;
  
  double start_time = omp_get_wtime();
  
  // OPTIMIZATION 2: Single parallel region (from v2) + better synchronization
  #pragma omp parallel
  {
    float local_error;
    
    while (error > tol && iter < iter_max) {
      
      // Stencil computation using flattened array indexing
      #pragma omp for collapse(2) nowait
      for (i = 1; i < n - 1; i++) {
        for (j = 1; j < m - 1; j++) {
          Anew[INDEX(i, j, m)] = (A[INDEX(i, j+1, m)] + A[INDEX(i, j-1, m)] + 
                                  A[INDEX(i-1, j, m)] + A[INDEX(i+1, j, m)]) * 0.25f;
        }
      }

      // Ensure stencil computation is complete
      #pragma omp barrier
      
      // Reset error for this iteration
      #pragma omp single
      error = 0.0f;
      
      // Error calculation with local reduction
      local_error = 0.0f;
      #pragma omp for collapse(2)
      for (i = 1; i < n - 1; i++) {
        for (j = 1; j < m - 1; j++) {
          float diff = sqrtf(fabsf(Anew[INDEX(i, j, m)] - A[INDEX(i, j, m)]));
          if (diff > local_error) local_error = diff;
        }
      }
      
      // Global reduction
      #pragma omp critical
      {
        if (local_error > error) error = local_error;
      }
      
      // Wait for all threads to complete error calculation
      #pragma omp barrier

      // OPTIMIZATION 3: Pointer swapping (works with flattened arrays)
      // Eliminates the need for matrix copy entirely
      #pragma omp single
      {
        float *temp = A;
        A = Anew;
        Anew = temp;
        
        // Reapply boundary conditions to swapped array
        for (int jj = 0; jj < m; jj++) {
          A[INDEX(0, jj, m)] = 0.0f;
          A[INDEX(n-1, jj, m)] = 0.0f;
        }
        for (int ii = 0; ii < n; ii++) {
          A[INDEX(ii, 0, m)] = sinf(PI * ii / (n - 1));
          A[INDEX(ii, m-1, m)] = sinf(PI * ii / (n - 1)) * expf(-PI);
        }
      }

      // Master thread handles iteration counting and printing
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
  free(A);
  free(Anew);
  
  return 0;
}