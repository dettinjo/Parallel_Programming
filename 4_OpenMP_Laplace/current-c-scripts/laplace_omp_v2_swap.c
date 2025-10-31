// OpenMP parallelized Laplace solver - Version 2 with pointer swapping
// Eliminates matrix copy overhead by swapping pointers
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
  
  while (error > tol && iter < iter_max) {
    
    // Stencil computation - compute Anew from A
    #pragma omp parallel for collapse(2)
    for (i = 1; i < n - 1; i++) {
      for (j = 1; j < m - 1; j++) {
        Anew[i][j] = (A[i][j + 1] + A[i][j - 1] + A[i - 1][j] + A[i + 1][j]) / 4;
      }
    }
    
    // Error calculation comparing Anew to A
    error = 0.0f;
    #pragma omp parallel for collapse(2) reduction(max:error)
    for (i = 1; i < n - 1; i++) {
      for (j = 1; j < m - 1; j++) {
        error = fmaxf(error, sqrtf(fabsf(Anew[i][j] - A[i][j])));
      }
    }

    // OPTIMIZATION: Swap pointers instead of copying data
    // This eliminates the need for the third parallel loop
    float **temp = A;
    A = Anew;
    Anew = temp;
    
    // Copy boundary conditions to the new A matrix (swapped)
    for (j = 0; j < m; j++) {
      A[0][j] = 0.f;
      A[n - 1][j] = 0.f;
    }
    for (i = 0; i < n; i++) {
      A[i][0] = sinf(PI * i / (n - 1));
      A[i][m - 1] = sinf(PI * i / (n - 1)) * expf(-PI);
    }

    iter++;
    if (iter % 10 == 0) printf("%5d, %0.6f\n", iter, error);
  }
  
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