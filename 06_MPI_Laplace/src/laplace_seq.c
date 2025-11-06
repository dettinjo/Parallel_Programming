// Solve warnings of implicit declaration
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h> // <<< ADD THIS

#define PI (3.1415926535897932384626)

int main(int argc, char** argv) {
  int i, j, iter = 0;

  // <<< ADD MPI_Init
  MPI_Init(&argc, &argv);

  // N and M are provided at runtime
  int n, m;
  // get runtime arguments
  if (argc > 1) n = atoi(argv[1]);
  else n = 100; // <<< Add default value
  if (argc > 2) m = atoi(argv[2]);
  else m = 100; // <<< Add default value


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

  // <<< START TIMER
  double start_time = MPI_Wtime();

  // Main loop: iterate until error <= tol a maximum of iter_max iterations
  while (error > tol && iter < iter_max) {
    // Calculate the new value
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < m - 1; j++)
        Anew[i][j] =
            (A[i][j + 1] + A[i][j - 1] + A[i - 1][j] + A[i + 1][j]) / 4;

    // Compute error
    error = 0.0f;
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < m - 1; j++)
        error = fmaxf(error, sqrtf(fabsf(Anew[i][j] - A[i][j])));

    // Update A with Anew
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < m - 1; j++) A[i][j] = Anew[i][j];

    iter++;
    
    // <<< REMOVE THIS PRINTF
    // if (iter % 10 == 0) printf("%5d, %0.6f\n", iter, error);
    
  }  // while

  // <<< STOP TIMER
  double end_time = MPI_Wtime();

  // <<< PRINT ONLY THE FINAL TIME
  printf("%f\n", end_time - start_time);

  // <<< ADD MPI_Finalize
  MPI_Finalize();
  
  // Free memory
  for (i = 0; i < n; i++) {
    free(A[i]);
    free(Anew[i]);
  }

  return 0;
}