#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define PI (3.1415926535897932384626)

int main(int argc, char** argv) {
  /* Declare ALL variables at the top (C89 style) */
  int i, j, iter = 0;
  int n, m;
  float **A, **Anew;
  const float tol = 1.0e-3f;
  float error = 1.0f;
  int iter_max = 100;
  double start_time, end_time;
  int rank, size;

  /* MPI setup */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Get runtime arguments */
  if (argc > 1) n = atoi(argv[1]);
  else n = 100;
  if (argc > 2) m = atoi(argv[2]);
  else m = 100;

  /* Allocate memory exactly like sequential version */
  A = malloc(n * sizeof(float*));
  Anew = malloc(n * sizeof(float*));
  for (i = 0; i < n; i++) {
    A[i] = (float*)malloc(m * sizeof(float));
    Anew[i] = (float*)malloc(m * sizeof(float));
  }

  /* Initialize exactly like sequential version */
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) A[i][j] = 0;

  /* Set boundary conditions exactly like sequential version */
  for (j = 0; j < m; j++) {
    A[0][j] = 0.f;
    A[n - 1][j] = 0.f;
  }
  for (i = 0; i < n; i++) {
    A[i][0] = sinf(PI * i / (n - 1));
    A[i][m - 1] = sinf(PI * i / (n - 1)) * expf(-PI);
  }

  /* START TIMER */
  start_time = MPI_Wtime();

  /* Main loop exactly like sequential version */
  while (error > tol && iter < iter_max) {
    /* Calculate the new value */
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < m - 1; j++)
        Anew[i][j] =
            (A[i][j + 1] + A[i][j - 1] + A[i - 1][j] + A[i + 1][j]) / 4;

    /* Compute error */
    error = 0.0f;
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < m - 1; j++)
        error = fmaxf(error, sqrtf(fabsf(Anew[i][j] - A[i][j])));

    /* Update A with Anew */
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < m - 1; j++) A[i][j] = Anew[i][j];

    iter++;
  }

  /* STOP TIMER */
  end_time = MPI_Wtime();

  /* Output EXACTLY like sequential version - only rank 0 prints */
  if (rank == 0) {
    printf("%f\n", end_time - start_time);
  }

  /* Free memory */
  for (i = 0; i < n; i++) {
    free(A[i]);
    free(Anew[i]);
  }
  free(A);
  free(Anew);

  MPI_Finalize();
  return 0;
}