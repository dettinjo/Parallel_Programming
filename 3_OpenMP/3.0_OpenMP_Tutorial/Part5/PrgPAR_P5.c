#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // Make sure to include this

// --- INSERT ALL 5 FUNCTIONS HERE ---
// (VectF1, VectF2, VectScan, VectAverage, VectSum)
// (Identical to 01_Prg.c)
void VectF1(double *IN, double *OUT, int n)
{
  for (int i = 0; i < n; i++)
  {
    long int T = IN[i];
    OUT[i] = (double)(T % 4) + 0.5 + (IN[i] - trunc(IN[i]));
  }
}

void VectF2(double *IN, double *OUT, double v, int n)
{
  for (int i = 0; i < n; i++)
    OUT[i] = v / (1.0 + fabs(IN[i]));
}

void VectScan(double *IN, double *OUT, int n)
{
  double sum = 0.0;
  for (int i = 0; i < n; i++)
  {
    sum += IN[i];
    OUT[i] = sum; // Inclusive: include current element
  }
}

void VectAverage(double *IN, double *OUT, int n)
{
  for (int i = 1; i < n - 1; i++)
  {
    OUT[i] = (2.0 * IN[i] + IN[i - 1] + IN[i + 1]) / 4.0;
  }
}

double VectSum(double *V, int n)
{
  double sum = 0;
  for (int i = 0; i < n; i++)
    sum = sum + V[i];
  return sum;
}
// --- END OF FUNCTIONS ---

int main(int argc, char **argv)
{
  int i, N = 20000, REP = 250000;

  if (argc > 1)
  {
    N = atoi(argv[1]);
  }
  if (argc > 2)
  {
    REP = atoi(argv[2]);
  }

  double *A = malloc(N * sizeof(double));
  double *B = malloc(N * sizeof(double));
  double *C = malloc(N * sizeof(double));
  double *D = malloc(N * sizeof(double));

  srand48(0);
  for (i = 0; i < N; i++)
    A[i] = drand48() - 0.5f;

  printf("Inputs: N= %d, Rep= %d\n", N, REP);

  double v = 10.0;
  // This variable is now safe because there is no "race-ahead" thread.
  double v_snapshot = v;

  double start_time = omp_get_wtime();

// Create the thread team ONCE.
// The 'i' loop counter must be private to each thread's stack.
#pragma omp parallel private(i) shared(A, B, C, D, v, v_snapshot, N, REP)
  {
    // All threads in the team (just 2) will execute this loop.
    for (i = 0; i < REP; i++)
    {
// 1. One thread (T0) executes VectF1, the other (T1) waits.
#pragma omp single
      {
        VectF1(A, B, N);
        // T0 reads 'v' from the previous iteration
        v_snapshot = v;
      }
// Implicit barrier: T0 and T1 sync. VectF1 is done.

// 2. Both threads (T0, T1) hit this.
#pragma omp sections
      {
#pragma omp section
        {
          // T1 executes the slow path, reading the safe v_snapshot
          VectF2(B, C, v_snapshot, N);
          VectScan(C, A, N);
        }
#pragma omp section
        {
          // T0 executes the fast path
          VectAverage(B, D, N);
          v = VectSum(D, N);
        }
      }
      // Implicit barrier: T0 waits for T1. Both branches are done.
    } // End of REP loop. T0 and T1 go to the next iteration.
  } // Implicit barrier: Parallel region ends, team is destroyed.

  double end_time = omp_get_wtime();
  printf("Execution time: %f seconds\n", end_time - start_time);
  printf("Outputs: v= %0.12e, A[%d]= %0.12e\n", v, N - 1, A[N - 1]);

  free(A);
  free(B);
  free(C);
  free(D);
}