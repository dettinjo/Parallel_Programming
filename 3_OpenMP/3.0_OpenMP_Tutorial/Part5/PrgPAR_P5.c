#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// --- ALL 5 FUNCTIONS GO HERE ---
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
    OUT[i] = sum;
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
  double v_snapshot = v; // This will be handled by the task's scope

  double start_time = omp_get_wtime();

// === SINGLE FORK/JOIN ===
// Create the 12-thread team ONCE.
#pragma omp parallel shared(A, B, C, D, v, N, REP, i)
  {
// === TASK PRODUCER ===
// One thread loops 250,000 times, creating all tasks
#pragma omp single
    {
      for (i = 0; i < REP; i++)
      {
// We create 5 tasks that represent the 5 functions.
// We use 'depend' clauses to enforce the *exact*
// dependency graph. This eliminates all barriers.

// The 'A', 'B', 'C', 'D', 'v' variables act as
// "dependency tokens".

// TASK 1: F1 (A -> B)
// Must wait for 'A' (from Scan) and 'v' (from Sum)
// from the *previous* iteration.
// Will write to 'B'.
#pragma omp task depend(in : A, v) depend(out : B)
        {
          VectF1(A, B, N);
        }

// TASK 2: Average (B -> D)
// Must wait for 'B' (from F1).
// Will write to 'D'.
#pragma omp task depend(in : B) depend(out : D)
        {
          VectAverage(B, D, N);
        }

// TASK 3: F2 (B, v -> C)
// Must wait for 'B' (from F1) and 'v' (from Sum).
// Will write to 'C'.
#pragma omp task depend(in : B, v) depend(out : C)
        {
          // We must snapshot 'v' *inside* this task
          // to get the correctly-depended-on value.
          double v_snapshot = v;
          VectF2(B, C, v_snapshot, N);
        }

// TASK 4: Scan (C -> A)
// Must wait for 'C' (from F2).
// Will write to 'A' (which F1 in the next iter needs).
#pragma omp task depend(in : C) depend(out : A)
        {
          VectScan(C, A, N);
        }

// TASK 5: Sum (D -> v)
// Must wait for 'D' (from Average).
// Will write to 'v' (which F1/F2 in the next iter need).
#pragma omp task depend(in : D) depend(out : v)
        {
          v = VectSum(D, N);
        }
      } // End of REP loop. Producer has created 1.25M tasks.
    } // End of 'single' producer block.
  } // End of 'parallel' region. All threads wait here.

  double end_time = omp_get_wtime();
  printf("Execution time: %f seconds\n", end_time - start_time);
  printf("Outputs: v= %0.12e, A[%d]= %0.12e\n", v, N - 1, A[N - 1]);

  free(A);
  free(B);
  free(C);
  free(D);
}