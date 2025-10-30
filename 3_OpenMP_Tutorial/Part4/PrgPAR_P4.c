#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

int main(int argc, char **argv)
{
    int i, N = 20000, REP = 250000;

    // Get program arguments at runtime
    if (argc > 1)
    {
        N = atoi(argv[1]);
    }
    if (argc > 2)
    {
        REP = atoi(argv[2]);
    }

    // Allocate memory space for arrays
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double *C = malloc(N * sizeof(double));
    double *D = malloc(N * sizeof(double));

    //  set initial values
    srand48(0);
    for (i = 0; i < N; i++)
        A[i] = drand48() - 0.5f; // values between -0.5 and 0.5

    printf("Inputs: N= %d, Rep= %d\n", N, REP);

    double v = 10.0;

    // =================================================================
    // === THIS IS THE NEW, CORRECT, AND OPTIMIZED LOOP FOR PART 4 ===
    // =================================================================
    for (i = 0; i < REP; i++)
    {
#pragma omp parallel shared(A, B, C, D, v)
        {
// --- P4 Optimization: Parallelize the main bottleneck ---
// The work of VectF1 is split into two parallel sections.
// An implicit barrier at the end ensures all of array B is ready.
#pragma omp sections
            {
#pragma omp section
                VectF1(A, B, N / 2); // Thread 0 works on the first half

#pragma omp section
                VectF1(A + N / 2, B + N / 2, N / 2); // Thread 1 works on the second half
            }

// --- Correct Dependency Handling ---
// VectF2 and VectAverage are independent and can run in parallel.
// An implicit barrier at the end ensures C and D are ready.
#pragma omp sections
            {
#pragma omp section
                VectF2(B, C, v, N);

#pragma omp section
                VectAverage(B, D, N);
            }

// --- Final Sequential Part ---
// VectScan and VectSum must run on a single thread after the others finish.
#pragma omp single
            {
                VectScan(C, A, N);
                v = VectSum(D, N);
            }
        } // Implicit barrier at the end of the parallel region.
    }
    // =================================================================

    printf("Outputs: v= %0.12e, A[%d]= %0.12e\n", v, N - 1, A[N - 1]);

    // Free memory space for arrays
    free(A);
    free(B);
    free(C);
    free(D);
}
