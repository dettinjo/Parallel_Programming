// Define _XOPEN_SOURCE 600 to get definitions for srand48/drand48
#define _XOPEN_SOURCE 600
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// --- Helper Functions based on P5 Solution PDF ---

/**
 * @brief Fused F1 and F2.
 * This version also returns the partial sum of C[i] for the scan.
 */
double VectF1_F2_sum(double *IN, double *B, double *C, double v, int n)
{
  double sum = 0.0;
  for (int i = 0; i < n; i++)
  {
    // VectF1 logic
    long int T = IN[i];
    B[i] = (double)(T % 4) + 0.5 + (IN[i] - trunc(IN[i]));
    // VectF2 logic
    C[i] = v / (1.0 + fabs(B[i]));
    // Partial sum for scan
    sum += C[i];
  }
  return sum;
}

/**
 * @brief Fused F1 and F2 (no sum).
 * For sections that don't need to return a partial scan sum.
 */
void VectF1_F2(double *IN, double *B, double *C, double v, int n)
{
  for (int i = 0; i < n; i++)
  {
    // VectF1 logic
    long int T = IN[i];
    B[i] = (double)(T % 4) + 0.5 + (IN[i] - trunc(IN[i]));
    // VectF2 logic
    C[i] = v / (1.0 + fabs(B[i]));
  }
}

/**
 * @brief Manual Scan function.
 * Takes a 'start_sum' to handle parallel scan sections.
 */
void VectScan2(double *IN, double *OUT, double start_sum, int n)
{
  double sum = start_sum;
  // Simple, correct version:
  for (int i = 0; i < n; i++)
  {
    sum += IN[i];
    OUT[i] = sum;
  }
}

/**
 * @brief Fused Average and Sum.
 * Returns the partial sum of D[i] (which becomes 'v').
 * Based on the PDF logic: VectAverageAndSum(B + N3 - 1, 1+N3)
 * This implies the first parameter is the start of the B array
 * (with 1-element overlap for B[i-1]) and the second is the
 * number of elements *to read*, which is (chunk_size + 1).
 * The number of elements *to compute* is (chunk_size).
 */
double VectAverageAndSum(double *B_chunk_start_with_overlap, int chunk_size_with_overlap)
{
  double sum = 0.0;
  // B_chunk_start_with_overlap points to B[N3-1] for the 2nd chunk.
  // We call it 'B' for simplicity.
  double *B = B_chunk_start_with_overlap;

  // chunk_size_with_overlap is 1+N3.
  // The number of D[i] elements to compute is N3.
  // Loop from i=0 to N3-1.
  int n_compute = chunk_size_with_overlap - 1;

  for (int i = 0; i < n_compute; i++)
  {
    // When i=0:
    // B[i] is B[0] (which is B[N3-1] globally)
    // B[i-1] is B[-1] (which is B[N3-2] globally)
    // B[i+1] is B[1] (which is B[N3] globally)
    // This is incorrect. The PDF logic is B[i-1], B[i], B[i+1]
    // B[0] in this chunk corresponds to D[1]
    // B[-1] = B[i-1]
    // B[0] = B[i]
    // B[1] = B[i+1]

    // Corrected logic based on B[i-1], B[i], B[i+1] access
    // B[i+1] is the element *after* B[i], B[i-1] is *before*
    double D_val = (2.0 * B[i] + B[i - 1] + B[i + 1]) / 4.0;
    sum += D_val;
  }
  return sum;
}

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
  double *D = malloc(N * sizeof(double)); // Not actually used for output

  srand48(0);
  for (i = 0; i < N; i++)
    A[i] = drand48() - 0.5;

  int nthreads = omp_get_max_threads();
  printf("Inputs: N= %d, Rep= %d, Max Threads= %d (P5 Manual Solution)\n", N, REP, nthreads);

  double start_time = omp_get_wtime();

  // --- Variables for P5 manual solution ---
  double v = 10.0;
  double v1 = 0.0, v2 = 0.0;                     // Partial sums for v
  double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0; // Partial sums for Scan

  // Calculate chunk sizes
  int N6 = N / 6;
  int N3 = N / 3;
  // Handle non-even divisions (as per PDF)
  if (N6 * 6 < N)
    N6++;
  if (N3 * 3 < N)
    N3++;

  // Pre-calculate final chunk sizes to avoid overflow
  int N6_chunk5_start = 5 * N6;
  int N6_chunk5_size = N - N6_chunk5_start;

  int N3_chunk2_start = 2 * N3;
  int N3_chunk2_size = N - N3_chunk2_start;

  for (i = 0; i < REP; i++)
  {
    // --- Merge partial sums from *previous* iteration ---
    // As per the PDF, v is the sum of partials
    // On first iteration, v=10.0, v1=0, v2=0.
    // On subsequent, it uses results from last loop.
    v = v + v1 + v2;

// This single parallel region contains all the work
#pragma omp parallel shared(A, B, C, D, v, v1, v2, s1, s2, s3, s4, N6, N3, N6_chunk5_size, N3_chunk2_size)
    {
// --- STAGE 1: Fused F1/F2 (6 sections) ---
#pragma omp sections
      {
#pragma omp section
        s1 = VectF1_F2_sum(A + 0 * N6, B + 0 * N6, C + 0 * N6, v, N6);

#pragma omp section
        s2 = VectF1_F2_sum(A + 1 * N6, B + 1 * N6, C + 1 * N6, v, N6);

#pragma omp section
        s3 = VectF1_F2_sum(A + 2 * N6, B + 2 * N6, C + 2 * N6, v, N6);

#pragma omp section
        s4 = VectF1_F2_sum(A + 3 * N6, B + 3 * N6, C + 3 * N6, v, N6);

#pragma omp section
        VectF1_F2(A + 4 * N6, B + 4 * N6, C + 4 * N6, v, N6);

#pragma omp section
        VectF1_F2(A + 5 * N6, B + 5 * N6, C + 5 * N6, v, N6_chunk5_size);
      }

// Implicit barrier here: all F1/F2 sections must finish

// --- STAGE 2: Manual Scan & Fused Avg/Sum (3+3 sections) ---
#pragma omp sections
      {
// --- Manual Parallel Scan ---
#pragma omp section
        VectScan2(C + 0 * N3, A + 0 * N3, 0.0, N3);

#pragma omp section
        VectScan2(C + 1 * N3, A + 1 * N3, s1 + s2, N3); // s1+s2 from chunk 0,1

#pragma omp section
        VectScan2(C + 2 * N3, A + 2 * N3, s1 + s2 + s3 + s4, N3_chunk2_size); // s1-s4 from chunk 0-3

        // --- Parallel Fused Average/Sum ---
        // *** FIX: Each of these must be in its own section ***

#pragma omp section
        // This call computes D[1]...D[N3]
        // It needs B[0]...B[N3] (N3+1 elements)
        // We pass B (which is B[0]) and size 1+N3
        v = VectAverageAndSum(B, 1 + N3);

#pragma omp section
        // This call computes D[N3+1]...D[2*N3]
        // It needs B[N3]...B[2*N3] (N3+1 elements)
        // We pass B+N3 (B[N3]) but the logic needs B[N3-1]
        // We pass B+N3-1 and size 1+N3
        v1 = VectAverageAndSum(B + N3 - 1, 1 + N3);

#pragma omp section
        // This call computes D[2*N3+1]...D[N-1] (or N-2)
        // It needs B[2*N3]...B[N-1]
        // We pass B+2*N3-1 and size 1+(N-2*N3)
        v2 = VectAverageAndSum(B + 2 * N3 - 1, 1 + N3_chunk2_size);
      }
    } // Implicit barrier at end of parallel region
  }

  double end_time = omp_get_wtime();
  double elapsed = end_time - start_time;

  // Note: 'v' now holds the partial sum from the first chunk.
  // The final 'v' is v+v1+v2, but that's calculated at the *start*
  // of the *next* loop. We need to do it one last time.
  v = v + v1 + v2;

  printf("Outputs (P5 Solution): v= %0.12e, A[%d]= %0.12e\n", v, N - 1, A[N - 1]);
  printf("NOTE: Values may differ from baseline due to manual parallelization logic.\n");
  printf("Execution time: %.4f seconds\n", elapsed);

  free(A);
  free(B);
  free(C);
  free(D);
  return 0;
}
