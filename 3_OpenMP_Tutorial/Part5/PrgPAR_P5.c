#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv)
{
  int i, N = 20000, REP = 250000;

  if (argc > 1) { N = atoi(argv[1]); }
  if (argc > 2) { REP = atoi(argv[2]); }

  double *A = malloc(N * sizeof(double));
  double *B = malloc(N * sizeof(double));
  double *C = malloc(N * sizeof(double));
  double *D = malloc(N * sizeof(double));

  srand48(0);
  for (i = 0; i < N; i++)
    A[i] = drand48() - 0.5f;

  int num_threads = omp_get_max_threads();
  printf("Inputs: N= %d, Rep= %d, Threads= %d\n", N, REP, num_threads);

  double start_time = omp_get_wtime();
  double v = 10.0;

  // Single parallel region for minimal overhead
  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    
    // Pre-calculate thread workload distribution
    int chunk_f1 = N / nthreads;
    int start_f1 = tid * chunk_f1;
    int end_f1 = (tid == nthreads - 1) ? N : start_f1 + chunk_f1;
    
    // For VectF2 and VectAverage: use half threads each
    int half = nthreads / 2;
    int is_f2_thread = (tid < half);
    int is_avg_thread = (tid >= half);
    
    // VectF2 work distribution
    int chunk_f2 = N / half;
    int start_f2 = tid * chunk_f2;
    int end_f2 = (tid == half - 1) ? N : start_f2 + chunk_f2;
    
    // VectAverage work distribution
    int tid_avg = tid - half;
    int work_avg = N - 2;
    int chunk_avg = work_avg / (nthreads - half);
    int start_avg = 1 + tid_avg * chunk_avg;
    int end_avg = (tid == nthreads - 1) ? N - 1 : start_avg + chunk_avg;
    
    // VectSum partial sums
    double partial_sum = 0.0;
    
    for (i = 0; i < REP; i++) {
      
      // ===== STAGE 1: VectF1 (all threads) =====
      for (int j = start_f1; j < end_f1; j++) {
        long int T = A[j];
        B[j] = (double)(T % 4) + 0.5 + (A[j] - trunc(A[j]));
      }
      
      #pragma omp barrier
      
      // ===== STAGE 2a: VectF2 (first half of threads) =====
      if (is_f2_thread) {
        for (int j = start_f2; j < end_f2; j++) {
          C[j] = v / (1.0 + fabs(B[j]));
        }
      }
      
      // ===== STAGE 2b: VectAverage (second half of threads) =====
      if (is_avg_thread) {
        for (int j = start_avg; j < end_avg; j++) {
          D[j] = (2.0 * B[j] + B[j - 1] + B[j + 1]) / 4.0;
        }
      }
      
      #pragma omp barrier
      
      // ===== STAGE 3a: VectScan (sequential - single thread) =====
      #pragma omp single nowait
      {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
          sum += C[j];
          A[j] = sum;
        }
      }
      
      // ===== STAGE 3b: VectSum (parallel with manual reduction) =====
      // While VectScan runs, compute partial sums
      partial_sum = 0.0;
      for (int j = start_f1; j < end_f1; j++) {
        partial_sum += D[j];
      }
      
      #pragma omp barrier
      
      // Reduce partial sums
      #pragma omp single
      {
        v = 0.0;
      }
      
      #pragma omp critical
      {
        v += partial_sum;
      }
      
      #pragma omp barrier
    }
  }

  double end_time = omp_get_wtime();
  printf("Outputs: v= %0.12e, A[%d]= %0.12e\n", v, N - 1, A[N - 1]);
  printf("Execution time: %.4f seconds\n", end_time - start_time);
  
  printf("\nPerformance Analysis:\n");
  printf("- VectF1: Parallelized across all threads\n");
  printf("- VectF2 & VectAverage: Run simultaneously on different thread groups\n");
  printf("- VectScan: Sequential bottleneck (cannot parallelize prefix sum)\n");
  printf("- VectSum: Parallelized with manual reduction\n");

  free(A); free(B); free(C); free(D);
  return 0;
}