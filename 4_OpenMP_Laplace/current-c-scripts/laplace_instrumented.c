// Instrumented Laplace solver for performance analysis
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define PI (3.1415926535897932384626)

// High resolution timer
double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main(int argc, char** argv) {
    int i, j, iter = 0;
    
    // Initialize n and m with default values
    int n = 100, m = 100;
    
    // Get runtime arguments
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) m = atoi(argv[2]);
    
    printf("=== Performance Analysis for %dx%d matrix ===\n", n, m);
    
    // Timing variables
    double start_time, end_time;
    double setup_time, main_loop_time, cleanup_time;
    double computation_time = 0, error_time = 0, update_time = 0;
    
    start_time = get_wall_time();
    
    // Memory allocation phase
    double alloc_start = get_wall_time();
    float *A[n], *Anew[n];
    for (i = 0; i < n; i++) {
        A[i] = (float*)malloc(m * sizeof(float));
        Anew[i] = (float*)malloc(m * sizeof(float));
    }
    double alloc_time = get_wall_time() - alloc_start;
    
    // Initialization phase
    double init_start = get_wall_time();
    // All the interior points in the 2D matrix are zero
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++) A[i][j] = 0;
    
    // Set boundary conditions
    for (j = 0; j < m; j++) {
        A[0][j] = 0.f;
        A[n - 1][j] = 0.f;
    }
    for (i = 0; i < n; i++) {
        A[i][0] = sinf(PI * i / (n - 1));
        A[i][m - 1] = sinf(PI * i / (n - 1)) * expf(-PI);
    }
    double init_time = get_wall_time() - init_start;
    
    setup_time = get_wall_time() - start_time;
    
    // Main computation phase
    const float tol = 1.0e-3f;
    float error = 1.0f;
    int iter_max = 100;
    
    double main_start = get_wall_time();
    
    while (error > tol && iter < iter_max) {
        // Time the computation phase
        double comp_start = get_wall_time();
        for (i = 1; i < n - 1; i++)
            for (j = 1; j < m - 1; j++)
                Anew[i][j] = (A[i][j + 1] + A[i][j - 1] + A[i - 1][j] + A[i + 1][j]) / 4;
        computation_time += get_wall_time() - comp_start;
        
        // Time the error calculation phase
        double error_start = get_wall_time();
        error = 0.0f;
        for (i = 1; i < n - 1; i++)
            for (j = 1; j < m - 1; j++)
                error = fmaxf(error, sqrtf(fabsf(Anew[i][j] - A[i][j])));
        error_time += get_wall_time() - error_start;
        
        // Time the update phase
        double update_start = get_wall_time();
        for (i = 1; i < n - 1; i++)
            for (j = 1; j < m - 1; j++) A[i][j] = Anew[i][j];
        update_time += get_wall_time() - update_start;
        
        iter++;
        if (iter % 10 == 0) printf("Iteration %5d, error: %0.6f\n", iter, error);
    }
    
    main_loop_time = get_wall_time() - main_start;
    
    // Cleanup phase
    double cleanup_start = get_wall_time();
    for (i = 0; i < n; i++) {
        free(A[i]);
        free(Anew[i]);
    }
    cleanup_time = get_wall_time() - cleanup_start;
    
    end_time = get_wall_time();
    
    // Performance analysis output
    printf("\n=== PERFORMANCE BREAKDOWN ===\n");
    printf("Matrix size: %d x %d (%d elements)\n", n, m, n*m);
    printf("Total iterations: %d\n", iter);
    printf("Final error: %f\n", error);
    printf("Total execution time: %.6f seconds\n", end_time - start_time);
    printf("\n--- Phase Breakdown ---\n");
    printf("Memory allocation: %.6f seconds (%.2f%%)\n", alloc_time, 100*alloc_time/(end_time-start_time));
    printf("Initialization: %.6f seconds (%.2f%%)\n", init_time, 100*init_time/(end_time-start_time));
    printf("Main computation: %.6f seconds (%.2f%%)\n", main_loop_time, 100*main_loop_time/(end_time-start_time));
    printf("Cleanup: %.6f seconds (%.2f%%)\n", cleanup_time, 100*cleanup_time/(end_time-start_time));
    printf("\n--- Main Loop Breakdown ---\n");
    printf("Stencil computation: %.6f seconds (%.2f%% of total, %.2f%% of main loop)\n", 
           computation_time, 100*computation_time/(end_time-start_time), 100*computation_time/main_loop_time);
    printf("Error calculation: %.6f seconds (%.2f%% of total, %.2f%% of main loop)\n", 
           error_time, 100*error_time/(end_time-start_time), 100*error_time/main_loop_time);
    printf("Matrix update: %.6f seconds (%.2f%% of total, %.2f%% of main loop)\n", 
           update_time, 100*update_time/(end_time-start_time), 100*update_time/main_loop_time);
    
    // Memory bandwidth analysis
    long long bytes_per_iter = (long long)(n-2) * (m-2) * sizeof(float) * 9; // 5 reads + 1 write for stencil, 3 for error/update
    double total_bytes = bytes_per_iter * iter;
    double bandwidth_gb_s = (total_bytes / 1e9) / main_loop_time;
    
    printf("\n--- Memory Analysis ---\n");
    printf("Bytes per iteration: %lld\n", bytes_per_iter);
    printf("Total data movement: %.2f GB\n", total_bytes / 1e9);
    printf("Average memory bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    printf("Operations per iteration: %lld\n", (long long)(n-2) * (m-2) * 4); // 4 ops per stencil
    printf("Total operations: %.2f billion\n", ((long long)(n-2) * (m-2) * 4 * iter) / 1e9);
    printf("Average GFLOPS: %.2f\n", (((long long)(n-2) * (m-2) * 4 * iter) / 1e9) / main_loop_time);
    
    return 0;
}