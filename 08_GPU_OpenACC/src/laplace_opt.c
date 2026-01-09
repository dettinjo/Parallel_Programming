#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Optimized implementation of Laplace Solver
// Uses dynamic memory (malloc) but optimized with OpenACC for GPU
// Strategy: 
// 1. Single Data Region to minimize HtoD and DtoH copies.
// 2. Inlined logic to avoid function call overhead.
// 3. 'Parallel loop collapse' for maximum parallelism.

int main(int argc, char** argv)
{
    // Defaults
    int n = 4096;
    int m = 4096;
    int iter_max = 100;
    
    // Command line arguments
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) m = atoi(argv[2]);
    if (argc > 3) iter_max = atoi(argv[3]);

    // Allocation (Dynamic Memory - flat 1D arrays simulating 2D)
    // Using long long for size calculation to prevent overflow on very large meshes
    size_t total_size = (size_t)n * m;
    float * restrict A    = (float*) malloc(total_size * sizeof(float));
    float * restrict Anew = (float*) malloc(total_size * sizeof(float));

    const float tol = 1.0e-8f;
    float error = 1.0f;
    const float pi = 2.0f * asinf(1.0f);

    // --- Initialization (Host Side) ---
    // We do this on the host before moving data to GPU
    memset(A, 0, total_size * sizeof(float));
    
    // Boundary conditions
    // Top and Bottom rows
    for (int i = 0; i < m; i++) {
        A[i] = 0.f;              // Row 0
        A[(n-1)*m + i] = 0.f;    // Row n-1
    }
    
    // Left and Right columns
    for (int j = 0; j < n; j++) {
        float val = sinf(pi * j / (n-1));
        A[j*m] = val;            // Col 0
        A[j*m + m - 1] = val * expf(-pi); // Col m-1
    }

    // Singular point
    A[(n/128)*m + m/128] = 1.0f;

    printf("Jacobi relaxation: %d x %d mesh, max %d iterations\n", n, m, iter_max);

    int iter = 0;

    // --- GPU Region ---
    // copy(A): Copy initial boundary conditions to GPU
    // create(Anew): Allocate scratch space on GPU (no copy needed)
    #pragma acc data copy(A[0:total_size]) create(Anew[0:total_size])
    {
        while (error > tol && iter < iter_max)
        {
            error = 0.f;

            // 1. Computation Step
            // collapse(2) merges j and i loops into one long loop of size (n-2)*(m-2)
            // independent ensures compiler knows iterations don't depend on each other
            #pragma acc parallel loop collapse(2) present(A, Anew)
            for (int j = 1; j < n - 1; j++) {
                for (int i = 1; i < m - 1; i++) {
                    // Access formula: [j*m + i]
                    Anew[j*m + i] = 0.25f * (
                        A[j*m + (i+1)] + 
                        A[j*m + (i-1)] + 
                        A[(j-1)*m + i] + 
                        A[(j+1)*m + i]
                    );
                }
            }

            // 2. Error Check Step
            // reduction(max:error) efficiently calculates max global error on GPU
            #pragma acc parallel loop collapse(2) reduction(max:error) present(A, Anew)
            for (int j = 1; j < n - 1; j++) {
                for (int i = 1; i < m - 1; i++) {
                    float diff = fabsf(A[j*m + i] - Anew[j*m + i]);
                    // Only apply sqrt/fmax if necessary to save cycles? 
                    // The math requires sqrt of difference? No, original code was:
                    // sqrtf( fabsf( old - new )). 
                    // Note: This matches the provided math in source templates.
                    error = fmaxf(error, sqrtf(diff));
                }
            }

            // 3. Copy Step (Swap)
            // Standard copy is memory bound. Parallelizing it is crucial.
            #pragma acc parallel loop collapse(2) present(A, Anew)
            for (int j = 1; j < n - 1; j++) {
                for (int i = 1; i < m - 1; i++) {
                    A[j*m + i] = Anew[j*m + i];
                }
            }

            iter++;
            if (iter % (iter_max/10) == 0) printf("%5d, %0.6f\n", iter, error);
        }
    } // End acc data (Implicit copyout of A happens here)

    printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
    printf("A[%d][%d]= %0.6f\n", n/128, m/128, A[(n/128)*m + m/128]);

    free(A);
    free(Anew);

    return 0;
}