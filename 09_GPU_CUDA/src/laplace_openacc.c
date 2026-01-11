#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// FINAL OPTIMIZED LAPLACE SOLVER
// Techniques used:
// 1. Dynamic 1D Memory: Runtime sizing + Coalesced Memory Access.
// 2. Pointer Swapping: Eliminates the expensive data copy kernel (O(N) savings).
// 3. Periodic Reduction: Checks error only every 100 iterations to hide CPU-GPU sync latency.
// 4. Collapse(2): Maximizes thread occupancy to hide memory latency.

int main(int argc, char** argv)
{
    // --- 1. Setup ---
    int n = 4096;
    int m = 4096;
    int iter_max = 1000;
    
    // Parse command line arguments for runtime flexibility
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) m = atoi(argv[2]);
    if (argc > 3) iter_max = atoi(argv[3]);

    size_t total_size = (size_t)n * m;
    
    // Allocate Host Memory
    // 'restrict' keyword promises the compiler that these pointers do not overlap,
    // enabling more aggressive optimization.
    float * restrict A    = (float*) malloc(total_size * sizeof(float));
    float * restrict Anew = (float*) malloc(total_size * sizeof(float));

    const float tol = 1.0e-8f;
    float error = 1.0f;
    const float pi = 2.0f * asinf(1.0f);

    // --- 2. Initialization ---
    // Initialize array with 0.0
    memset(A, 0, total_size * sizeof(float));
    
    // Boundary Conditions: Top and Bottom rows
    for (int i = 0; i < m; i++) {
        A[i] = 0.f;              
        A[(n-1)*m + i] = 0.f;    
    }
    // Boundary Conditions: Left and Right columns
    for (int j = 0; j < n; j++) {
        float val = sinf(pi * j / (n-1));
        A[j*m] = val;            
        A[j*m + m - 1] = val * expf(-pi); 
    }
    
    // Singular point initialization
    A[(n/128)*m + m/128] = 1.0f;

    printf("Jacobi relaxation: %d x %d mesh, max %d iterations\n", n, m, iter_max);

    // --- 3. GPU Execution ---
    int iter = 0;
    
    // Pointers for "Ping-Pong" buffering (swapping instead of copying)
    float *input = A;
    float *output = Anew;

    // DATA REGION
    // We copy 'A' in, create space for 'Anew', and keep data on GPU for the whole loop.
    #pragma acc data copy(A[0:total_size]) create(Anew[0:total_size])
    {
        while (error > tol && iter < iter_max)
        {
            // KERNEL 1: Stencil Computation
            // 'collapse(2)' flattens the loop into a 1D grid of 16 million threads.
            // This ensures maximum occupancy (active warps) to hide memory latency.
            // 'async(1)' allows the CPU to proceed without waiting for the GPU.
            #pragma acc parallel loop collapse(2) present(input, output) async(1)
            for (int j = 1; j < n - 1; j++) {
                for (int i = 1; i < m - 1; i++) {
                    output[j*m + i] = 0.25f * (
                        input[j*m + (i+1)] + 
                        input[j*m + (i-1)] + 
                        input[(j-1)*m + i] + 
                        input[(j+1)*m + i]
                    );
                }
            }

            // KERNEL 2: Error Reduction
            // Optimization: Only check error every 100 iterations.
            // This eliminates 99% of the CPU-GPU synchronization overhead.
            if (iter % 100 == 0 || iter == iter_max - 1) 
            {
                error = 0.f; 
                
                #pragma acc parallel loop collapse(2) reduction(max:error) present(input, output) async(1)
                for (int j = 1; j < n - 1; j++) {
                    for (int i = 1; i < m - 1; i++) {
                        float diff = fabsf(output[j*m + i] - input[j*m + i]);
                        error = fmaxf(error, diff);
                    }
                }
                // We must wait here to read the error value on the host
                #pragma acc wait(1) 
                
                if (iter % (iter_max/10) == 0) printf("%5d, %0.6f\n", iter, error);
            }

            // OPTIMIZATION: Pointer Swap (Ping-Pong)
            // Swap pointers on CPU. Next iteration reads from 'output' and writes to 'input'.
            float *temp = input;
            input = output;
            output = temp;

            iter++;
        }

        // --- 4. Final Data Handling ---
        // If the final valid data is in 'input' (which effectively points to Anew's buffer),
        // we must copy it to 'output' (A's buffer) so the implicit copyout works correctly.
        if (input != A) {
            #pragma acc parallel loop collapse(2) present(input, output)
            for (int j = 1; j < n - 1; j++) {
                 for (int i = 1; i < m - 1; i++) {
                     output[j*m + i] = input[j*m + i];
                 }
            }
        }
    } // End data region (Implicit copyout of A)

    printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
    printf("A[%d][%d]= %0.6f\n", n/128, m/128, A[(n/128)*m + m/128]);

    free(A);
    free(Anew);

    return 0;
}