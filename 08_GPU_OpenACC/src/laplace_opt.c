#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Optimized Laplace Solver (OpenACC)
// Techniques:
// 1. Pointer Swapping: Eliminates the expensive data copy kernel.
// 2. Loop Collapse: Maximizes GPU thread occupancy (16M threads).
// 3. Data Residency: Data stays on GPU memory for the entire loop.

int main(int argc, char** argv)
{
    // --- 1. Setup ---
    int n = 4096;
    int m = 4096;
    int iter_max = 100;
    
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) m = atoi(argv[2]);
    if (argc > 3) iter_max = atoi(argv[3]);

    size_t total_size = (size_t)n * m;
    
    // Allocate Host Memory (restrict helps compiler optimizations)
    float * restrict A    = (float*) malloc(total_size * sizeof(float));
    float * restrict Anew = (float*) malloc(total_size * sizeof(float));

    const float tol = 1.0e-8f;
    float error = 1.0f;
    const float pi = 2.0f * asinf(1.0f);

    // --- 2. Initialization ---
    memset(A, 0, total_size * sizeof(float));
    
    // Set Boundary Conditions
    for (int i = 0; i < m; i++) {
        A[i] = 0.f;              
        A[(n-1)*m + i] = 0.f;    
    }
    for (int j = 0; j < n; j++) {
        float val = sinf(pi * j / (n-1));
        A[j*m] = val;            
        A[j*m + m - 1] = val * expf(-pi); 
    }
    
    // Singular point
    A[(n/128)*m + m/128] = 1.0f;

    printf("Jacobi relaxation: %d x %d mesh, max %d iterations\n", n, m, iter_max);

    // --- 3. GPU Execution ---
    int iter = 0;
    
    // Pointers for "Ping-Pong" buffering
    float *input = A;
    float *output = Anew;

    // DATA REGION:
    // We map the memory of 'A' and 'Anew' to the device once.
    // 'present' inside the loop will lookup these addresses.
    #pragma acc data copy(A[0:total_size]) create(Anew[0:total_size])
    {
        while (error > tol && iter < iter_max)
        {
            error = 0.f;

            // KERNEL 1: Stencil Computation
            // Reads from 'input', Writes to 'output'
            // collapse(2) creates a massive 1D grid of threads
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
            // Reads both arrays. 'async(1)' allows it to queue after Kernel 1.
            // Note: This forces a synchronization on the host to read 'error' back.
            #pragma acc parallel loop collapse(2) reduction(max:error) present(input, output) async(1)
            for (int j = 1; j < n - 1; j++) {
                for (int i = 1; i < m - 1; i++) {
                    float diff = fabsf(output[j*m + i] - input[j*m + i]);
                    error = fmaxf(error, sqrtf(diff));
                }
            }
            
            // Wait for GPU to finish this step before checking error on CPU
            #pragma acc wait(1)

            // OPTIMIZATION: Pointer Swap (Ping-Pong)
            // Instead of copying data back (O(N)), we just swap pointers (O(1)).
            float *temp = input;
            input = output;
            output = temp;

            iter++;
            if (iter % (iter_max/10) == 0) printf("%5d, %0.6f\n", iter, error);
        }

        // --- 4. Final Data Handling ---
        // Since we swapped pointers, the final result might be in 'Anew' (buffer 2).
        // If 'input' is currently pointing to 'Anew', we must copy it back to 'A'
        // so the 'acc data copyout(A)' works correctly.
        
        if (input != A) {
            // The valid data is in 'input' (which is Anew's buffer).
            // We copy it to 'output' (which is A's buffer).
            #pragma acc parallel loop collapse(2) present(input, output)
            for (int j = 1; j < n - 1; j++) {
                 for (int i = 1; i < m - 1; i++) {
                     output[j*m + i] = input[j*m + i];
                 }
            }
        }

    } // End acc data (Implicit copyout of A)

    printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
    printf("A[%d][%d]= %0.6f\n", n/128, m/128, A[(n/128)*m + m/128]);

    free(A);
    free(Anew);

    return 0;
}