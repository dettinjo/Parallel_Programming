#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

// Global
#define N 100
#define M 100

int main(int argc, char** argv) {
    
    // Check there are three arguments {script maxiter tol}
    if (argc != 3) {
        printf("Usage: %s <Max Iterations> <Tolerance>\n", argv[0]);
        return 1;
    }

    // Declare tolerance and max iterations
    int maxiter = atoi(argv[1]);
    float tol = atof(argv[2]);
    
    // Declare matrices
    float A[N][M];
    float Anew[N][M];
    int iter = 0;

    // Step 1: Set all values to zero
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[i][j] = 0.0f;
            Anew[i][j] = 0.0f;
        }
    }

    for (int j = 0; j < M; j++) {
        A[0][j] = 0.0f;       // Top row
        A[N-1][j] = 0.0f;     // Bottom row
    }
    
    for (int i = 0; i < N; i++) {
        A[i][0] = sin(i * M_PI / (N - 1));                    // Left column
        A[i][M-1] = exp(-M_PI) * sin(i * M_PI / (N - 1));    // Right column
    }
    
/*     // Print the initial matrix to verify
    printf("Initial (K) matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%8.4f ", A[i][j]);
        }
        printf("\n");
    }
    
    printf("\n"); */
    
    bool iterstop = false;
    while ((iter < maxiter) && !iterstop)
    {
        float maxdiff = 0.0f;

        for (int i = 1; i < (N-1); i++){
            for (int j = 1; j < (M-1); j++){
                Anew[i][j] = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1])/4;
                float diff = (fabs(A[i][j] - Anew[i][j]));
                if (diff > maxdiff){
                    maxdiff = diff;
                }
            }
        }

        if (maxdiff < tol){
            iterstop = true;
        }

        for (int i = 0; i < N; i++){
            Anew[i][0] = A[i][0];
            Anew[i][M-1] = A[i][M-1];
            for (int j = 0; j < M; j++){
                A[i][j] = Anew[i][j];
            }
        }
        iter++;

        if (iter % 10 == 0) {
            printf("Iteration %d: error = %f\n", iter, maxdiff);
        }
    }
    if (iterstop == true){
        printf("Tolerance reached. Tolerance given = %8.9f \n", tol);
    }
    else {
        printf("Maximum iterations reached. Max iterations given = %i \n", maxiter);
    }
    return 0;
}