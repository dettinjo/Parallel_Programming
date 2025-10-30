#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#define INDEX(i, j, M) ((i) * (M) + (j))

int main(int argc, char** argv) {
    
    if (argc != 5) {
        printf("Usage: %s <M dim (int)> <N dim (int)> <Max Iterations> <Tolerance>\n", argv[0]);
        return 1;
    }
    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int maxiter = atoi(argv[3]);
    float tol = atof(argv[4]);
    
    float *A = NULL;
    float *Anew = NULL;
    
    A = (float*) malloc(N * M * sizeof(float));
    Anew = (float*) malloc(N * M * sizeof(float));

    if (A == NULL || Anew == NULL) {
        printf("Error: Memory allocation failed!\n");
        if (A) free(A);
        if (Anew) free(Anew);
        return 1;
    }

    int iter = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[INDEX(i, j, M)] = 0.0f;
            Anew[INDEX(i, j, M)] = 0.0f;
        }
    }

    for (int j = 0; j < M; j++) {
        A[INDEX(0, j, M)] = 0.0f;
        A[INDEX(N-1, j, M)] = 0.0f;
    }
    
    for (int i = 0; i < N; i++) {
        A[INDEX(i, 0, M)] = sin(i * M_PI / (N - 1));
        A[INDEX(i, M-1, M)] = exp(-M_PI) * sin(i * M_PI / (N - 1));
    }
    
    printf("\n");
    
    bool iterstop = false;
    while ((iter < maxiter) && !iterstop)
    {
        float maxdiff = 0.0f;

        for (int i = 1; i < (N-1); i++){
            for (int j = 1; j < (M-1); j++){
                Anew[INDEX(i, j, M)] = (A[INDEX(i-1, j, M)] + 
                                        A[INDEX(i+1, j, M)] + 
                                        A[INDEX(i, j-1, M)] + 
                                        A[INDEX(i, j+1, M)]) / 4.0f;
                
                float diff = fabsf(A[INDEX(i, j, M)] - Anew[INDEX(i, j, M)]);
                if (diff > maxdiff){
                    maxdiff = diff;
                }
            }
        }

        if (maxdiff < tol){
            iterstop = true;
        }

        float *temp = A;
        A = Anew;
        Anew = temp;
        
        for (int j = 0; j < M; j++) {
            A[INDEX(0, j, M)] = 0.0f;
            A[INDEX(N-1, j, M)] = 0.0f;
        }
        
        for (int i = 0; i < N; i++){
            A[INDEX(i, 0, M)] = sin(i * M_PI / (N - 1));
            A[INDEX(i, M-1, M)] = exp(-M_PI) * sin(i * M_PI / (N - 1));
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

    free(A);
    free(Anew);

    return 0;
}