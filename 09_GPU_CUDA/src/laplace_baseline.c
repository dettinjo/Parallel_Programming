#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    int i, j;
    int n = 4096, m = 4096, iter_max = 10000;
    
    /* Get runtime arguments */
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) m = atoi(argv[2]);
    if (argc > 3) iter_max = atoi(argv[3]);
    
    /* FIXED: Allocate arrays AFTER parameters are parsed */
    float *A = malloc(n * m * sizeof(float));
    float *Anew = malloc(n * m * sizeof(float));
    float *y = malloc(n * sizeof(float));
    
    /* Check allocation success */
    if (!A || !Anew || !y) {
        printf("Error: Memory allocation failed!\n");
        free(A); free(Anew); free(y);
        return 1;
    }

    const float pi = 2.0f * asinf(1.0f);
    const float tol = 1.0e-8f;
    float error = 1.0f;    

    /* REMOVED: Duplicate/incorrect line: if (argc>1) {  iter_max = atoi(argv[1]); } */

    /* Initialize matrix - using 1D indexing: A[i][j] becomes A[i*m + j] */
    memset(A, 0, n * m * sizeof(float));
    
    /* Set boundary conditions - top and bottom rows */
    for (i = 0; i < m; i++) {
       A[0 * m + i] = 0.f;           /* A[0][i] */
       A[(n-1) * m + i] = 0.f;       /* A[n-1][i] */
    }

    /* Set boundary conditions - left and right columns */
    for (j = 0; j < n; j++) {
       y[j] = sinf(pi * j / (n-1));
       A[j * m + 0] = y[j];                    /* A[j][0] */
       A[j * m + (m-1)] = y[j] * expf(-pi);    /* A[j][m-1] */
    }

    /* Set singular point */
    A[(n/128) * m + (m/128)] = 1.0f;           /* A[n/128][m/128] */

    printf("Jacobi relaxation Calculation: %d x %d mesh, maximum of %d iterations\n", 
           n, m, iter_max);

    int iter = 0;

    /* FIXED: Pure CPU baseline - NO OpenACC pragmas */
    while (error > tol && iter < iter_max) {
       error = 0.f;

       /* Stencil computation */
       for (i = 1; i < n-1; i++) {
          for (j = 1; j < m-1; j++) {
              Anew[i * m + j] = (A[i * m + (j+1)] + A[i * m + (j-1)] + 
                                A[(i-1) * m + j] + A[(i+1) * m + j]) / 4.0f;
          }
       }

       /* Error calculation */
       for (i = 1; i < n-1; i++) {
          for (j = 1; j < m-1; j++) {
              error = fmaxf(error, sqrtf(fabsf(A[i * m + j] - Anew[i * m + j])));
          }
       }

       /* Copy new values back */
       for (i = 1; i < n-1; i++) {
          for (j = 1; j < m-1; j++) {
              A[i * m + j] = Anew[i * m + j];
          }
       }

       iter++;
       if (iter % (iter_max/10) == 0) printf("%5d, %0.6f\n", iter, error);
    }

    printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
    printf("A[%d][%d]= %0.6f\n", n/128, m/128, A[(n/128) * m + (m/128)]);

    /* FIXED: Free allocated memory */
    free(A);
    free(Anew); 
    free(y);

    return 0;
}
