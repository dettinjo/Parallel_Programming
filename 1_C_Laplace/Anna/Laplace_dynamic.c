#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) 
{ 
    // declare local variables: error, tol, iter_max ... 
    double error;
    double tol;
    int iter_max;
    // get iter_max from command line at execution time
    iter_max = atoi(argv[1]);
    // get tol from command line at execution time
    tol = atof(argv[2]);
    // boundary conditions for up and down
    n= atof(argv[3]);
    m= atof(argv[4]);

    double *A = (double*)calloc(n * m, sizeof(double));
    double *Anew = (double*)calloc(n * m, sizeof(double));

    for (int j = 0; j < m+1; j++){
        A[0 * m + j] = 0;
        A[(n-1) * m + j] = 0;
    }
    // boundary conditions for left and right
    for (int i = 1; i < n+1; i++) {
        A[i * m + 0] = sin(M_PI * i / (n - 1));
        A[i * m + (m-1)] = pow(M_E, -M_PI) * sin(M_PI * i / (n - 1));
    }
    error = tol + 1; // to ensure at least one iteration
    int iter = 0; // iteration counter  

  // Main loop: iterate until error <= tol a maximum of iter_max iterations 
  while ( error > tol && iter < iter_max ) { 
    // Compute new values using main matrix and writing into auxiliary matrix 
    // Compute error = maximum of the square root of the absolute differences 
    // Copy from auxiliary matrix to main matrix 
    // if number of iterations is multiple of 10 then print error on the screen    
    for(i = 1; i < n; i++) {
        
        for(j = 1; j < m; j++) {
            //new matrix computation
            Anew[i * m + j] = 0.25 * (A[(i+1) * m + j] + A[(i-1) * m + j] + A[i * m + (j+1)] + A[i * m + (j-1)]);
            error = fmax(error, fabs(Anew[i * m + j] - A[i * m + j]));
        }

        // Copy from auxiliary matrix to main matrix
        for(j = 1; j < m; j++) {
            A[i * m + j] = Anew[i * m + j];
        }
    }
  } // while 
}