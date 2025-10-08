#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// define n and m 
int m = 5;
int n = 4;
// declare global variables: A and Anew
double A[4][5];
double Anew[4][5];

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
    for (int j = 0; j < m+1; j++){
        A[0][j] = 0;
        A[n-1][j] = 0;
    }
    // boundary conditions for left and right
    for (int i = 1; i < n+1; i++) {
        A[i][0]= sin(M_PI * i / (n - 1));
        A[i][m-1]= pow(M_E, -M_PI) * sin(M_PI * i / (n - 1));
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
            Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j+1] + A[i][j-1]);
            error = fmax(error, fabs(Anew[i][j] - A[i][j]));
        }

        // Copy from auxiliary matrix to main matrix
        for(j = 1; j < m; j++) {
            A[i][j] = Anew[i][j];
        }
    }
  } // while 
}