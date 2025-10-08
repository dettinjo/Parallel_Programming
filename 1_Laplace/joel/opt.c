#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Need M_PI. If not defined by math.h (depends on standard), define it.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==========================================
// Provided Structure Starts Here
// ==========================================

// define n and m (Making them global integers for dynamic sizing per Task 2)
int n;
int m;

// declare global variables: A and Anew (Pointers for dynamic allocation)
float **A;
float **Anew;

// Helper function to allocate 2D contiguous memory (Not explicitly in structure, but needed)
float** allocate_matrix(int rows, int cols) {
    float **matrix = (float **)malloc(rows * sizeof(float *));
    if (matrix == NULL) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    
    // Allocate the actual data block as one contiguous chunk
    float *data = (float *)malloc(rows * cols * sizeof(float));
    if (data == NULL) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }

    // Set pointers to the start of each row
    for (int i = 0; i < rows; i++) {
        matrix[i] = &(data[i * cols]);
    }
    return matrix;
}

// Helper function to free memory
void free_matrix(float **matrix) {
    free(matrix[0]); // Free the contiguous data block
    free(matrix);    // Free the row pointers
}

int main(int argc, char** argv) 
{ 
  // declare local variables: error, tol, iter_max ... 
  int iter = 0;
  int iter_max;
  float tol;
  float error;
  
  // Set defaults if arguments aren't provided
  n = 128; 
  m = 128;
  iter_max = 1000;
  tol = 1e-5f;

  // get iter_max from command line at execution time (and n, m, tol for completeness)
  if (argc > 1) n = atoi(argv[1]);
  if (argc > 2) m = atoi(argv[2]);
  if (argc > 3) iter_max = atoi(argv[3]);
  if (argc > 4) tol = (float)atof(argv[4]);

  printf("Parameters: n=%d, m=%d, iter_max=%d, tol=%e\n", n, m, iter_max, tol);

  // Allocate memory for A and Anew
  A = allocate_matrix(n, m);
  Anew = allocate_matrix(n, m);

  // set all values in matrix as zero 
  // (Initializing the interior and default boundaries)
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
          A[i][j] = 0.0f;
          Anew[i][j] = 0.0f; // Initialize Anew as well
      }
  }

  // set boundary conditions 
  // According to assignment: 
  // Row 0 and n-1 are 0 (already set above).
  // Col 0: sin(i * pi / (n-1))
  // Col m-1: e^-pi * sin(i * pi / (n-1))
  float exp_neg_pi = expf((float)-M_PI);
  for (int i = 0; i < n; i++) {
      // Avoid division by zero if n=1, though n should be > 2 for stencil
      float denominator = (n > 1) ? (float)(n - 1) : 1.0f;
      float sin_val = sinf( (float)i * (float)M_PI / denominator );
      
      A[i][0] = sin_val;           // Left border
      A[i][m - 1] = exp_neg_pi * sin_val; // Right border
  }

  // Ensure we enter the loop
  error = tol + 1.0f;

  // Main loop: iterate until error <= tol a maximum of iter_max iterations 
  while ( error > tol && iter < iter_max ) { 

    // Compute new values using main matrix and writing into auxiliary matrix 
    // Only update interior points (1 to n-2, 1 to m-2)
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < m - 1; j++) {
            Anew[i][j] = 0.25f * (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
        }
    }

 // Compute error and copy from auxiliary matrix in a single fused loop
error = 0.0f;
for (int i = 1; i < n - 1; i++) {
    for (int j = 1; j < m - 1; j++) {
        // Calculate error for the current point
        float diff = fabsf(Anew[i][j] - A[i][j]);
        float current_error = sqrtf(diff);
        if (current_error > error) {
            error = current_error;
        }
        // Copy the new value to the main matrix for the next iteration
        A[i][j] = Anew[i][j];
    }
}

    iter++;

    // if number of iterations is multiple of 10 then print error on the screen    
    if (iter % 10 == 0 || iter == 1) {
        printf("Iteration: %d, Maximum Error: %f\n", iter, error);
    }

  } // while 

  // Final report
  if (error <= tol) {
      printf("Converged at iteration %d with error %f\n", iter, error);
  } else {
      printf("Reached maximum iterations (%d) with error %f\n", iter_max, error);
  }

  // Cleanup
  free_matrix(A);
  free_matrix(Anew);

  return 0;
}