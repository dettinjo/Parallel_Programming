// VERSION 1: STATIC ALLOCATION
// Fulfills Task 1 of the assignment.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// For this version, n and m are constant values defined at compile time.
#define N 4096  
#define M 4096

// Define Pi if it's not already available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Declare global variables for the matrices.
// Because N and M are constants, we can declare the arrays directly.
// This is called STATIC allocation.
float A[N][M];
float Anew[N][M];

int main(int argc, char** argv) 
{ 
  // Local variables for the simulation control
  int iter = 0;
  // For this simple version, iter_max and tol are also hard-coded.
  int iter_max = 1000;
  float tol = 1e-5f;
  float error = 0.0f;

  printf("Parameters: n=%d, m=%d, iter_max=%d, tol=%e\n", N, M, iter_max, tol);

  // NOTE: No memory allocation is needed here. The arrays A and Anew
  // were created automatically when the program started.

  // Set all values in both matrices to zero
  for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
          A[i][j] = 0.0f;
          Anew[i][j] = 0.0f;
      }
  }

  // Set the boundary conditions on the main matrix 'A'
  float exp_neg_pi = expf((float)-M_PI);
  for (int i = 0; i < N; i++) {
      float denominator = (N > 1) ? (float)(N - 1) : 1.0f;
      float sin_val = sinf((float)i * (float)M_PI / denominator);
      
      A[i][0] = sin_val;        // Left border
      A[i][M - 1] = exp_neg_pi * sin_val; // Right border
  }

  // Set the initial error to a value greater than the tolerance to ensure the loop runs at least once
  error = tol + 1.0f;

  // Main convergence loop
  while (error > tol && iter < iter_max) { 

    // 1. Compute new values and store them in the auxiliary matrix 'Anew'
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < M - 1; j++) {
            Anew[i][j] = 0.25f * (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
        }
    }

    // 2. Compute the maximum error between the old and new values
    error = 0.0f;
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < M - 1; j++) {
            float diff = fabsf(Anew[i][j] - A[i][j]);
            float current_error = sqrtf(diff);
            if (current_error > error) {
                error = current_error;
            }
        }
    }

    // 3. Copy the new values from 'Anew' back to 'A' for the next iteration
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < M - 1; j++) {
            A[i][j] = Anew[i][j];
        }
    }

    iter++;

    // Print the error every 10 iterations
    if (iter % 10 == 0 || iter == 1) {
        printf("Iteration: %d, Maximum Error: %f\n", iter, error);
    }
  }

  // Final report on why the loop stopped
  if (error <= tol) {
      printf("Converged at iteration %d with error %f\n", iter, error);
  } else {
      printf("Reached maximum iterations (%d) with error %f\n", iter_max, error);
  }

  // NOTE: No memory cleanup (free) is needed. The arrays will be
  // automatically destroyed when the program ends.

  return 0;
}