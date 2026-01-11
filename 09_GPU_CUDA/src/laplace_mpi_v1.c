#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define PI (3.1415926535897932384626)

void exchange_boundaries(float **A, int local_rows, int m, int rank, int size) {
    MPI_Status status;
    
    /* Send to upper neighbour, receive from lower */
    if (rank > 0) {
        MPI_Send(A[1], m, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);
        MPI_Recv(A[0], m, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD, &status);
    }
    
    /* Send to lower neighbour, receive from upper */
    if (rank < size - 1) {
        MPI_Send(A[local_rows], m, MPI_FLOAT, rank + 1, 2, MPI_COMM_WORLD);
        MPI_Recv(A[local_rows + 1], m, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &status);
    }
}

int main(int argc, char** argv) {
    /* Original variables */
    int i, j, iter = 0;
    int n, m;
    int iter_max;
    float **A, **Anew;
    const float tol = 1.0e-3f;
    float error = 1.0f;
    double start_time, end_time;
    int rank, size;
    
    /* NEW: MPI-specific variables */
    int rows_per_proc, start_row, end_row, local_rows, total_local_rows;
    float **A_local, **Anew_local;
    float *A_data, *Anew_data;
    float local_error;
    MPI_Status status;

    /* MPI setup */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Get runtime arguments */
    if (argc > 1) n = atoi(argv[1]);
    else n = 100;
    if (argc > 2) m = atoi(argv[2]);
    else m = 100;
    if (argc > 3) iter_max = atoi(argv[3]);
    else iter_max = 1000;

    /* Calculate how many rows each process handles */
    rows_per_proc = (n - 2) / size;  /* Interior rows only */
    start_row = rank * rows_per_proc + 1;
    end_row = (rank == size - 1) ? n - 1 : start_row + rows_per_proc;
    local_rows = end_row - start_row;
    total_local_rows = local_rows + 2;  /* +2 for ghost rows */

    /* Allocate local arrays with contiguous memory */
    A_data = malloc(total_local_rows * m * sizeof(float));
    Anew_data = malloc(total_local_rows * m * sizeof(float));
    A_local = malloc(total_local_rows * sizeof(float*));
    Anew_local = malloc(total_local_rows * sizeof(float*));

    for (i = 0; i < total_local_rows; i++) {
        A_local[i] = A_data + i * m;
        Anew_local[i] = Anew_data + i * m;
    }

    /* Rank 0 handles full matrix initialization and distribution */
    if (rank == 0) {
        A = malloc(n * sizeof(float*));
        Anew = malloc(n * sizeof(float*));
        for (i = 0; i < n; i++) {
            A[i] = malloc(m * sizeof(float));
            Anew[i] = malloc(m * sizeof(float));
        }

        /* Initialize full matrix on rank 0 */
        for (i = 0; i < n; i++)
            for (j = 0; j < m; j++) A[i][j] = 0;

        /* Set boundary conditions */
        for (j = 0; j < m; j++) {
            A[0][j] = 0.f;
            A[n - 1][j] = 0.f;
        }
        for (i = 0; i < n; i++) {
            A[i][0] = sinf(PI * i / (n - 1));
            A[i][m - 1] = sinf(PI * i / (n - 1)) * expf(-PI);
        }
        
        /* Distribute data to other processes */
        for (int proc = 1; proc < size; proc++) {
            int proc_start = proc * rows_per_proc + 1;
            int proc_end = (proc == size - 1) ? n - 1 : proc_start + rows_per_proc;
            int proc_rows = proc_end - proc_start;
            
            /* Send local data (including one row above and below for ghost rows) */
            for (i = 0; i < proc_rows + 2; i++) {
                int global_row = proc_start - 1 + i;
                if (global_row >= 0 && global_row < n) {
                    MPI_Send(A[global_row], m, MPI_FLOAT, proc, i, MPI_COMM_WORLD);
                }
            }
        }
        
        /* Copy rank 0's portion to local arrays */
        for (i = 0; i < total_local_rows; i++) {
            for (j = 0; j < m; j++) {
                A_local[i][j] = A[i][j];
            }
        }
    } else {
        /* Other processes receive their data */
        for (i = 0; i < total_local_rows; i++) {
            MPI_Recv(A_local[i], m, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
        }
    }

    /* Initialize Anew_local with A_local values */
    for (i = 0; i < total_local_rows; i++)
        for (j = 0; j < m; j++)
            Anew_local[i][j] = A_local[i][j];

    /* START TIMER */
    start_time = MPI_Wtime();

    /* Main iteration loop */
    while (error > tol && iter < iter_max) {
        /* Exchange boundary data between processes */
        exchange_boundaries(A_local, local_rows, m, rank, size);
        
        /* Calculate the new values for local domain */
        for (i = 1; i <= local_rows; i++)
            for (j = 1; j < m - 1; j++)
                Anew_local[i][j] =
                    (A_local[i][j + 1] + A_local[i][j - 1] + A_local[i - 1][j] + A_local[i + 1][j]) * 0.25f;

        /* Compute local error */
        local_error = 0.0f;
        for (i = 1; i <= local_rows; i++)
            for (j = 1; j < m - 1; j++)
                local_error = fmaxf(local_error, fabsf(Anew_local[i][j] - A_local[i][j]));

        /* Reduce to find global maximum error */
        MPI_Allreduce(&local_error, &error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

        /* Update A_local with Anew_local */
        for (i = 1; i <= local_rows; i++)
            for (j = 1; j < m - 1; j++) 
                A_local[i][j] = Anew_local[i][j];

        iter++;
    }

    /* STOP TIMER */
    end_time = MPI_Wtime();

    /* Output EXACTLY like sequential version - only rank 0 prints */
    if (rank == 0) {
        printf("%f\n", end_time - start_time);
    }

    /* Free local arrays */
    free(A_data);
    free(Anew_data);
    free(A_local);
    free(Anew_local);

    /* Free full arrays (only rank 0 allocated them) */
    if (rank == 0) {
        for (i = 0; i < n; i++) {
            free(A[i]);
            free(Anew[i]);
        }
        free(A);
        free(Anew);
    }

    MPI_Finalize();
    return 0;
}