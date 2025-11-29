#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define PI (3.1415926535897932384626)

void exchange_boundaries_non_blocking(float **A, int local_rows, int local_cols, 
                                     int rank_row, int rank_col, int p_rows, int p_cols, 
                                     MPI_Comm cart_comm, MPI_Request *requests, int *req_count) {
    int up, down, left, right;
    int coords[2];
    
    // Get neighbour ranks in 2D topology
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);     // row dimension (vertical)
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);  // col dimension (horizontal)
    
    *req_count = 0;
    
    // Exchange with upper neighbour (send top row, receive into top ghost)
    if (up != MPI_PROC_NULL) {
        MPI_Isend(A[1], local_cols, MPI_FLOAT, up, 1, cart_comm, &requests[(*req_count)++]);
        MPI_Irecv(A[0], local_cols, MPI_FLOAT, up, 2, cart_comm, &requests[(*req_count)++]);
    }
    
    // Exchange with lower neighbour (send bottom row, receive into bottom ghost)
    if (down != MPI_PROC_NULL) {
        MPI_Isend(A[local_rows], local_cols, MPI_FLOAT, down, 2, cart_comm, &requests[(*req_count)++]);
        MPI_Irecv(A[local_rows + 1], local_cols, MPI_FLOAT, down, 1, cart_comm, &requests[(*req_count)++]);
    }
    
    // Create temporary arrays for column exchange (for non-contiguous data)
    static float *left_send = NULL, *left_recv = NULL, *right_send = NULL, *right_recv = NULL;
    static int prev_rows = 0;
    
    if (local_rows + 2 != prev_rows) {
        if (left_send) free(left_send);
        if (left_recv) free(left_recv);
        if (right_send) free(right_send);
        if (right_recv) free(right_recv);
        
        left_send = malloc((local_rows + 2) * sizeof(float));
        left_recv = malloc((local_rows + 2) * sizeof(float));
        right_send = malloc((local_rows + 2) * sizeof(float));
        right_recv = malloc((local_rows + 2) * sizeof(float));
        prev_rows = local_rows + 2;
    }
    
    // Exchange with left neighbour
    if (left != MPI_PROC_NULL) {
        // Pack left column to send
        for (int i = 0; i < local_rows + 2; i++) {
            left_send[i] = A[i][1];
        }
        MPI_Isend(left_send, local_rows + 2, MPI_FLOAT, left, 3, cart_comm, &requests[(*req_count)++]);
        MPI_Irecv(left_recv, local_rows + 2, MPI_FLOAT, left, 4, cart_comm, &requests[(*req_count)++]);
    }
    
    // Exchange with right neighbour
    if (right != MPI_PROC_NULL) {
        // Pack right column to send
        for (int i = 0; i < local_rows + 2; i++) {
            right_send[i] = A[i][local_cols];
        }
        MPI_Isend(right_send, local_rows + 2, MPI_FLOAT, right, 4, cart_comm, &requests[(*req_count)++]);
        MPI_Irecv(right_recv, local_rows + 2, MPI_FLOAT, right, 3, cart_comm, &requests[(*req_count)++]);
    }
}

void unpack_column_data(float **A, int local_rows, int local_cols, 
                       int rank_row, int rank_col, int p_rows, int p_cols,
                       MPI_Comm cart_comm) {
    int up, down, left, right;
    static float *left_recv = NULL, *right_recv = NULL;
    static int prev_rows = 0;
    
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);
    
    if (local_rows + 2 != prev_rows) {
        if (left_recv) free(left_recv);
        if (right_recv) free(right_recv);
        left_recv = malloc((local_rows + 2) * sizeof(float));
        right_recv = malloc((local_rows + 2) * sizeof(float));
        prev_rows = local_rows + 2;
    }
    
    // Unpack received column data
    if (left != MPI_PROC_NULL) {
        for (int i = 0; i < local_rows + 2; i++) {
            A[i][0] = left_recv[i];
        }
    }
    
    if (right != MPI_PROC_NULL) {
        for (int i = 0; i < local_rows + 2; i++) {
            A[i][local_cols + 1] = right_recv[i];
        }
    }
}

void compute_interior(float **A, float **Anew, int local_rows, int local_cols, 
                     int start_row, int end_row, int start_col, int end_col) {
    // Compute interior points that don't depend on ghost data
    for (int i = start_row; i <= end_row; i++) {
        for (int j = start_col; j <= end_col; j++) {
            Anew[i][j] = (A[i][j + 1] + A[i][j - 1] + A[i - 1][j] + A[i + 1][j]) * 0.25f;
        }
    }
}

void compute_boundary_dependent(float **A, float **Anew, int local_rows, int local_cols) {
    // Compute points that depend on ghost data (boundary-dependent regions)
    
    // Top and bottom rows (depend on vertical neighbours)
    for (int j = 1; j <= local_cols; j++) {
        if (local_rows > 0) {
            Anew[1][j] = (A[1][j + 1] + A[1][j - 1] + A[0][j] + A[2][j]) * 0.25f;
            Anew[local_rows][j] = (A[local_rows][j + 1] + A[local_rows][j - 1] + 
                                  A[local_rows - 1][j] + A[local_rows + 1][j]) * 0.25f;
        }
    }
    
    // Left and right columns (depend on horizontal neighbours)
    for (int i = 2; i < local_rows; i++) {  // Skip corners already computed above
        Anew[i][1] = (A[i][2] + A[i][0] + A[i - 1][1] + A[i + 1][1]) * 0.25f;
        Anew[i][local_cols] = (A[i][local_cols + 1] + A[i][local_cols - 1] + 
                              A[i - 1][local_cols] + A[i + 1][local_cols]) * 0.25f;
    }
}

int main(int argc, char** argv) {
    int i, j, iter = 0;
    int n, m;
    float **A_local, **Anew_local;
    float *A_data, *Anew_data;
    const float tol = 1.0e-3f;
    float error = 1.0f;
    int iter_max = 100;
    double start_time, end_time;
    
    // MPI variables
    int rank, size;
    int p_rows, p_cols;  // 2D process grid dimensions
    int rank_row, rank_col;  // Position in 2D grid
    int local_rows, local_cols, total_local_rows, total_local_cols;
    int start_row, end_row, start_col, end_col;
    float local_error;
    MPI_Comm cart_comm;
    MPI_Request requests[8];  // Maximum 8 requests for non-blocking communication
    int req_count;
    
    // MPI setup
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get runtime arguments
    if (argc > 1) n = atoi(argv[1]);
    else n = 100;
    if (argc > 2) m = atoi(argv[2]);
    else m = 100;
    
    // Determine optimal 2D decomposition
    // Try to make the grid as square as possible
    p_rows = (int)sqrt(size);
    while (size % p_rows != 0) p_rows--;
    p_cols = size / p_rows;
    
    // Create 2D Cartesian topology
    int dims[2] = {p_rows, p_cols};
    int periods[2] = {0, 0};  // No periodic boundaries
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    
    // Get coordinates in the 2D grid
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    rank_row = coords[0];
    rank_col = coords[1];
    
    // Calculate local domain size
    int interior_rows = n - 2;  // Total interior rows
    int interior_cols = m - 2;  // Total interior columns
    
    local_rows = interior_rows / p_rows;
    local_cols = interior_cols / p_cols;
    
    // Handle remainder rows/columns
    if (rank_row < interior_rows % p_rows) local_rows++;
    if (rank_col < interior_cols % p_cols) local_cols++;
    
    // Calculate starting positions in global grid
    start_row = rank_row * (interior_rows / p_rows) + 1;  // +1 for boundary
    start_col = rank_col * (interior_cols / p_cols) + 1;  // +1 for boundary
    
    if (rank_row < interior_rows % p_rows) {
        start_row += rank_row;
    } else {
        start_row += interior_rows % p_rows;
    }
    
    if (rank_col < interior_cols % p_cols) {
        start_col += rank_col;
    } else {
        start_col += interior_cols % p_cols;
    }
    
    end_row = start_row + local_rows - 1;
    end_col = start_col + local_cols - 1;
    
    // Allocate local arrays with ghost cells
    total_local_rows = local_rows + 2;  // +2 for ghost rows
    total_local_cols = local_cols + 2;  // +2 for ghost columns
    
    // Optimised memory allocation - contiguous layout for better cache performance
    A_data = malloc(total_local_rows * total_local_cols * sizeof(float));
    Anew_data = malloc(total_local_rows * total_local_cols * sizeof(float));
    A_local = malloc(total_local_rows * sizeof(float*));
    Anew_local = malloc(total_local_rows * sizeof(float*));
    
    // Set up row pointers for 2D access
    for (i = 0; i < total_local_rows; i++) {
        A_local[i] = A_data + i * total_local_cols;
        Anew_local[i] = Anew_data + i * total_local_cols;
    }
    
    // Initialize local domain
    for (i = 0; i < total_local_rows; i++) {
        for (j = 0; j < total_local_cols; j++) {
            A_local[i][j] = 0.0f;
            Anew_local[i][j] = 0.0f;
        }
    }
    
    // Set boundary conditions for processes that own global boundaries
    if (rank_row == 0) {  // Top boundary
        for (j = 1; j <= local_cols; j++) {
            A_local[1][j] = 0.0f;
        }
    }
    if (rank_row == p_rows - 1) {  // Bottom boundary
        for (j = 1; j <= local_cols; j++) {
            A_local[local_rows][j] = 0.0f;
        }
    }
    if (rank_col == 0) {  // Left boundary
        for (i = 1; i <= local_rows; i++) {
            int global_i = start_row + i - 1;
            A_local[i][1] = sinf(PI * global_i / (n - 1));
        }
    }
    if (rank_col == p_cols - 1) {  // Right boundary
        for (i = 1; i <= local_rows; i++) {
            int global_i = start_row + i - 1;
            A_local[i][local_cols] = sinf(PI * global_i / (n - 1)) * expf(-PI);
        }
    }
    
    // Copy initial values to Anew_local
    for (i = 0; i < total_local_rows; i++) {
        for (j = 0; j < total_local_cols; j++) {
            Anew_local[i][j] = A_local[i][j];
        }
    }
    
    // START TIMER
    start_time = MPI_Wtime();
    
    // Main iteration loop
    while (error > tol && iter < iter_max) {
        // Start non-blocking boundary exchange
        exchange_boundaries_non_blocking(A_local, local_rows, local_cols, 
                                       rank_row, rank_col, p_rows, p_cols, 
                                       cart_comm, requests, &req_count);
        
        // Compute interior points while communication is in progress
        // These don't depend on ghost data, so can be computed immediately
        if (local_rows > 2 && local_cols > 2) {
            compute_interior(A_local, Anew_local, local_rows, local_cols, 2, local_rows - 1, 2, local_cols - 1);
        }
        
        // Wait for all communications to complete
        if (req_count > 0) {
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        }
        
        // Unpack column data from temporary buffers
        unpack_column_data(A_local, local_rows, local_cols, rank_row, rank_col, p_rows, p_cols, cart_comm);
        
        // Compute boundary-dependent points
        compute_boundary_dependent(A_local, Anew_local, local_rows, local_cols);
        
        // Compute local error
        local_error = 0.0f;
        for (i = 1; i <= local_rows; i++) {
            for (j = 1; j <= local_cols; j++) {
                float diff = fabsf(Anew_local[i][j] - A_local[i][j]);
                if (diff > local_error) local_error = diff;
            }
        }
        
        // Reduce to find global maximum error
        MPI_Allreduce(&local_error, &error, 1, MPI_FLOAT, MPI_MAX, cart_comm);
        
        // Update A_local with Anew_local values
        // Optimised memory copy using the contiguous layout
        memcpy(A_data, Anew_data, total_local_rows * total_local_cols * sizeof(float));
        
        iter++;
    }
    
    // STOP TIMER
    end_time = MPI_Wtime();
    
    // Output timing - only rank 0 prints
    if (rank == 0) {
        printf("%f\n", end_time - start_time);
    }
    
    // Clean up
    free(A_data);
    free(Anew_data);
    free(A_local);
    free(Anew_local);
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}