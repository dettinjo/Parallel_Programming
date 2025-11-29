/*
 * Hybrid MPI+OpenMP Laplace Solver V3.1 - Targeted Performance Fix
 * 
 * Compilation:
 *   mpicc -fopenmp -O3 laplace_mpi_v3_1_fix.c -o laplace_v31 -lm -lgomp
 * 
 * Key fixes:
 * - Conservative OpenMP usage for single-node execution
 * - Better decomposition thresholds
 * - Reduced overhead in critical sections
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads(void) { return 1; }
static inline void omp_set_num_threads(int num_threads) { (void)num_threads; }
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_num_threads(void) { return 1; }
#endif

#define PI (3.1415926535897932384626)
#define MIN_BLOCK_SIZE 32
#define ERROR_CHECK_FREQ 3
#define BOUNDARY_EXCHANGE_FREQ 1
#define MIN_OMP_WORK 64  // Minimum work size to justify OpenMP overhead

typedef struct {
    int is_inter_node;
    int processes_per_node;
    int node_rank;
    int nodes_used;
    MPI_Comm node_comm;
    MPI_Comm cart_comm;
} topology_info_t;

void detect_topology(topology_info_t *topo, int rank, int size) {
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    
    MPI_Get_processor_name(processor_name, &name_len);
    
    // Create node-local communicator
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &topo->node_comm);
    MPI_Comm_rank(topo->node_comm, &topo->node_rank);
    MPI_Comm_size(topo->node_comm, &topo->processes_per_node);
    
    topo->nodes_used = size / topo->processes_per_node;
    if (size % topo->processes_per_node != 0) topo->nodes_used++;
    
    topo->is_inter_node = (topo->nodes_used > 1);
}

void choose_decomposition_conservative(int n, int m, int size, topology_info_t *topo, 
                                     int *p_rows, int *p_cols, int *use_2d, int *use_omp) {
    // Much more conservative decomposition strategy
    
    int total_points = (n-2) * (m-2);
    int points_per_process = total_points / size;
    
    // Default: no OpenMP, 1D decomposition
    *use_omp = 0;
    *use_2d = 0;
    *p_rows = size;
    *p_cols = 1;
    
    // Only enable OpenMP for larger problems with sufficient work per process
    if (points_per_process >= MIN_OMP_WORK * MIN_OMP_WORK) {
        *use_omp = 1;
    }
    
    // For small problems or many processes on single node, stick to 1D
    if (points_per_process < MIN_BLOCK_SIZE * MIN_BLOCK_SIZE || 
        (!topo->is_inter_node && size > 8)) {
        return;  // Keep 1D, potentially with OpenMP
    }
    
    // Only use 2D for larger problems across multiple nodes
    if (topo->is_inter_node && points_per_process >= MIN_BLOCK_SIZE * MIN_BLOCK_SIZE * 4) {
        *use_2d = 1;
        *p_rows = (int)sqrt(size);
        while (size % *p_rows != 0) (*p_rows)--;
        *p_cols = size / *p_rows;
    }
}

void exchange_boundaries_optimized(float **A, int local_rows, int local_cols, 
                                  int use_2d, MPI_Comm comm, MPI_Request *requests, int *req_count) {
    *req_count = 0;
    
    if (!use_2d) {
        // 1D decomposition - only vertical neighbours
        int up, down;
        MPI_Cart_shift(comm, 0, 1, &up, &down);
        
        if (up != MPI_PROC_NULL) {
            MPI_Isend(A[1], local_cols, MPI_FLOAT, up, 1, comm, &requests[(*req_count)++]);
            MPI_Irecv(A[0], local_cols, MPI_FLOAT, up, 2, comm, &requests[(*req_count)++]);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Isend(A[local_rows], local_cols, MPI_FLOAT, down, 2, comm, &requests[(*req_count)++]);
            MPI_Irecv(A[local_rows + 1], local_cols, MPI_FLOAT, down, 1, comm, &requests[(*req_count)++]);
        }
    } else {
        // 2D decomposition - simplified for better performance
        int up, down, left, right;
        MPI_Cart_shift(comm, 0, 1, &up, &down);
        MPI_Cart_shift(comm, 1, 1, &left, &right);
        
        // Vertical exchange (always contiguous)
        if (up != MPI_PROC_NULL) {
            MPI_Isend(A[1], local_cols, MPI_FLOAT, up, 1, comm, &requests[(*req_count)++]);
            MPI_Irecv(A[0], local_cols, MPI_FLOAT, up, 2, comm, &requests[(*req_count)++]);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Isend(A[local_rows], local_cols, MPI_FLOAT, down, 2, comm, &requests[(*req_count)++]);
            MPI_Irecv(A[local_rows + 1], local_cols, MPI_FLOAT, down, 1, comm, &requests[(*req_count)++]);
        }
        
        // Horizontal exchange - pack into temporary buffers for simplicity
        static float *left_send = NULL, *left_recv = NULL, *right_send = NULL, *right_recv = NULL;
        static int buffer_size = 0;
        
        if (local_rows + 2 > buffer_size) {
            if (left_send) { free(left_send); free(left_recv); free(right_send); free(right_recv); }
            buffer_size = local_rows + 2;
            left_send = malloc(buffer_size * sizeof(float));
            left_recv = malloc(buffer_size * sizeof(float));
            right_send = malloc(buffer_size * sizeof(float));
            right_recv = malloc(buffer_size * sizeof(float));
        }
        
        if (left != MPI_PROC_NULL) {
            for (int i = 0; i < local_rows + 2; i++) left_send[i] = A[i][1];
            MPI_Isend(left_send, local_rows + 2, MPI_FLOAT, left, 3, comm, &requests[(*req_count)++]);
            MPI_Irecv(left_recv, local_rows + 2, MPI_FLOAT, left, 4, comm, &requests[(*req_count)++]);
        }
        if (right != MPI_PROC_NULL) {
            for (int i = 0; i < local_rows + 2; i++) right_send[i] = A[i][local_cols];
            MPI_Isend(right_send, local_rows + 2, MPI_FLOAT, right, 4, comm, &requests[(*req_count)++]);
            MPI_Irecv(right_recv, local_rows + 2, MPI_FLOAT, right, 3, comm, &requests[(*req_count)++]);
        }
    }
}

void unpack_columns_if_2d(float **A, int local_rows, int local_cols, int use_2d, MPI_Comm comm) {
    if (!use_2d) return;
    
    int up, down, left, right;
    MPI_Cart_shift(comm, 0, 1, &up, &down);
    MPI_Cart_shift(comm, 1, 1, &left, &right);
    
    static float *left_recv = NULL, *right_recv = NULL;
    static int buffer_size = 0;
    
    if (local_rows + 2 > buffer_size) {
        if (left_recv) { free(left_recv); free(right_recv); }
        buffer_size = local_rows + 2;
        left_recv = malloc(buffer_size * sizeof(float));
        right_recv = malloc(buffer_size * sizeof(float));
    }
    
    if (left != MPI_PROC_NULL) {
        for (int i = 0; i < local_rows + 2; i++) A[i][0] = left_recv[i];
    }
    if (right != MPI_PROC_NULL) {
        for (int i = 0; i < local_rows + 2; i++) A[i][local_cols + 1] = right_recv[i];
    }
}

void compute_stencil_adaptive(float **A, float **Anew, int start_i, int end_i, int start_j, int end_j, int use_omp) {
    int work_size = (end_i - start_i + 1) * (end_j - start_j + 1);
    
    if (use_omp && work_size >= MIN_OMP_WORK) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = start_i; i <= end_i; i++) {
            for (int j = start_j; j <= end_j; j++) {
                Anew[i][j] = (A[i][j + 1] + A[i][j - 1] + A[i - 1][j] + A[i + 1][j]) * 0.25f;
            }
        }
    } else {
        // Serial computation for small work sizes
        for (int i = start_i; i <= end_i; i++) {
            for (int j = start_j; j <= end_j; j++) {
                Anew[i][j] = (A[i][j + 1] + A[i][j - 1] + A[i - 1][j] + A[i + 1][j]) * 0.25f;
            }
        }
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
    int rank, size, provided;
    topology_info_t topo;
    int p_rows, p_cols, use_2d, use_omp;
    int rank_row, rank_col;
    int local_rows, local_cols, total_local_rows, total_local_cols;
    int start_row, end_row, start_col, end_col;
    float local_error;
    MPI_Request requests[8];
    int req_count;
    
    // Initialize MPI
#ifdef _OPENMP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#else
    MPI_Init(&argc, &argv);
    provided = MPI_THREAD_SINGLE;
#endif
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Detect topology
    detect_topology(&topo, rank, size);
    
    // Get runtime arguments
    if (argc > 1) n = atoi(argv[1]);
    else n = 100;
    if (argc > 2) m = atoi(argv[2]);
    else m = 100;
    
    // Choose decomposition strategy - much more conservative
    choose_decomposition_conservative(n, m, size, &topo, &p_rows, &p_cols, &use_2d, &use_omp);
    
    // Set OpenMP threads very conservatively
    if (use_omp) {
        int max_threads = omp_get_max_threads();
        int num_threads = 1;
        
        // Only use multiple threads if we have sufficient work and are not competing heavily
        if (!topo.is_inter_node && size <= 4) {
            num_threads = (max_threads + size - 1) / size;  // Share threads among processes
        } else if (topo.is_inter_node) {
            num_threads = max_threads / topo.processes_per_node;  // Conservative sharing
        }
        
        if (num_threads < 1) num_threads = 1;
        if (num_threads > 4) num_threads = 4;  // Cap at 4 threads to avoid overhead
        
        omp_set_num_threads(num_threads);
    } else {
        omp_set_num_threads(1);
    }
    
    // Create communicator
    if (use_2d) {
        int dims[2] = {p_rows, p_cols};
        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &topo.cart_comm);
        
        int coords[2];
        MPI_Cart_coords(topo.cart_comm, rank, 2, coords);
        rank_row = coords[0];
        rank_col = coords[1];
    } else {
        int dims[1] = {p_rows};
        int periods[1] = {0};
        MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 1, &topo.cart_comm);
        
        rank_row = rank;
        rank_col = 0;
        p_cols = 1;
    }
    
    // Calculate local domain size (same logic as V3)
    int interior_rows = n - 2;
    int interior_cols = m - 2;
    
    if (use_2d) {
        local_rows = interior_rows / p_rows;
        local_cols = interior_cols / p_cols;
        if (rank_row < interior_rows % p_rows) local_rows++;
        if (rank_col < interior_cols % p_cols) local_cols++;
        
        start_row = rank_row * (interior_rows / p_rows) + 1;
        start_col = rank_col * (interior_cols / p_cols) + 1;
        if (rank_row < interior_rows % p_rows) start_row += rank_row;
        else start_row += interior_rows % p_rows;
        if (rank_col < interior_cols % p_cols) start_col += rank_col;
        else start_col += interior_cols % p_cols;
    } else {
        local_rows = interior_rows / p_rows;
        local_cols = interior_cols;
        if (rank_row < interior_rows % p_rows) local_rows++;
        
        start_row = rank_row * (interior_rows / p_rows) + 1;
        start_col = 1;
        if (rank_row < interior_rows % p_rows) start_row += rank_row;
        else start_row += interior_rows % p_rows;
    }
    
    end_row = start_row + local_rows - 1;
    end_col = start_col + local_cols - 1;
    
    // Allocate memory
    total_local_rows = local_rows + 2;
    total_local_cols = local_cols + 2;
    
    A_data = malloc(total_local_rows * total_local_cols * sizeof(float));
    Anew_data = malloc(total_local_rows * total_local_cols * sizeof(float));
    A_local = malloc(total_local_rows * sizeof(float*));
    Anew_local = malloc(total_local_rows * sizeof(float*));
    
    for (i = 0; i < total_local_rows; i++) {
        A_local[i] = A_data + i * total_local_cols;
        Anew_local[i] = Anew_data + i * total_local_cols;
    }
    
    // Initialize arrays and boundary conditions (same as V3)
    memset(A_data, 0, total_local_rows * total_local_cols * sizeof(float));
    memset(Anew_data, 0, total_local_rows * total_local_cols * sizeof(float));
    
    // Set boundary conditions
    if ((use_2d && rank_row == 0) || (!use_2d && rank == 0)) {
        for (j = 1; j <= local_cols; j++) A_local[1][j] = 0.0f;
    }
    if ((use_2d && rank_row == p_rows - 1) || (!use_2d && rank == p_rows - 1)) {
        for (j = 1; j <= local_cols; j++) A_local[local_rows][j] = 0.0f;
    }
    if (!use_2d || rank_col == 0) {
        for (i = 1; i <= local_rows; i++) {
            int global_i = start_row + i - 1;
            A_local[i][1] = sinf(PI * global_i / (n - 1));
        }
    }
    if (!use_2d || rank_col == p_cols - 1) {
        for (i = 1; i <= local_rows; i++) {
            int global_i = start_row + i - 1;
            A_local[i][local_cols] = sinf(PI * global_i / (n - 1)) * expf(-PI);
        }
    }
    
    memcpy(Anew_data, A_data, total_local_rows * total_local_cols * sizeof(float));
    
    // START TIMER
    start_time = MPI_Wtime();
    
    // Main iteration loop - simplified
    while (error > tol && iter < iter_max) {
        // Exchange boundaries
        exchange_boundaries_optimized(A_local, local_rows, local_cols, use_2d, 
                                    topo.cart_comm, requests, &req_count);
        
        // Compute interior (most of the work)
        if (local_rows > 2 && local_cols > 2) {
            compute_stencil_adaptive(A_local, Anew_local, 2, local_rows - 1, 2, local_cols - 1, use_omp);
        }
        
        // Wait for communication
        if (req_count > 0) {
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        }
        
        // Unpack column data if needed
        unpack_columns_if_2d(A_local, local_rows, local_cols, use_2d, topo.cart_comm);
        
        // Compute boundaries
        if (local_rows > 0) {
            compute_stencil_adaptive(A_local, Anew_local, 1, 1, 1, local_cols, 0);  // Top row, no OpenMP
            compute_stencil_adaptive(A_local, Anew_local, local_rows, local_rows, 1, local_cols, 0);  // Bottom row
        }
        if (use_2d && local_rows > 2) {
            compute_stencil_adaptive(A_local, Anew_local, 2, local_rows - 1, 1, 1, 0);  // Left column
            compute_stencil_adaptive(A_local, Anew_local, 2, local_rows - 1, local_cols, local_cols, 0);  // Right column
        }
        
        // Compute error less frequently
        if (iter % ERROR_CHECK_FREQ == 0) {
            local_error = 0.0f;
            for (i = 1; i <= local_rows; i++) {
                for (j = 1; j <= local_cols; j++) {
                    float diff = fabsf(Anew_local[i][j] - A_local[i][j]);
                    if (diff > local_error) local_error = diff;
                }
            }
            MPI_Allreduce(&local_error, &error, 1, MPI_FLOAT, MPI_MAX, topo.cart_comm);
        }
        
        // Efficient array swap
        float **temp = A_local; A_local = Anew_local; Anew_local = temp;
        float *temp_data = A_data; A_data = Anew_data; Anew_data = temp_data;
        
        iter++;
    }
    
    // STOP TIMER
    end_time = MPI_Wtime();
    
    // Output timing
    if (rank == 0) {
        printf("%f\n", end_time - start_time);
    }
    
    // Cleanup
    free(A_data);
    free(Anew_data);
    free(A_local);
    free(Anew_local);
    
    if (topo.cart_comm != MPI_COMM_NULL) MPI_Comm_free(&topo.cart_comm);
    MPI_Comm_free(&topo.node_comm);
    MPI_Finalize();
    return 0;
}