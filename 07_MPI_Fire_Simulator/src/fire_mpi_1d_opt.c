/*
 * Simplified simulation of fire extinguishing
 *
 * v1.4
 *
 * (c) 2019 Arturo Gonzalez Escribano
 *
 * MPI parallel version
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<float.h>
#include<sys/time.h>
#include<mpi.h> // MPI: Include MPI header

/* * Replaced sequential timer with MPI_Wtime,
 * which is called from within main()
 */

#define RADIUS_TYPE_1       3
#define RADIUS_TYPE_2_3     9
#define THRESHOLD   0.1f

/* Structure to store data of an extinguishing team */
typedef struct {
    int x,y;
    int type;
    int target;
} Team;

/* Structure to store data of a fire focal point */
typedef struct {
    int x,y;
    int start;
    int heat;
    int active; // States: 0 Not yet activated; 1 Active; 2 Deactivated by a team
} FocalPoint;


/* Macro function to simplify accessing with two coordinates to a flattened array */
/* NOTE: This macro is for the *global* surface, used only by rank 0 for setup/output */
#define accessMat( arr, exp1, exp2 )    arr[ (exp1) * columns + (exp2 ) ]


/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
    fprintf(stderr,"Usage: %s <config_file> | <command_line_args>\n", program_name );
    fprintf(stderr,"\t<config_file> ::= -f <file_name>\n");
    fprintf(stderr,"\t<command_line_args> ::= <rows> <columns> <maxIter> <numTeams> [ <teamX> <teamY> <teamType> ... ] <numFocalPoints> [ <focalX> <focalY> <focalStart> <focalTemperature> ... ]\n");
    fprintf(stderr,"\n");
}

#ifdef DEBUG
/* * Function: Print the current state of the simulation 
 * WARNING: This function is not parallelized and will only work
 * if called by rank 0 *after* a full surface gather.
 */
void print_status( int iteration, int rows, int columns, float *surface, int num_teams, Team *teams, int num_focal, FocalPoint *focal, float global_residual ) {
    int i,j;

    printf("Iteration: %d\n", iteration );
    printf("+");
    for( j=0; j<columns; j++ ) printf("---");
    printf("+\n");
    for( i=0; i<rows; i++ ) {
        printf("|");
        for( j=0; j<columns; j++ ) {
            char symbol;
            if ( accessMat( surface, i, j ) >= 1000 ) symbol = '*';
            else if ( accessMat( surface, i, j ) >= 100 ) symbol = '0' + (int)(accessMat( surface, i, j )/100);
            else if ( accessMat( surface, i, j ) >= 50 ) symbol = '+';
            else if ( accessMat( surface, i, j ) >= 25 ) symbol = '.';
            else symbol = '0';

            int t;
            int flag_team = 0;
            for( t=0; t<num_teams; t++ ) 
                if ( teams[t].x == i && teams[t].y == j ) { flag_team = 1; break; }
            if ( flag_team ) printf("[%c]", symbol );
            else {
                int f;
                int flag_focal = 0;
                for( f=0; f<num_focal; f++ ) 
                    if ( focal[f].x == i && focal[f].y == j && focal[f].active == 1 ) { flag_focal = 1; break; }
                if ( flag_focal ) printf("(%c)", symbol );
                else printf(" %c ", symbol );
            }
        }
        printf("|\n");
    }
    printf("+");
    for( j=0; j<columns; j++ ) printf("---");
    printf("+\n");
    printf("Global residual: %f\n\n", global_residual);
}
#endif



/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    int i,j,t;

    // MPI: MPI Environment
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Simulation data
    int rows, columns, max_iter;
    float *surface, *surfaceCopy; // MPI: These are now global pointers, only used by rank 0
    int num_teams, num_focal;
    Team *teams;
    FocalPoint *focal;

    /* 1. Read simulation arguments */
    /* 1.1. Check minimum number of arguments */
    /* MPI: Only rank 0 prints usage errors */
    if (argc<2) {
        if (rank == 0) {
            fprintf(stderr,"-- Error in arguments: No arguments\n");
            show_usage( argv[0] );
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int read_from_file = ! strcmp( argv[1], "-f" );
    /* 1.2. Read configuration from file */
    /* MPI: All processes read the file to replicate teams and focal data */
    if ( read_from_file ) {
        /* 1.2.1. Open file */
        if (argc<3) {
            if (rank == 0) {
                fprintf(stderr,"-- Error in arguments: file-name argument missing\n");
                show_usage( argv[0] );
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        FILE *args = fopen( argv[2], "r" );
        if ( args == NULL ) {
            if (rank == 0) fprintf(stderr,"-- Error in file: not found: %s\n", argv[2]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }   

        /* 1.2.2. Read surface and maximum number of iterations */
        int ok;
        ok = fscanf(args, "%d %d %d", &rows, &columns, &max_iter);
        if ( ok != 3 ) {
            if (rank == 0) fprintf(stderr,"-- Error in file: reading rows, columns, max_iter from file: %s\n", argv[2]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* 1.2.3. Teams information */
        ok = fscanf(args, "%d", &num_teams );
        if ( ok != 1 ) {
            if (rank == 0) fprintf(stderr,"-- Error file, reading num_teams from file: %s\n", argv[2]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
        if ( teams == NULL ) {
            if (rank == 0) fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for( i=0; i<num_teams; i++ ) {
            ok = fscanf(args, "%d %d %d", &teams[i].x, &teams[i].y, &teams[i].type);
            if ( ok != 3 ) {
                if (rank == 0) fprintf(stderr,"-- Error in file: reading team %d from file: %s\n", i, argv[2]);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

        /* 1.2.4. Focal points information */
        ok = fscanf(args, "%d", &num_focal );
        if ( ok != 1 ) {
            if (rank == 0) fprintf(stderr,"-- Error in file: reading num_focal from file: %s\n", argv[2]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
        if ( focal == NULL ) {
            if (rank == 0) fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for( i=0; i<num_focal; i++ ) {
            ok = fscanf(args, "%d %d %d %d", &focal[i].x, &focal[i].y, &focal[i].start, &focal[i].heat);
            if ( ok != 4 ) {
                if (rank == 0) fprintf(stderr,"-- Error in file: reading focal point %d from file: %s\n", i, argv[2]);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            focal[i].active = 0;
        }
        fclose(args); // Close the file after reading
    }
    /* 1.3. Read configuration from arguments */
    else {
        /* MPI: This is complex to parallelize, so we'll enforce file-based reading for MPI */
        if (rank == 0) {
            fprintf(stderr, "-- Error: Command-line argument parsing is not supported for MPI.\n");
            fprintf(stderr, "-- Please use the file-based configuration: %s -f <filename>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* MPI: Sanity check for row divisibility */
    if (rows % nprocs != 0) {
        if (rank == 0) {
            fprintf(stderr, "-- Error: Number of rows (%d) must be perfectly divisible by the number of processes (%d).\n", rows, nprocs);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }


#ifdef DEBUG
    /* 1.4. Print arguments */
    if (rank == 0) {
        printf("Arguments, Rows: %d, Columns: %d, max_iter: %d, threshold: %f\n", rows, columns, max_iter, THRESHOLD);
        printf("Arguments, Teams: %d, Focal points: %d\n", num_teams, num_focal );
        for( i=0; i<num_teams; i++ ) {
            printf("\tTeam %d, position (%d,%d), type: %d\n", i, teams[i].x, teams[i].y, teams[i].type );
        }
        for( i=0; i<num_focal; i++ ) {
            printf("\tFocal_point %d, position (%d,%d), start time: %d, temperature: %d\n", i, 
            focal[i].x,
            focal[i].y,
            focal[i].start,
            focal[i].heat );
        }
        printf("\nLEGEND:\n");
        printf("\t( ) : Focal point\n");
        printf("\t[ ] : Team position\n");
        printf("\t0-9 : Temperature value in hundreds of degrees\n");
        printf("\t* : Temperature equal or higher than 1000 degrees\n\n");
    }
#endif // DEBUG

    /* 2. Start global timer */
    /* MPI: Use MPI_Wtime and a barrier for synchronization */
    double tstart, ttotal;
    MPI_Barrier(MPI_COMM_WORLD);
    tstart = MPI_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

    /* 3. Initialize surfaces */
    /* MPI: Macro for accessing the *local* surface buffers */
    #define accessLocalMat( arr, exp1, exp2 )    arr[ (exp1) * columns + (exp2) ]

    /* MPI: Local surface data setup (Dynamic remainder handling included) */
    int base_rows = rows / nprocs;
    int remainder = rows % nprocs;
    
    int local_rows = (rank < remainder) ? base_rows + 1 : base_rows;

    int first_row;
    if (rank < remainder) {
        first_row = rank * (base_rows + 1);
    } else {
        first_row = remainder * (base_rows + 1) + (rank - remainder) * base_rows;
    }
    int last_row = first_row + local_rows - 1;

    int local_alloc_rows = local_rows + 2; // +2 for Ghost Rows

    /* MPI: Pointers for local surfaces */
    float *local_surface, *local_surfaceCopy;

    /* MPI: Allocate local surfaces */
    local_surface = (float *)malloc( sizeof(float) * (size_t)local_alloc_rows * (size_t)columns );
    local_surfaceCopy = (float *)malloc( sizeof(float) * (size_t)local_alloc_rows * (size_t)columns );

    /* MPI: Rank 0 Allocates global surface */
    if (rank == 0) {
        surface = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
        if ( surface == NULL ) {
            fprintf(stderr,"-- Error allocating global surface for gather\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    } else {
        surface = NULL; 
    }
    surfaceCopy = NULL; 

    if ( local_surface == NULL || local_surfaceCopy == NULL ) {
        fprintf(stderr,"-- Error allocating: local surface structures on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    /* MPI: Initialization */
    for( i=0; i<local_alloc_rows * columns; i++ ) {
        local_surface[i] = 0.0f;
        local_surfaceCopy[i] = 0.0f;
    }

    /* 4. Simulation */
    int iter;
    int flag_stability = 0;
    int first_activation = 0;

    /* MPI: Setup for Async Comm */
    MPI_Request requests[4];
    MPI_Status statuses[4];
    int up_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down_neighbor = (rank == nprocs - 1) ? MPI_PROC_NULL : rank + 1;

    for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {

        /* 4.1. Activate focal points */
        int num_deactivated = 0;
        for( i=0; i<num_focal; i++ ) {
            if ( focal[i].start == iter ) {
                focal[i].active = 1;
                if ( ! first_activation ) first_activation = 1;
            }
            if ( focal[i].active == 2 ) num_deactivated++;
        }

        /* 4.2. Propagate heat */
        float local_residual = 0.0f;
        float global_residual = 0.0f;
        int step;
        
        for( step=0; step<10; step++ )  {
            /* 4.2.1. Update heat on active focal points */
            for( i=0; i<num_focal; i++ ) {
                if ( focal[i].active != 1 ) continue;
                int x = focal[i].x;
                int y = focal[i].y;
                if ( x < 0 || x > rows-1 || y < 0 || y > columns-1 ) continue;

                if ( x >= first_row && x <= last_row ) {
                    int local_i = x - first_row + 1; 
                    accessLocalMat( local_surface, local_i, y ) = focal[i].heat;
                }
            }

            /* OPTIMIZATION 1: Async Ghosts */
            MPI_Irecv( &accessLocalMat(local_surface, 0, 0), columns, MPI_FLOAT, up_neighbor, 0, MPI_COMM_WORLD, &requests[0] );
            MPI_Irecv( &accessLocalMat(local_surface, local_rows+1, 0), columns, MPI_FLOAT, down_neighbor, 0, MPI_COMM_WORLD, &requests[1] );
            MPI_Isend( &accessLocalMat(local_surface, 1, 0), columns, MPI_FLOAT, up_neighbor, 0, MPI_COMM_WORLD, &requests[2] );
            MPI_Isend( &accessLocalMat(local_surface, local_rows, 0), columns, MPI_FLOAT, down_neighbor, 0, MPI_COMM_WORLD, &requests[3] );

            /*
             * OPTIMIZATION 2 & 4: Overlapping Computation + LOOP FUSION
             * We calculate the update AND the residual difference in one pass.
             */
            
            local_residual = 0.0f; // Reset accumulator for this step
            
            // Compute Inner Rows (Fusion)
            int start_inner = 2;
            int end_inner = local_rows - 1;
            
            if (local_rows >= 3) { 
                for( i=start_inner; i<=end_inner; i++ ) {
                    for( j=1; j<columns-1; j++ ) {
                        float val = ( 
                            accessLocalMat( local_surface, i-1, j ) +
                            accessLocalMat( local_surface, i+1, j ) +
                            accessLocalMat( local_surface, i, j-1 ) +
                            accessLocalMat( local_surface, i, j+1 ) ) / 4.0f;
                        
                        accessLocalMat( local_surfaceCopy, i, j ) = val;
                        
                        /* Fused Residual Check */
                        float diff = fabs( val - accessLocalMat( local_surface, i, j ) );
                        if (diff > local_residual) local_residual = diff;
                    }
                }
            }

            /* Wait for Ghosts */
            MPI_Waitall(4, requests, statuses);

            /* Compute Boundary Rows (Fusion) */
            int boundary_rows[2] = {1, local_rows};
            for( int b=0; b<2; b++ ) {
                int r = boundary_rows[b];
                // Skip global boundaries
                if (rank == 0 && r == 1) continue;
                if (rank == nprocs-1 && r == local_rows) continue;
                // Skip already computed
                if (r >= start_inner && r <= end_inner && local_rows >= 3) continue; 
                
                if (r >= 1 && r <= local_rows) {
                    for( j=1; j<columns-1; j++ ) {
                        float val = ( 
                            accessLocalMat( local_surface, r-1, j ) +
                            accessLocalMat( local_surface, r+1, j ) +
                            accessLocalMat( local_surface, r, j-1 ) +
                            accessLocalMat( local_surface, r, j+1 ) ) / 4.0f;
                            
                        accessLocalMat( local_surfaceCopy, r, j ) = val;
                        
                        /* Fused Residual Check */
                        float diff = fabs( val - accessLocalMat( local_surface, r, j ) );
                        if (diff > local_residual) local_residual = diff;
                    }
                }
            }

            /* OPTIMIZATION 3: Pointer Swap */
            float *swap_tmp = local_surface;
            local_surface = local_surfaceCopy;
            local_surfaceCopy = swap_tmp;
        }
        
        MPI_Allreduce(&local_residual, &global_residual, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        if( num_deactivated == num_focal && global_residual < THRESHOLD ) flag_stability = 1;

        /* 4.3. Move teams */
        for( t=0; t<num_teams; t++ ) {
            float distance = FLT_MAX;
            int target = -1;
            for( j=0; j<num_focal; j++ ) {
                if ( focal[j].active != 1 ) continue; 
                float dx = focal[j].x - teams[t].x;
                float dy = focal[j].y - teams[t].y;
                float local_distance = sqrtf( dx*dx + dy*dy );
                if ( local_distance < distance ) {
                    distance = local_distance;
                    target = j;
                }
            }
            teams[t].target = target;
            if ( target == -1 ) continue; 

            if ( teams[t].type == 1 ) { 
                if ( focal[target].x < teams[t].x ) teams[t].x--;
                if ( focal[target].x > teams[t].x ) teams[t].x++;
                if ( focal[target].y < teams[t].y ) teams[t].y--;
                if ( focal[target].y > teams[t].y ) teams[t].y++;
            }
            else if ( teams[t].type == 2 ) { 
                if ( focal[target].y < teams[t].y ) teams[t].y--;
                else if ( focal[target].y > teams[t].y ) teams[t].y++;
                else if ( focal[target].x < teams[t].x ) teams[t].x--;
                else if ( focal[target].x > teams[t].x ) teams[t].x++;
            }
            else {
                if ( focal[target].x < teams[t].x ) teams[t].x--;
                else if ( focal[target].x > teams[t].x ) teams[t].x++;
                else if ( focal[target].y < teams[t].y ) teams[t].y--;
                else if ( focal[target].y > teams[t].y ) teams[t].y++;
            }
        }

        /* 4.4. Team actions */
        for( t=0; t<num_teams; t++ ) {
            int target = teams[t].target;
            if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
                && focal[target].active == 1 )
                focal[target].active = 2;

            /* OPTIMIZATION 5: Optimized Bounding Box Intersection for Teams */
            int radius;
            if ( teams[t].type == 1 ) radius = RADIUS_TYPE_1;
            else radius = RADIUS_TYPE_2_3;
            
            // Calculate intersection of Team box and Local domain
            int start_i = teams[t].x - radius;
            int end_i = teams[t].x + radius;
            
            // Clamp to Global Surface
            if (start_i < 1) start_i = 1; 
            if (end_i > rows-2) end_i = rows-2;
            
            // Clamp to Local Domain
            if (start_i < first_row) start_i = first_row;
            if (end_i > last_row) end_i = last_row;
            
            // Only iterate if there is an overlap
            if (start_i <= end_i) {
                for( i=start_i; i<=end_i; i++ ) {
                    int local_i = i - first_row + 1;
                    
                    for( j=teams[t].y-radius; j<=teams[t].y+radius; j++ ) {
                        if ( j<1 || j>=columns-1 ) continue; 

                        float dx = teams[t].x - i;
                        float dy = teams[t].y - j;
                        float distance = sqrtf( dx*dx + dy*dy );
                        if ( distance <= radius ) {
                            accessLocalMat( local_surface, local_i, j ) = accessLocalMat( local_surface, local_i, j ) * ( 1 - 0.25 ); 
                        }
                    }
                }
            }
        }
    }

    /* MPI: Gather (Using Gatherv for safety with dynamic decomposition) */
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        recvcounts = (int *)malloc(nprocs * sizeof(int));
        displs = (int *)malloc(nprocs * sizeof(int));
        int current_disp = 0;
        for (int r = 0; r < nprocs; r++) {
            int r_rows = (r < remainder) ? base_rows + 1 : base_rows;
            recvcounts[r] = r_rows * columns;
            displs[r] = current_disp;
            current_disp += recvcounts[r];
        }
    }

    MPI_Gatherv(&accessLocalMat(local_surface, 1, 0), local_rows * columns, MPI_FLOAT,
                surface, recvcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
    
    free( local_surface );
    free( local_surfaceCopy );
    
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

    /* 5. Stop global time */
    /* MPI: Barrier for synchronization before stopping timer */
    MPI_Barrier(MPI_COMM_WORLD);
    ttotal = MPI_Wtime() - tstart;

    /* 6. Output for leaderboard */
    /* MPI: Only rank 0 prints the final results */
    if (rank == 0) {
        printf("\n");
        /* 6.1. Total computation time */
        printf("Time: %lf\n", ttotal );
        /* 6.2. Results: Number of iterations, residual heat on the focal points */
        /* MPI: Must read from the gathered 'surface' */
        printf("Result: %d", iter);
        for (i=0; i<num_focal; i++) {
            int x = focal[i].x;
            int y = focal[i].y;
            if ( x < 0 || x > rows-1 || y < 0 || y > columns-1 ) continue;
            printf(" %.6f", accessMat( surface, x, y ) );
        }
        printf("\n");
    }

    /* 7. Free resources */ 
    /* MPI: All processes free their replicated data */
    free( teams );
    free( focal );
    /* MPI: Rank 0 frees the global surface, other ranks free NULL (which is safe) */
    free( surface );
    /* MPI: 'surfaceCopy' is NULL for all ranks (which is safe) */
    free( surfaceCopy );

    /* 8. End */
    /* MPI: Finalize MPI Environment */
    MPI_Finalize();
    return 0;
}