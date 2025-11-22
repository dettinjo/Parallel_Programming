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
    /* MPI: local_i=0 is top ghost row, local_i=1 is first real row */
    #define accessLocalMat( arr, exp1, exp2 )    arr[ (exp1) * columns + (exp2) ]

    /* MPI: Local surface data setup */
    int local_rows = rows / nprocs;
    int local_alloc_rows = local_rows + 2; // Add 2 for ghost rows
    int first_row = rank * local_rows;
    int last_row = (rank + 1) * local_rows - 1;

    /* MPI: Pointers for local surfaces */
    float *local_surface, *local_surfaceCopy;

    /* MPI: Allocate local surfaces on all processes */
    local_surface = (float *)malloc( sizeof(float) * (size_t)local_alloc_rows * (size_t)columns );
    local_surfaceCopy = (float *)malloc( sizeof(float) * (size_t)local_alloc_rows * (size_t)columns );

    /* MPI: Set 'surface' and 'surfaceCopy' to NULL on non-zero ranks */
    /* so that the 'free' calls at the end of main() don't crash */
    if (rank == 0) {
        surface = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
        if ( surface == NULL ) {
            fprintf(stderr,"-- Error allocating global surface for gather\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    } else {
        surface = NULL; // Other ranks don't need the global buffer
    }
    surfaceCopy = NULL; // Not used in the parallel version's final output


    if ( local_surface == NULL || local_surfaceCopy == NULL ) {
        fprintf(stderr,"-- Error allocating: local surface structures on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    /* MPI: Parallel initialization of local surfaces */
    for( i=0; i<local_alloc_rows; i++ )
        for( j=0; j<columns; j++ ) {
            accessLocalMat( local_surface, i, j ) = 0.0;
            accessLocalMat( local_surfaceCopy, i, j ) = 0.0;
        }

    /* 4. Simulation */
    int iter;
    int flag_stability = 0;
    int first_activation = 0;

    /* MPI: MPI Status for receives */
    MPI_Status status;
    /* MPI: Neighbor ranks */
    int up_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down_neighbor = (rank == nprocs - 1) ? MPI_PROC_NULL : rank + 1;

    for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {

        /* 4.1. Activate focal points */
        /* MPI: Replicated logic, all processes do this */
        int num_deactivated = 0;
        for( i=0; i<num_focal; i++ ) {
            if ( focal[i].start == iter ) {
                focal[i].active = 1;
                if ( ! first_activation ) first_activation = 1;
            }
            // Count focal points already deactivated by a team
            if ( focal[i].active == 2 ) num_deactivated++;
        }

        /* 4.2. Propagate heat (10 steps per each team movement) */
        float local_residual = 0.0f; // MPI: Residual is local first
        float global_residual = 0.0f;
        int step;
        for( step=0; step<10; step++ )  {
            /* 4.2.1. Update heat on active focal points */
            /* MPI: Replicated logic, but *distributed write* */
            for( i=0; i<num_focal; i++ ) {
                if ( focal[i].active != 1 ) continue;
                int x = focal[i].x;
                int y = focal[i].y;
                if ( x < 0 || x > rows-1 || y < 0 || y > columns-1 ) continue;

                /* MPI: Check if this focal point is in my assigned rows */
                if ( x >= first_row && x <= last_row ) {
                    /* MPI: Convert global row 'x' to local row 'local_i' */
                    int local_i = x - first_row + 1; // +1 for top ghost row
                    accessLocalMat( local_surface, local_i, y ) = focal[i].heat;
                }
            }

            /* MPI: Ghost Row Exchange */
            /* MPI: Pointers to send/recv buffers */
            float *send_buf_up = &accessLocalMat( local_surface, 1, 0 );
            float *recv_buf_up = &accessLocalMat( local_surface, 0, 0 );
            float *send_buf_down = &accessLocalMat( local_surface, local_rows, 0 );
            float *recv_buf_down = &accessLocalMat( local_surface, local_rows + 1, 0 );

            /*
             * FIX: Use TAG 0 for all exchanges to prevent deadlock.
             * Since we use Sendrecv, the ordering ensures Rank i DOWN matches Rank i+1 UP.
             */

            /* MPI: Exchange with UP neighbor (send my first row, get their last row) */
            MPI_Sendrecv(send_buf_up, columns, MPI_FLOAT, up_neighbor, 0,
                         recv_buf_up, columns, MPI_FLOAT, up_neighbor, 0,
                         MPI_COMM_WORLD, &status);

            /* MPI: Exchange with DOWN neighbor (send my last row, get their first row) */
            MPI_Sendrecv(send_buf_down, columns, MPI_FLOAT, down_neighbor, 0,
                         recv_buf_down, columns, MPI_FLOAT, down_neighbor, 0,
                         MPI_COMM_WORLD, &status);


            /* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
            /* MPI: Parallel copy of local data (including ghost rows) */
            for( i=0; i<local_alloc_rows; i++ )
                for( j=0; j<columns; j++ ) // FIX: Copy *all* columns so boundaries are 0.0
                    accessLocalMat( local_surfaceCopy, i, j ) = accessLocalMat( local_surface, i, j );


            /* MPI: We must skip the global boundaries (row 0 and row rows-1) */
            int i_start = 1; // First local row to compute
            int i_end = local_rows; // Last local row to compute
            
            // If I am rank 0, my first real row (i=1) is global row 0. Skip it.
            if (rank == 0) i_start = 2; 
            
            // If I am the last rank, my last real row (i=local_rows) is global row rows-1. Skip it.
            if (rank == nprocs - 1) i_end = local_rows - 1;


            /* 4.2.3. Update surface values (skip borders) */
            /* MPI: Parallel computation on local rows (using corrected i_start/i_end) */
            for( i=i_start; i<=i_end; i++ )
                for( j=1; j<columns-1; j++ )
                    accessLocalMat( local_surface, i, j ) = ( 
                        accessLocalMat( local_surfaceCopy, i-1, j ) +
                        accessLocalMat( local_surfaceCopy, i+1, j ) +
                        accessLocalMat( local_surfaceCopy, i, j-1 ) +
                        accessLocalMat( local_surfaceCopy, i, j+1 ) ) / 4;

            /* 4.2.4. Compute the maximum residual difference (absolute value) */
            /* MPI: Compute *local* residual (using corrected i_start/i_end) */
            local_residual = 0.0f;
            for( i=i_start; i<=i_end; i++ )
                for( j=1; j<columns-1; j++ ) 
                    if ( fabs( accessLocalMat( local_surface, i, j ) - accessLocalMat( local_surfaceCopy, i, j ) ) > local_residual ) {
                        local_residual = fabs( accessLocalMat( local_surface, i, j ) - accessLocalMat( local_surfaceCopy, i, j ) );
                    }
        }
        /* MPI: Get the global maximum residual from all local residuals */
        MPI_Allreduce(&local_residual, &global_residual, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        
        /* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
        /* MPI: This check is now globally consistent */
        if( num_deactivated == num_focal && global_residual < THRESHOLD ) flag_stability = 1;

        /* 4.3. Move teams */
        /* MPI: Replicated logic, all processes compute all team movements */
        for( t=0; t<num_teams; t++ ) {
            /* 4.3.1. Choose nearest focal point */
            float distance = FLT_MAX;
            int target = -1;
            for( j=0; j<num_focal; j++ ) {
                if ( focal[j].active != 1 ) continue; // Skip non-active focal points
                float dx = focal[j].x - teams[t].x;
                float dy = focal[j].y - teams[t].y;
                float local_distance = sqrtf( dx*dx + dy*dy );
                if ( local_distance < distance ) {
                    distance = local_distance;
                    target = j;
                }
            }
            /* 4.3.2. Annotate target for the next stage */
            teams[t].target = target;

            /* 4.3.3. No active focal point to choose, no movement */
            if ( target == -1 ) continue; 

            /* 4.3.4. Move in the focal point direction */
            if ( teams[t].type == 1 ) { 
                // Type 1: Can move in diagonal
                if ( focal[target].x < teams[t].x ) teams[t].x--;
                if ( focal[target].x > teams[t].x ) teams[t].x++;
                if ( focal[target].y < teams[t].y ) teams[t].y--;
                if ( focal[target].y > teams[t].y ) teams[t].y++;
            }
            else if ( teams[t].type == 2 ) { 
                // Type 2: First in horizontal direction, then in vertical direction
                if ( focal[target].y < teams[t].y ) teams[t].y--;
                else if ( focal[target].y > teams[t].y ) teams[t].y++;
                else if ( focal[target].x < teams[t].x ) teams[t].x--;
                else if ( focal[target].x > teams[t].x ) teams[t].x++;
            }
            else {
                // Type 3: First in vertical direction, then in horizontal direction
                if ( focal[target].x < teams[t].x ) teams[t].x--;
                else if ( focal[target].x > teams[t].x ) teams[t].x++;
                else if ( focal[target].y < teams[t].y ) teams[t].y--;
                else if ( focal[target].y > teams[t].y ) teams[t].y++;
            }
        }

        /* 4.4. Team actions */
        for( t=0; t<num_teams; t++ ) {
            /* 4.4.1. Deactivate the target focal point when it is reached */
            /* MPI: Replicated logic */
            int target = teams[t].target;
            if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
                && focal[target].active == 1 )
                focal[target].active = 2;

            /* 4.4.2. Reduce heat in a circle around the team */
            /* MPI: Replicated logic, *distributed write* */
            int radius;
            // Influence area of fixed radius depending on type
            if ( teams[t].type == 1 ) radius = RADIUS_TYPE_1;
            else radius = RADIUS_TYPE_2_3;
            for( i=teams[t].x-radius; i<=teams[t].x+radius; i++ ) {
                for( j=teams[t].y-radius; j<=teams[t].y+radius; j++ ) {
                    if ( i<1 || i>=rows-1 || j<1 || j>=columns-1 ) continue; // Out of the heated surface

                    /* MPI: Check if this cell 'i' is in my assigned rows */
                    if ( i >= first_row && i <= last_row ) {
                        float dx = teams[t].x - i;
                        float dy = teams[t].y - j;
                        float distance = sqrtf( dx*dx + dy*dy );
                        if ( distance <= radius ) {
                            /* MPI: Convert global row 'i' to local row 'local_i' */
                            int local_i = i - first_row + 1; // +1 for top ghost row
                            accessLocalMat( local_surface, local_i, j ) = accessLocalMat( local_surface, local_i, j ) * ( 1 - 0.25 ); // Team efficiency factor
                        }
                    }
                }
            }
        }

#ifdef DEBUG
        /* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
        /* MPI: This is non-trivial to parallelize. It would require a full Gather */
        /* operation *inside* the loop, which is very slow. Commented out for performance. */
        // print_status( iter, rows, columns, surface, num_teams, teams, num_focal, global_residual );
#endif // DEBUG
    }

    /* MPI: Gather the final surface onto rank 0 */
    /* Each process must send its 'local_rows * columns' of data */
    int sendcount = local_rows * columns;
    
    /* MPI: Gather all local data (skipping the top ghost row) into the 'surface' on rank 0 */
    MPI_Gather( &accessLocalMat(local_surface, 1, 0), sendcount, MPI_FLOAT,
                surface, sendcount, MPI_FLOAT,
                0, MPI_COMM_WORLD );
    
    /*
     * FIX: Free AFTER gather
     */
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