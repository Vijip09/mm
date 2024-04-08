#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#define N 3       /* number of rows and columns in matrix */

MPI_Status status;

int a[N][N], b[N][N], c[N][N];

int main(int argc, char **argv) {
    int numtasks, taskid, numworkers, source, dest, rows, offset, i, j, k;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    numworkers = numtasks - 1;

    /*---------------------------- master ----------------------------*/
    if (taskid == 0) {
        srand(time(NULL)); // Seed random number generator

        // Initialize matrices A and B with random values
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                a[i][j] = rand() % 10;
                b[i][j] = rand() % 10;
            }
        }

        // Print matrices A and B
        printf("Matrix A:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%d   ", a[i][j]);
            }
            printf("\n");
        }

        printf("Matrix B:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%d   ", b[i][j]);
            }
            printf("\n");
        }

        // Send matrix data to the worker tasks
        rows = N / numworkers;
        int remaining_rows = N % numworkers; // Calculate remaining rows
        offset = 0;

        for (dest = 1; dest <= numworkers; dest++) {
            // Distribute remaining rows to the last process
            if (dest == numworkers) {
                rows += remaining_rows;
            }

            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows * N, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&b, N * N, MPI_INT, dest, 1, MPI_COMM_WORLD);
            offset += rows;
        }

        // Receive results from worker tasks
        for (i = 1; i <= numworkers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows * N, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
        }

        // Print result matrix C
        printf("Result Matrix C:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%d   ", c[i][j]);
            }
            printf("\n");
        }
    }

    /*---------------------------- worker----------------------------*/
    if (taskid > 0) {
        source = 0;
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows * N, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, N * N, MPI_INT, source, 1, MPI_COMM_WORLD, &status);

        // Matrix multiplication
        for (k = 0; k < N; k++) {
            for (i = 0; i < rows; i++) {
                c[i][k] = 0;
                for (j = 0; j < N; j++) {
                    c[i][k] += a[i][j] * b[j][k];
                }
            }
        }

        // Send the result back to process 0
        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&c, rows * N, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}


