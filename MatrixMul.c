#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_SIZE 3

void generate_matrix(int matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    int i, j;
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = rand() % 10; // random values between 0 and 9
        }
    }
}

void print_matrix(int matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    int i, j;
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size, i, j, k;
    int A[MATRIX_SIZE][MATRIX_SIZE], B[MATRIX_SIZE][MATRIX_SIZE], C[MATRIX_SIZE][MATRIX_SIZE];
    int row_A[MATRIX_SIZE], row_C[MATRIX_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Master process generates matrices A and B
    if (rank == 0) {
        generate_matrix(A);
        generate_matrix(B);
        printf("Matrix A:\n");
        print_matrix(A);
        printf("\nMatrix B:\n");
        print_matrix(B);
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(B, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter rows of matrix A to all processes
    MPI_Scatter(A, MATRIX_SIZE * MATRIX_SIZE / size, MPI_INT, row_A, MATRIX_SIZE * MATRIX_SIZE / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform local multiplication
    for (i = 0; i < MATRIX_SIZE; i++) {
        row_C[i] = 0;
        for (j = 0; j < MATRIX_SIZE; j++) {
            row_C[i] += row_A[j] * B[j][i];
        }
    }

    // Gather results from all processes
    MPI_Gather(row_C, MATRIX_SIZE * MATRIX_SIZE / size, MPI_INT, C, MATRIX_SIZE * MATRIX_SIZE / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Master process prints the result
    if (rank == 0) {
        printf("\nMatrix C (Result of A * B):\n");
        print_matrix(C);
    }

    MPI_Finalize();
    return 0;
}


