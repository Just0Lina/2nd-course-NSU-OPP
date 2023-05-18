#include <fstream>
#include <iostream>
#include <sstream>

#include "mpi.h"
// mpicxx main.cpp -o mpi && mpiexec -n 4 ./mpi
const int n_1 = 512 * 3, n_2 = 512 * 3, n_3 = 512 * 3;
#include <math.h>
const int DIMS_COUNT = 2;
const int X = 0;
const int Y = 1;

using namespace std;

double* fillMatix(int val, int row, int col);
void setMatrixPartsB(int* sendCounts, int* displs, int col, int processSize);
void setMatrixPartsA(int* sendCounts, int* displs, int col, int processSize);

void gather_C(const double* C_block, double* C, int A_block_size,
              int B_block_size, int aligned_n1, int aligned_n3, int proc_count,
              MPI_Comm comm_grid) {
  MPI_Datatype not_resized_recv_t;
  MPI_Datatype resized_recv_t;

  int dims_Y = aligned_n1 / A_block_size;
  int dims_X = aligned_n3 / B_block_size;
  int* recv_counts = new int[proc_count];
  int* displs = new int[proc_count];

  MPI_Type_vector(A_block_size, B_block_size, aligned_n3, MPI_DOUBLE,
                  &not_resized_recv_t);
  MPI_Type_commit(&not_resized_recv_t);

  MPI_Type_create_resized(not_resized_recv_t, 0, B_block_size * sizeof(double),
                          &resized_recv_t);
  MPI_Type_commit(&resized_recv_t);

  for (int i = 0; i < dims_Y; ++i)
    for (int j = 0; j < dims_X; ++j) {
      recv_counts[i * dims_X + j] = 1;
      displs[i * dims_X + j] = j + i * dims_X * A_block_size;
    }

  MPI_Gatherv(C_block, A_block_size * B_block_size, MPI_DOUBLE, C, recv_counts,
              displs, resized_recv_t, 0, comm_grid);

  MPI_Type_free(&not_resized_recv_t);
  MPI_Type_free(&resized_recv_t);
}

void split_A(const double* A, double* A_block, int A_block_size, int n_2,
             int coords_y, MPI_Comm comm_rows, MPI_Comm comm_columns) {
  if (coords_y == 0) {
    MPI_Scatter(A, A_block_size * n_2, MPI_DOUBLE, A_block, A_block_size * n_2,
                MPI_DOUBLE, 0, comm_columns);
  }

  MPI_Bcast(A_block, A_block_size * n_2, MPI_DOUBLE, 0, comm_rows);
}

void split_B(const double* B, double* B_block, int B_block_size, int n_2,
             int aligned_n3, int coords_x, MPI_Comm comm_rows,
             MPI_Comm comm_columns) {
  if (coords_x == 0) {
    MPI_Datatype column_not_resized_t;
    MPI_Datatype column_resized_t;

    MPI_Type_vector(n_2, B_block_size, aligned_n3, MPI_DOUBLE,
                    &column_not_resized_t);
    MPI_Type_commit(&column_not_resized_t);

    MPI_Type_create_resized(column_not_resized_t, 0,
                            B_block_size * sizeof(double), &column_resized_t);
    MPI_Type_commit(&column_resized_t);

    MPI_Scatter(B, 1, column_resized_t, B_block, B_block_size * n_2, MPI_DOUBLE,
                0, comm_rows);

    MPI_Type_free(&column_not_resized_t);
    MPI_Type_free(&column_resized_t);
  }

  MPI_Bcast(B_block, B_block_size * n_2, MPI_DOUBLE, 0, comm_columns);
}

void multiply(const double* A_block, const double* B_block, double* C_block,
              int A_block_size, int B_block_size, int n_2) {
  for (int i = 0; i < A_block_size; ++i)
    for (int j = 0; j < B_block_size; ++j) C_block[i * B_block_size + j] = 0;
  // C_block = fillMatix(0, A_block_size, B_block_size);

  for (int i = 0; i < A_block_size; ++i)
    for (int j = 0; j < n_2; ++j)
      for (int k = 0; k < B_block_size; ++k)
        C_block[i * B_block_size + k] +=
            A_block[i * n_2 + j] * B_block[j * B_block_size + k];
}

void initCommunicators(const int dims[DIMS_COUNT], MPI_Comm* comm_grid,
                       MPI_Comm* comm_rows, MPI_Comm* comm_columns) {
  int reorder = 1;
  int periods[DIMS_COUNT] = {};
  int sub_dims[DIMS_COUNT] = {};

  MPI_Cart_create(MPI_COMM_WORLD, DIMS_COUNT, dims, periods, reorder,
                  comm_grid);

  sub_dims[X] = false;
  sub_dims[Y] = true;
  MPI_Cart_sub(*comm_grid, sub_dims, comm_rows);

  sub_dims[X] = true;
  sub_dims[Y] = false;
  MPI_Cart_sub(*comm_grid, sub_dims, comm_columns);
}

void simple_multiply(double* mat1, double* mat2, int N, int M, int O,
                     double* res) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      res[i * M + j] = 0;
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < O; k++)
        res[i * M + j] += mat1[i * O + k] * mat2[k * M + j];
    }
  }
}

double* generateMatix(int row, int col) {
  double* x = new double[row * col];
  for (int i = 0; i < row * col; ++i) {
    x[i] = rand() % 100;
  }
  return x;
}

int main(int argc, char* argv[]) {
  double* submatrix;
  int processRank, processSize;
  double startTime, endTime;
  int *sendCountsA, *displsA;
  int *sendCountsB, *displsB;
  double *A, *B, *C, *A_block, *B_block, *C_block;
  int A_block_size, B_block_size;
  int dims[DIMS_COUNT] = {};
  int coords[DIMS_COUNT] = {};
  MPI_Comm comm_grid, comm_rows, comm_columns;

  double* partBuffer;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
  MPI_Comm_size(MPI_COMM_WORLD, &processSize);

  MPI_Dims_create(processSize, DIMS_COUNT, dims);
  initCommunicators(dims, &comm_grid, &comm_rows, &comm_columns);

  // Get coordinates of processes
  MPI_Cart_coords(comm_grid, processRank, DIMS_COUNT, coords);

  if (!processRank) cout << dims[X] << "  " << dims[Y] << "\n";

  A_block_size = ceil((double)n_1 / dims[X]);
  B_block_size = ceil((double)n_3 / dims[Y]);

  if (coords[X] == 0 && coords[Y] == 0) {
    srand(0);
    A = generateMatix(n_1, n_2);
    B = generateMatix(n_1, n_2);

    // A = getFileMatrix("mat_a.txt", n_1, n_2);
    // B = getFileMatrix("mat_b.txt", n_2, n_3);

    C = fillMatix(0, n_1, n_3);
  }

  startTime = MPI_Wtime();

  A_block = new double[A_block_size * n_2];
  B_block = new double[B_block_size * n_2];
  C_block = new double[A_block_size * B_block_size];

  split_A(A, A_block, A_block_size, n_2, coords[Y], comm_rows, comm_columns);
  split_B(B, B_block, B_block_size, n_2, n_3, coords[X], comm_rows,
          comm_columns);

  startTime = MPI_Wtime();

  multiply(A_block, B_block, C_block, A_block_size, B_block_size, n_2);

  gather_C(C_block, C, A_block_size, B_block_size, n_1, n_3, processSize,
           comm_grid);

  endTime = MPI_Wtime();
  if (processRank == 0) {
    std::cout << endTime - startTime << "\n";
  }

  MPI_Finalize();
  return 0;
}

double* fillMatix(int val, int row, int col) {
  double* x = new double[row * col];
  for (int i = 0; i < row * col; ++i) {
    x[i] = val;
  }
  return x;
}

void setMatrixPartsB(int* sendCounts, int* displs, int col, int processSize) {
  int offset = 0, bufSize = col / processSize;
  int bufRemainder = col % processSize;
  for (int i = 0; i < processSize; ++i) {
    displs[i] = offset;
    sendCounts[i] = bufSize;
    if (i < bufRemainder) {
      ++sendCounts[i];
      ++offset;
    }
    offset += bufSize;
  }
}

void setMatrixPartsA(int* sendCounts, int* displs, int N, int processSize) {
  int offset = 0, bufSize = N / processSize;
  int bufRemainder = N % processSize;
  for (int i = 0; i < processSize; ++i) {
    displs[i] = offset;
    sendCounts[i] = bufSize;
    if (i < bufRemainder) ++sendCounts[i];
    offset += sendCounts[i] * N;
  }
}