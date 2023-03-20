#include "mpi.h"

#include <iostream>
#include <vector>

//  mpicxx mpi.cpp -o mpi && mpiexec -n 4 ./mpi

void setMatrixParts(int* lineSize, int* lineOffset, int N, int processSize);
double* generateAMatix();
double* generateMatix(int v);
void getAxb(const double* Achunk, const double* x, const double* b,
            double* AxbChunk, int chunkSize, int chunkOffset);
void getNextX(const double* AxbChunk, const double* x, double* x_chunk,
              int chunkSize, int chunkOffset);

double* simpleIterationMethod(double* Achunk, double* b, double* x,
                              double bNorm, int* line_count, int* line_offsets);
double norm(double* vec, int size);

constexpr double eps = 0.00001f, tau = 0.0001f;
constexpr int N = 10;
int main(int argc, char* argv[]) {
  int processRank;  // current
  int processSize;  // all
  double startTime, endTime;
  int *lineOffset, *lineSize;
  double* partBuffer;
  double *b, *X, *A;
  double bNorm;

  A = generateAMatix();

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
  MPI_Comm_size(MPI_COMM_WORLD, &processSize);
  lineSize = new int[processSize];
  lineOffset = new int[processSize];
  setMatrixParts(lineSize, lineOffset, N, processSize);
  // std::cout << "RANK " << processRank << " " << lineSize[processRank] << " "
  //           << N << "\n";
  // std::cout << "RANK " << lineSize[processRank] * N << "\n";
  double* AChunk = new double[(lineSize[processRank])];
  for (int i = 0; i < (lineSize[processRank] * N); ++i) {
    AChunk[i] = -1;
  }
  startTime = MPI_Wtime();

  // generateAMatixChunk(lineSize[processRank], lineOffset[processRank]);
  if (processRank == 0) {
    for (int i = 0; i < N; ++i) {
      // for (int j = 0; j < N; ++j) {
      //   std::cout << A[i * N + j] << " ";
      // }
      // std::cout << "\n";
      printf("sendcounts[%d] = %d\tdispls[%d] = %d\n", i, lineSize[i], i,
             lineOffset[i]);
    }

    // b = generateMatix(N + 1);
    // bNorm = norm(b, N);
    // X = generateMatix(0);
    // MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&bNorm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  // MPI_Scatterv(A, lineSize, lineOffset, MPI_DOUBLE, AChunk,
  //              lineSize[processRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // std::cout << "Raaank " << processRank << "\n";
  // for (int i = 0; i < lineSize[processRank] / N; ++i) {
  //   for (int j = 0; j < N; ++j)
  //     std::cout << i << " " << j << " --> " << AChunk[i * N + j] <<
  //     std::endl;
  // }
  // b = generateMatix(N + 1);
  // bNorm = norm(b, N);
  // X = generateMatix(0);

  // if (processRank == 0)
  //   for (int i = 0; i < lineOffset[processRank] * N; ++i)
  //     std::cout << AChunk[i] << " ";
  // std::cout << "\n";

  // simpleIterationMethod(AChunk, b, X, bNorm, lineSize, lineOffset);

  endTime = MPI_Wtime();
  if (processRank == 0)
    // std::cout << "Total time is " << endTime - startTime << std::endl;
    std::cout << endTime - startTime << "\n";

  MPI_Finalize();

  return 0;
}

void setMatrixParts(int* lineSize, int* lineOffset, int N, int processSize) {
  int offset = 0, bufSize = N / processSize;
  int bufRemainder = N % processSize;
  for (int i = 0; i < processSize; ++i) {
    lineOffset[i] = offset;
    lineSize[i] = bufSize;
    if (i < bufRemainder) ++lineSize[i];
    lineSize[i] *= N;
    std::cout << lineSize[i] << " " << lineOffset[i] << "\n";

    offset += lineSize[i];
  }
}

// double* generateAMatixChunk(int lineSize, int offset) {
//   double* AChunk = new double[lineSize * N];
//   for (int i = 0; i < lineSize; ++i) {
//     for (int j = 0; j < N; ++j) AChunk[i * N + j] = 1;

//     AChunk[i * N + offset + i] = 2;  // Чтобы поставить на нужное место
//   }
//   return AChunk;
// }

double* generateMatix(int val) {
  double* x = new double[N];
  for (int i = 0; i < N; ++i) {
    x[i] = val;
  }
  return x;
}

double* generateAMatix() {
  double* A = new double[N * N];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) A[i * N + j] = 1;

    A[i * N + i] = 2;  // Чтобы поставить на нужное место
  }
  return A;
}

double* simpleIterationMethod(double* Achunk, double* b, double* x,
                              double bNorm, int* line_count,
                              int* line_offsets) {
  int i = 0;
  double acc = eps * eps;
  int processRank;  // current
  MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
  int numberOfLines = line_count[processRank], offset = line_count[processRank];
  int iCount = 0;
  double* AxbChunk = new double[numberOfLines];
  double* x_chunk = new double[numberOfLines];
  for (; acc >= eps * eps; ++iCount) {
    getAxb(Achunk, x, b, AxbChunk, numberOfLines, offset);
    getNextX(AxbChunk, x, x_chunk, numberOfLines, offset);
    MPI_Allgatherv(x_chunk, numberOfLines, MPI_DOUBLE, x, line_count,
                   line_offsets, MPI_DOUBLE, MPI_COMM_WORLD);

    double AxbChunk_norm_square = norm(AxbChunk, numberOfLines);
    MPI_Reduce(&AxbChunk_norm_square, &acc, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (processRank == 0) acc = acc / bNorm;
    MPI_Bcast(&acc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  // if (processRank == 0) std::cout << iCount << std::endl;

  return x;
}

inline void getNextX(const double* AxbChunk, const double* x, double* x_chunk,
                     int chunkSize, int chunkOffset) {
  for (int i = 0; i < chunkSize; ++i)
    x_chunk[i] = x[chunkOffset + i] - tau * AxbChunk[i];
}

void getAxb(const double* Achunk, const double* x, const double* b,
            double* AxbChunk, int chunkSize, int chunkOffset) {
  for (int i = 0; i < chunkSize; ++i) {
    AxbChunk[i] = -b[chunkOffset + i];
    for (int j = 0; j < N; ++j) AxbChunk[i] += Achunk[i * N + j] * x[j];
  }
}

double norm(double* vec, int size) {
  double norn = 0;
  for (int i = 0; i < size; ++i) {
    norn += vec[i] * vec[i];
  }
  return norn;
}
