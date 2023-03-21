#include "mpi.h"

#include <iostream>
#include <vector>

//  mpicxx mpi.cpp -o mpi && mpiexec -n 4 ./mpi

void setMatrixParts(int* lineSize, int* lineOffset, int* sendCounts,
                    int* displs, int N, int processSize);
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
constexpr int N = 1024 * 13;
int main(int argc, char* argv[]) {
  int processRank;  // current
  int processSize;  // all
  double startTime, endTime;
  int *lineOffset, *lineSize, *sendCounts, *displs;
  double* partBuffer;
  double *b, *X, *A;
  double bNorm;

  A = generateAMatix();
  b = generateMatix(N + 1);
  bNorm = norm(b, N);
  X = generateMatix(0);
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
  MPI_Comm_size(MPI_COMM_WORLD, &processSize);
  lineSize = new int[processSize];
  lineOffset = new int[processSize];
  sendCounts = new int[processSize];
  displs = new int[processSize];
  setMatrixParts(lineSize, lineOffset, sendCounts, displs, N, processSize);
  double* AChunk = new double[(sendCounts[processRank])];
  for (int i = 0; i < (sendCounts[processRank]); ++i) {
    AChunk[i] = -1;
  }
  startTime = MPI_Wtime();
  MPI_Scatterv(A, sendCounts, displs, MPI_DOUBLE, AChunk,
               sendCounts[processRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  simpleIterationMethod(AChunk, b, X, bNorm, lineSize, lineOffset);

  endTime = MPI_Wtime();
  if (processRank == 0) std::cout << endTime - startTime << "\n";

  MPI_Finalize();

  return 0;
}

void setMatrixParts(int* lineSize, int* lineOffset, int* sendCounts,
                    int* displs, int N, int processSize) {
  int offset = 0, bufSize = N / processSize;
  int bufRemainder = N % processSize;
  for (int i = 0; i < processSize; ++i) {
    lineOffset[i] = offset;
    lineSize[i] = bufSize;
    if (i < bufRemainder) ++lineSize[i];
    sendCounts[i] = lineSize[i] * N;
    displs[i] = offset * N;
    offset += lineSize[i];
  }
}

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
    A[i * N + i] = 2;
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
  int numberOfLines = line_count[processRank],
      offset = line_offsets[processRank];
  int iCount = 0;
  double* AxbChunk = new double[numberOfLines]();
  double* x_chunk = new double[numberOfLines];
  for (; acc >= eps * eps;) {
    getAxb(Achunk, x, b, AxbChunk, numberOfLines, offset);
    getNextX(AxbChunk, x, x_chunk, numberOfLines, offset);
    MPI_Allgatherv(x_chunk, numberOfLines, MPI_DOUBLE, x, line_count,
                   line_offsets, MPI_DOUBLE, MPI_COMM_WORLD);

    double AxbChunk_norm_square = norm(AxbChunk, numberOfLines);
    MPI_Reduce(&AxbChunk_norm_square, &acc, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (processRank == 0) {
      acc = acc / bNorm;
      ++iCount;
    }
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
    AxbChunk[i] = 0;
    for (int j = 0; j < N; ++j) AxbChunk[i] += Achunk[i * N + j] * x[j];
    AxbChunk[i] -= b[chunkOffset + i];
  }
}

double norm(double* vec, int size) {
  double norn = 0;
  for (int i = 0; i < size; ++i) {
    norn += vec[i] * vec[i];
  }
  return norn;
}
