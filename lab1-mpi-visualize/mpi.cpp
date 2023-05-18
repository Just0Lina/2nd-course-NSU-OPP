#include "mpi.h"

#include <math.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

double* generateMatixU();

void calc_Axb(const double* A_chunk, const double* x, const double* b,
              double* Axb_chunk, int chunk_size, int chunk_offset);
void calc_next_x(const double* Axb_chunk, const double* x, double* x_chunk,
                 double tau, int chunk_size, int chunk_offset);
double calc_TAU(const double* A_chunk, const double* Axb_chunk, int chunk_size,
                int chunk_offset, int process_rank, int* line_counts,
                int* line_offsets, double* Ay);

double* anotherMethod(double* Achunk, double* b, double* x, double bNorm,
                      int* line_count, int* line_offsets);

double* getFileMatrix(std::string path, int size);

constexpr double eps = 0.00001f, tau = 0.00001f, MAX_ITER = 1000;
constexpr int N = 2500;

//  mpicxx mpi.cpp -o mpi && mpiexec -n 4 ./mpi
void writeToFile(double* X);
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

int main(int argc, char* argv[]) {
  // double* A = getFileMatrix("matA.bin", N * N);
  // double* b = getFileMatrix("vecB.bin", N);
  // double* XRes = getFileMatrix("vecX.bin", N);
  double* X = generateMatix(0);

  double* A = generateAMatix();
  double* b = generateMatix(N + 1);
  int processRank;  // current
  int processSize;  // all
  double startTime, endTime;
  int *lineOffset, *lineSize, *sendCounts, *displs;

  double* partBuffer;
  double bNorm;

  bNorm = sqrt(norm(b, N));
  // double* X = generateMatix(0);
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
  MPI_Comm_size(MPI_COMM_WORLD, &processSize);
  lineSize = new int[processSize];
  lineOffset = new int[processSize];
  sendCounts = new int[processSize];
  displs = new int[processSize];
  setMatrixParts(lineSize, lineOffset, sendCounts, displs, N, processSize);
  double* AChunk = new double[(sendCounts[processRank])];
  // for (int i = 0; i < processSize; ++i) {
  //   std::cout << lineSize[i] << " ";
  // }

  startTime = MPI_Wtime();
  MPI_Scatterv(A, sendCounts, displs, MPI_DOUBLE, AChunk,
               sendCounts[processRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  X = anotherMethod(AChunk, b, X, bNorm, lineSize, lineOffset);
  // X = simpleIterationMethod(AChunk, b, X, bNorm, lineSize, lineOffset);

  endTime = MPI_Wtime();
  if (processRank == 0) {
    std::cout << endTime - startTime << "\n";
    writeToFile(X);
    // std::cout << "Matrix comp: \n";
    // for (int i = 0; i < N; ++i) {
    //   //   // if (A[i]) std::cout << A[i] << "  " << i << "\n";
    //   //   // if (X[i]) std::cout << X[i] << "  " << i << "\n";

    //   // if (X[i] || XRes[i]) std::cout << XRes[i] << " " << X[i] << "\n";
    // }
    // writeToFile(X);
  }

  MPI_Finalize();
  // if (!processRank) {
  //   // writeToFile(X);
  //   // X = getFileMatrix("vecMyX.bin", N);

  // }
  return 0;
}

void writeToFile(double* X) {
  std::ofstream out("vecA.bin", std::ios::out);
  for (int i = 0; i < N; i++) {
    float f = (float)X[i];
    // std::cout << f << " " << X[i] << "\n";
    out.write(reinterpret_cast<const char*>(&f), sizeof(float));
  }
  out.close();
}

void setMatrixParts(int* lineSize, int* lineOffset, int* sendCounts,
                    int* displs, int N, int processSize) {
  int offset = 0, bufSize = N / processSize;
  int bufRemainder = N % processSize;
  // std::cout << bufSize << " " << processSize << "\n";
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

double* generateMatixU() {
  double* x = new double[N];
  for (int i = 0; i < N; ++i) {
    // x[i] = sin(2 * M_PI / N);
    x[i] = (rand() % 100) / 100.0;
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

double* getFileMatrix(std::string path, int size) {
  float f;
  std::ifstream fin(path, std::ios::binary);
  double* matrix = new double[size]();
  int i = 0;
  while (i < size) {
    float f;
    fin.read(reinterpret_cast<char*>(&f), sizeof(float));
    matrix[i] = f;
    i++;
  }
  return matrix;
}

double* simpleIterationMethod(double* Achunk, double* b, double* x,
                              double bNorm, int* line_count,
                              int* line_offsets) {
  int i = 0;
  double acc = eps;
  int processRank;  // current
  MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
  int numberOfLines = line_count[processRank],
      offset = line_offsets[processRank];
  int iCount = 0;
  double* AxbChunk = new double[numberOfLines]();
  double* x_chunk = new double[numberOfLines];
  for (; acc >= eps && iCount < 100;) {
    getAxb(Achunk, x, b, AxbChunk, numberOfLines, offset);
    getNextX(AxbChunk, x, x_chunk, numberOfLines, offset);
    MPI_Allgatherv(x_chunk, numberOfLines, MPI_DOUBLE, x, line_count,
                   line_offsets, MPI_DOUBLE, MPI_COMM_WORLD);
    // std::cout << "Here";

    double AxbChunk_norm_square = norm(AxbChunk, numberOfLines);
    MPI_Reduce(&AxbChunk_norm_square, &acc, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (processRank == 0) {
      acc = sqrt(acc) / bNorm;
      ++iCount;
    }
    MPI_Bcast(&acc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  // if (processRank == 0) std::cout << iCount << std::endl;

  return x;
}

double* anotherMethod(double* Achunk, double* b, double* x, double bNorm,
                      int* line_counts, int* line_offsets) {
  int processRank;  // current

  // setMatrixParts(lineSize, lineOffset, sendCounts, displs, N, processSize);

  MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
  // MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
  double acc = eps * eps;
  double* Ay = new double[N];
  double* x_chunk = new double[line_counts[processRank]];
  double* Axb_chunk = new double[line_counts[processRank]];
  int iter_count;

  for (iter_count = 0; acc >= eps * eps && iter_count < MAX_ITER;
       ++iter_count) {
    calc_Axb(Achunk, x, b, Axb_chunk, line_counts[processRank],
             line_offsets[processRank]);

    MPI_Allgatherv(Axb_chunk, line_counts[processRank], MPI_DOUBLE, Ay,
                   line_counts, line_offsets, MPI_DOUBLE, MPI_COMM_WORLD);

    double TAU = calc_TAU(Achunk, Axb_chunk, line_counts[processRank],
                          line_offsets[processRank], processRank, line_counts,
                          line_offsets, Ay);
    // printf("%f \n", TAU);

    MPI_Bcast(&TAU, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (TAU != TAU) break;

    // printf("%f \n", TAU);
    calc_next_x(Axb_chunk, x, x_chunk, TAU, line_counts[processRank],
                line_offsets[processRank]);

    MPI_Allgatherv(x_chunk, line_counts[processRank], MPI_DOUBLE, x,
                   line_counts, line_offsets, MPI_DOUBLE, MPI_COMM_WORLD);

    double Axb_chunk_norm_square = norm(Axb_chunk, line_counts[processRank]);
    MPI_Reduce(&Axb_chunk_norm_square, &acc, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (processRank == 0) acc = sqrt(acc) / bNorm;
    MPI_Bcast(&acc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  return x;
}

double calc_TAU(const double* A_chunk, const double* Axb_chunk, int chunk_size,
                int chunk_offset, int process_rank, int* line_counts,
                int* line_offsets, double* Ay) {
  double* Ay_chunk = new double[chunk_size];
  // std::cout << chunk_size << "\n";
  for (int i = 0; i < chunk_size; ++i) {
    Ay_chunk[i] = 0.0;
    // std::cout << "!!!" << A_chunk[i] << " ";
  }

  for (int i = 0; i < chunk_size; ++i) {
    for (int j = 0; j < N; j++) {
      Ay_chunk[i] += A_chunk[i * N + j] * Ay[j];
      // printf("Ay[j]: %f \n", Ay[j]);
      // if (i == 0 && process_rank == 0)
      // printf("Ay_chunk[i] = %f, j = %d\n, ", Ay_chunk[i], j);
    }
    // printf("Ay_chunk[i] = %f, i = %d\n", Ay_chunk[i], i);
  }

  double first = 0, second = 0;

  for (int i = 0; i < chunk_size; ++i) {
    first += Axb_chunk[i] * Ay_chunk[i];
    // printf("first: %f\n", first);
    second += Ay_chunk[i] * Ay_chunk[i];
    // printf("second: %f\n", second);
  }

  double divider = 0, divided = 0;

  MPI_Reduce(&first, &divided, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&second, &divider, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // printf("процесс: %d, first: %f, second: %f, divided: %f, divider: %f\n",
  // process_rank, first, second, divided, divider);

  double TAU = divided / divider;

  // printf("TAU: %f \n", TAU);

  return TAU;
}

void calc_next_x(const double* Axb_chunk, const double* x, double* x_chunk,
                 double tau, int chunk_size, int chunk_offset) {
  // std::cout << tau << " " << chunk_size << " " << chunk_offset << "\n";

  for (int i = 0; i < chunk_size; ++i) {
    // printf("%f ", x[chunk_offset + i]);
    x_chunk[i] = x[chunk_offset + i] - tau * Axb_chunk[i];
  }
}

void calc_Axb(const double* A_chunk, const double* x, const double* b,
              double* Axb_chunk, int chunk_size, int chunk_offset) {
  for (int i = 0; i < chunk_size; ++i) {
    Axb_chunk[i] = -b[chunk_offset + i];
    for (int j = 0; j < N; ++j) {
      Axb_chunk[i] += A_chunk[i * N + j] * x[j];
      // printf("%f ", x[i]);
    }
    // printf("\n");
  }
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
