#include "mpiTau.h"

#include "mpi.h"
//  mpicxx mpiTau.cpp -o mpi && mpiexec -n 4 ./mpi

int main(int argc, char* argv[]) {
  double* A = getFileMatrix("matA.bin", N * N);
  double* b = getFileMatrix("vecB.bin", N);
  // double* XRes = getFileMatrix("vecX.bin", N);
  double* X = generateMatix(0);

  int processRank;  // current
  int processSize;  // all
  double startTime, endTime;

  int *lineOffset, *lineSize, *sendCounts, *displs;
  double bNorm = sqrt(norm(b, N));
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
  MPI_Comm_size(MPI_COMM_WORLD, &processSize);
  lineSize = new int[processSize];
  lineOffset = new int[processSize];
  sendCounts = new int[processSize];
  displs = new int[processSize];
  setMatrixParts(lineSize, lineOffset, sendCounts, displs, N, processSize);
  double* AChunk = new double[(sendCounts[processRank])];

  startTime = MPI_Wtime();
  MPI_Scatterv(A, sendCounts, displs, MPI_DOUBLE, AChunk,
               sendCounts[processRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  X = anotherMethod(AChunk, b, X, bNorm, lineSize, lineOffset);

  endTime = MPI_Wtime();
  if (processRank == 0) {
    std::cout << endTime - startTime << "\n";
    writeToFile(X);
  }

  MPI_Finalize();
  return 0;
}

void writeToFile(double* X) {
  std::ofstream out("vecMyXClear.bin", std::ios::binary);
  for (int i = 0; i < N; i++) {
    float f = (float)X[i];
    out.write(reinterpret_cast<const char*>(&f), sizeof(float));
  }
  out.close();
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

double* anotherMethod(double* Achunk, double* b, double* x, double bNorm,
                      int* line_counts, int* line_offsets) {
  int processRank;  // current
  MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
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
    MPI_Bcast(&TAU, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (TAU != TAU) break;
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
  for (int i = 0; i < chunk_size; ++i) {
    Ay_chunk[i] = 0.0;
  }

  for (int i = 0; i < chunk_size; ++i) {
    for (int j = 0; j < N; j++) {
      Ay_chunk[i] += A_chunk[i * N + j] * Ay[j];
    }
  }

  double first = 0, second = 0;

  for (int i = 0; i < chunk_size; ++i) {
    first += Axb_chunk[i] * Ay_chunk[i];
    second += Ay_chunk[i] * Ay_chunk[i];
  }
  double divider = 0, divided = 0;

  MPI_Reduce(&first, &divided, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&second, &divider, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  double TAU = divided / divider;
  return TAU;
}

void calc_next_x(const double* Axb_chunk, const double* x, double* x_chunk,
                 double tau, int chunk_size, int chunk_offset) {
  for (int i = 0; i < chunk_size; ++i) {
    x_chunk[i] = x[chunk_offset + i] - tau * Axb_chunk[i];
  }
}

void calc_Axb(const double* A_chunk, const double* x, const double* b,
              double* Axb_chunk, int chunk_size, int chunk_offset) {
  for (int i = 0; i < chunk_size; ++i) {
    Axb_chunk[i] = -b[chunk_offset + i];
    for (int j = 0; j < N; ++j) {
      Axb_chunk[i] += A_chunk[i * N + j] * x[j];
    }
  }
}

double norm(double* vec, int size) {
  double norn = 0;
  for (int i = 0; i < size; ++i) {
    norn += vec[i] * vec[i];
  }
  return norn;
}

double* generateMatix(int val) {
  double* x = new double[N];
  for (int i = 0; i < N; ++i) {
    x[i] = val;
  }
  return x;
}
