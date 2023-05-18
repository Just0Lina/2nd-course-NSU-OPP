
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
