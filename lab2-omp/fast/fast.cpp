// #include <mpi.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>

#include <iostream>
#include <vector>

#include "../matrix.h"
void simple_iteration_method(Matrix& A, Matrix& b, Matrix& x);
Matrix ChangingX(Matrix& x, Matrix& A);
double Norm(Matrix vec);

using namespace std;
constexpr double eps = 0.00001f, tau = 0.0001f;
constexpr int N = 1024 * 13;
int main(int argc, char* argv[]) {
  Matrix A(N, N);
  Matrix X(N, 1), b(N, 1);

#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    X(i, 0) = 0;
  }

#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A(i, j) = 1.0f;
      // std::cout << "Index (" << i << ", " << j << ") "
      //           << "Thread: " << omp_get_thread_num() << std::endl;
    }
  }
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    b(i, 0) = N + 1;
    A(i, i) = 2.0f;
  }
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  simple_iteration_method(A, b, X);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  // X.show_matrix();
  std::cout << (float)(end.tv_sec - start.tv_sec) +
                   (1e-9 * (end.tv_nsec - start.tv_nsec))
            << std::endl;
  return 0;
}

void simple_iteration_method(Matrix& A, Matrix& b, Matrix& x) {
  Matrix Axb = (A * x) - b;
  while (Norm(Axb) / Norm(b) >= eps * eps) {
    x = ChangingX(x, Axb);
    Axb = (A * x) - b;
  }
}

inline Matrix ChangingX(Matrix& x, Matrix& Axb) { return x - (Axb * tau); }

double Norm(Matrix vec) {
  if (vec.get_cols() != 1) throw std::out_of_range("Error: more than 1 col");
  double res = 0;
#pragma omp parallel for reduction(+ : res)
  for (int i = 0; i < vec.get_rows(); ++i) {
    res += vec(i, 0) * vec(i, 0);
  }
  return res;
}