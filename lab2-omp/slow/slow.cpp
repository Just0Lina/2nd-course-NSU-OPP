#include <iostream>
#include <vector>

#include "../matrix.h"
Matrix simple_iteration_method(Matrix& A, Matrix& b);
Matrix ChangingX(Matrix& x, Matrix& A, Matrix& b);
double Norm(Matrix vec);

using namespace std;
constexpr double eps = 0.00001f, tau = 0.0001f;
constexpr int N = 2048 * 2;
int main(int argc, char* argv[]) {
  Matrix A(N, N);
  Matrix X(N, 1), b(N, 1);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A(i, j) = 1.0f;
    }
    A(i, i) = 2.0f;
  }

  for (int i = 0; i < N; ++i) {
    b(i, 0) = N + 1;
  }
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  X = simple_iteration_method(A, b);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  // X.show_matrix();
  std::cout << "\nElapsed time is "
            << (float)(end.tv_sec - start.tv_sec) +
                   (1e-9 * (end.tv_nsec - start.tv_nsec))
            << std::endl;
  // X = simple_iteration_method(A, b);
  //  X.show_matrix();

  return 0;
}

Matrix simple_iteration_method(Matrix& A, Matrix& b) {
  Matrix x(N, 1);
  for (int i = 0; i < N; ++i) {
    x(i, 0) = 0;
  }
  while (Norm((A * x) - b) / Norm(b) >= eps * eps) {
    x = ChangingX(x, A, b);
  }
  return x;
}

Matrix ChangingX(Matrix& x, Matrix& A, Matrix& b) {
  return x - (((A * x) - b) * tau);
}

double Norm(Matrix vec) {
  if (vec.get_cols() != 1) throw std::out_of_range("Error: more than 1 col");
  double res = 0;
  for (int i = 0; i < vec.get_rows(); ++i) {
    res += vec(i, 0) * vec(i, 0);
  }
  return res;
}