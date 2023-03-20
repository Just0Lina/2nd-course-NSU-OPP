#include <iostream>
#include <vector>

double* simple_iteration_method(double* A, double* b);
void getNextX(const double* Axb, double* x);
void getAxb(const double* A, const double* x, const double* b, double* Axb);
double Norm(double* vec);

using namespace std;
constexpr double eps = 0.00001f, tau = 0.0001f;
constexpr int N = 1024 * 16;
int main(int argc, char* argv[]) {
  double* A = new double[N * N];
  double *X = new double[N], *b = new double[N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = 1.0f;
    }
    A[i * N + i] = 2.0f;
  }

  for (int i = 0; i < N; ++i) {
    b[i] = N + 1;
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

double* simple_iteration_method(double* A, double* b) {
  double* x = new double[N];
  double* Axb = new double[N];

  for (int i = 0; i < N; ++i) {
    x[i] = 0;
  }
  int iCount = 0;
  double acc = eps * eps;
  int bNorm = Norm(b);
  for (; acc >= eps * eps; ++iCount) {
    getAxb(A, x, b, Axb);
    getNextX(Axb, x);
    double Axb_norm = Norm(Axb);
    std::cout << Axb_norm << "\n";

    acc = Axb_norm / bNorm;
  }
  std::cout << iCount;
  return x;
}

inline void getNextX(const double* Axb, double* x) {
  for (int i = 0; i < N; ++i) x[i] = x[i] - tau * Axb[i];
}

void getAxb(const double* A, const double* x, const double* b, double* Axb) {
  for (int i = 0; i < N; ++i) {
    Axb[i] = -b[i];
    for (int j = 0; j < N; ++j) Axb[i] += A[i * N + j] * x[j];
  }
}

double Norm(double* vec) {
  double res = 0;
  for (int i = 0; i < N; ++i) {
    res += vec[i] * vec[i];
  }
  return res;
}

// #include <mpi.h>
// #include <omp.h>
// #include <sys/time.h>
// #include <time.h>

// #include <iostream>
// #include <vector>

// #include "../../double*.h"
// void simple_iteration_method(double* A, double* b, double* x);
// double* ChangingX(double* x, double* A);
// double Norm(double* vec);

// using namespace std;
// constexpr double eps = 0.00001f, tau = 0.0001f;
// constexpr int N = 1024 * 13;
// int main(int argc, char* argv[]) {
//   double* A(N, N);
//   double* X(N, 1), b(N, 1);

// #pragma omp parallel for collapse(2)
//   for (int i = 0; i < N; i++) {
//     for (int j = 0; j < N; j++) {
//       A(i, j) = 1.0f;
//     }
//   }
// #pragma omp parallel for
//   for (int i = 0; i < N; ++i) {
//     b(i, 0) = N + 1;
//     A(i, i) = 2.0f;
//     X(i, 0) = 0;
//   }
//   struct timespec start, end;
//   clock_gettime(CLOCK_MONOTONIC_RAW, &start);
//   simple_iteration_method(A, b, X);
//   clock_gettime(CLOCK_MONOTONIC_RAW, &end);
//   // X.show_matrix();
//   std::cout << (float)(end.tv_sec - start.tv_sec) +
//                    (1e-9 * (end.tv_nsec - start.tv_nsec))
//             << std::endl;
//   return 0;
// }

// void simple_iteration_method(double* A, double* b, double* x) {
//   double* Axb = (A * x) - b;
//   double normB = Norm(b);

//   int i = 0;
//   while (Norm(Axb) / normB >= eps * eps) {
//     x = ChangingX(x, Axb);
//     Axb = (A * x) - b;
//     ++i;
//   }
//   std::cout << i << std::endl;
// }

// inline double* ChangingX(double* x, double* Axb) { return x - (Axb * tau); }

// double Norm(double* vec) {
//   if (vec.get_cols() != 1) throw std::out_of_range("Error: more than 1 col");
//   double res = 0;
// #pragma omp parallel for reduction(+ : res)
//   for (int i = 0; i < vec.get_rows(); ++i) {
//     res += vec(i, 0) * vec(i, 0);
//   }
//   return res;
// }

// #include <math.h>
// #include <omp.h>
// #include <stdio.h>
// #include <time.h>

// #include <cstdlib>
// #include <iostream>

// constexpr double eps = 0.00001f, tau = 0.0001f;
// constexpr int N = 2048 * 2;

// using namespace std;

// void Multiply(double *A, double *x, double *res) {
//   int i, j;
//   for (i = 0; i < N; i++) {
//     for (j = 0; j < N; j++) {
//       res[i] += A[i * N + j] * x[j];
//     }
//   }
//   return;
// }

// void SubVectors(double *a, double *b) {
//   for (int i = 0; i < N; i++) {
//     a[i] = a[i] - b[i];
//   }
// }

// void MulConst(double value, double *a) {
//   for (int i = 0; i < N; i++) {
//     a[i] = a[i] * value;
//   }
// }

// void Initialize(double value, double *a, int N) {
//   for (int i = 0; i < N; i++) {
//     a[i] = value;
//   }
// }
// void InitializeDiagonal(double value, double *a) {
//   for (int i = 0; i < N; i++) {
//     a[i * N + i] = value;
//   }
// }

// void PrintVector(double *vec, int N) {
//   for (int i = 0; i < N; i++) {
//     printf("%f ", vec[i]);
//   }
//   printf("\n");
// }

// void PrintMatrix(double *mat, int N) {
//   for (int k = 0; k < N; k++) {
//     for (int i = 0; i < N; i++) {
//       printf("%f ", mat[k * N + i]);
//     }
//     // printf("\n");
//   }
// }

// double Norm(double *vec) {
//   double sum = 0;
//   for (int i = 0; i < N; i++) {
//     sum += vec[i] * vec[i];
//   }
//   return sum;
// }

// void Iteration(double *A, double *x, double *b) {
//   double res[N];
//   Multiply(A, x, res);
//   SubVectors(res, b);
//   MulConst(tau, res);
//   SubVectors(x, res);
//   return;
// }

// void simple_iteration_method(double *A, double *x, double *b) {
//   double *res = new double[N];
//   double normB = Norm(b);
//   // double dif = 0.001;
//   int i = 0;
//   do {
//     Multiply(A, x, res);
//     SubVectors(res, b);
//     Iteration(A, x, b);
//     i++;
//     // std::cout << i << std::endl;
//   } while (Norm(res) / normB >= eps * eps);
//   // std::cout << i << std::endl;
// }

// int main() {
//   double *A = new double[N * N];
//   double *x = new double[N];
//   double *b = new double[N];

//   Initialize(1.0, A, N * N);
//   InitializeDiagonal(2.0, A);

//   Initialize(N + 1, b, N);
//   Initialize(0.0, x, N);
//   struct timespec start, end;
//   clock_gettime(CLOCK_MONOTONIC_RAW, &start);
//   simple_iteration_method(A, x, b);
//   clock_gettime(CLOCK_MONOTONIC_RAW, &end);
//   std::cout << (float)(end.tv_sec - start.tv_sec) +
//                    (1e-9 * (end.tv_nsec - start.tv_nsec))
//             << std::endl;

//   // printf("%f\n", x[0]);
//   return 0;
// }

// #include <mpi.h>
// #include <omp.h>
// #include <sys/time.h>
// #include <time.h>

// #include <iostream>
// #include <vector>

// #include "../../double*.h"
// void simple_iteration_method(double* A, double* b, double* x);
// double* ChangingX(double* x, double* A);
// double Norm(double* vec);

// using namespace std;
// constexpr double eps = 0.00001f, tau = 0.0001f;
// constexpr int N = 1024 * 13;
// int main(int argc, char* argv[]) {
//   double* A(N, N);
//   double* X(N, 1), b(N, 1);

// #pragma omp parallel for collapse(2)
//   for (int i = 0; i < N; i++) {
//     for (int j = 0; j < N; j++) {
//       A(i, j) = 1.0f;
//     }
//   }
// #pragma omp parallel for
//   for (int i = 0; i < N; ++i) {
//     b(i, 0) = N + 1;
//     A(i, i) = 2.0f;
//     X(i, 0) = 0;
//   }
//   struct timespec start, end;
//   clock_gettime(CLOCK_MONOTONIC_RAW, &start);
//   simple_iteration_method(A, b, X);
//   clock_gettime(CLOCK_MONOTONIC_RAW, &end);
//   // X.show_matrix();
//   std::cout << (float)(end.tv_sec - start.tv_sec) +
//                    (1e-9 * (end.tv_nsec - start.tv_nsec))
//             << std::endl;
//   return 0;
// }

// void simple_iteration_method(double* A, double* b, double* x) {
//   double* Axb = (A * x) - b;
//   double normB = Norm(b);

//   int i = 0;
//   while (Norm(Axb) / normB >= eps * eps) {
//     x = ChangingX(x, Axb);
//     Axb = (A * x) - b;
//     ++i;
//   }
//   std::cout << i << std::endl;
// }

// inline double* ChangingX(double* x, double* Axb) { return x - (Axb * tau); }

// double Norm(double* vec) {
//   if (vec.get_cols() != 1) throw std::out_of_range("Error: more than 1 col");
//   double res = 0;
// #pragma omp parallel for reduction(+ : res)
//   for (int i = 0; i < vec.get_rows(); ++i) {
//     res += vec(i, 0) * vec(i, 0);
//   }
//   return res;
// }

// #include <math.h>
// #include <omp.h>
// #include <stdio.h>
// #include <time.h>

// #include <cstdlib>
// #include <iostream>

// constexpr double eps = 0.00001f, tau = 0.0001f;
// constexpr int N = 2048 * 2;

// using namespace std;

// void Multiply(double *A, double *x, double *res) {
//   int i, j;
//   for (i = 0; i < N; i++) {
//     for (j = 0; j < N; j++) {
//       res[i] += A[i * N + j] * x[j];
//     }
//   }
//   return;
// }

// void SubVectors(double *a, double *b) {
//   for (int i = 0; i < N; i++) {
//     a[i] = a[i] - b[i];
//   }
// }

// void MulConst(double value, double *a) {
//   for (int i = 0; i < N; i++) {
//     a[i] = a[i] * value;
//   }
// }

// void Initialize(double value, double *a, int N) {
//   for (int i = 0; i < N; i++) {
//     a[i] = value;
//   }
// }
// void InitializeDiagonal(double value, double *a) {
//   for (int i = 0; i < N; i++) {
//     a[i * N + i] = value;
//   }
// }

// void PrintVector(double *vec, int N) {
//   for (int i = 0; i < N; i++) {
//     printf("%f ", vec[i]);
//   }
//   printf("\n");
// }

// void PrintMatrix(double *mat, int N) {
//   for (int k = 0; k < N; k++) {
//     for (int i = 0; i < N; i++) {
//       printf("%f ", mat[k * N + i]);
//     }
//     // printf("\n");
//   }
// }

// double Norm(double *vec) {
//   double sum = 0;
//   for (int i = 0; i < N; i++) {
//     sum += vec[i] * vec[i];
//   }
//   return sum;
// }

// void Iteration(double *A, double *x, double *b) {
//   double res[N];
//   Multiply(A, x, res);
//   SubVectors(res, b);
//   MulConst(tau, res);
//   SubVectors(x, res);
//   return;
// }

// void simple_iteration_method(double *A, double *x, double *b) {
//   double *res = new double[N];
//   double normB = Norm(b);
//   // double dif = 0.001;
//   int i = 0;
//   do {
//     Multiply(A, x, res);
//     SubVectors(res, b);
//     Iteration(A, x, b);
//     i++;
//     // std::cout << i << std::endl;
//   } while (Norm(res) / normB >= eps * eps);
//   // std::cout << i << std::endl;
// }

// int main() {
//   double *A = new double[N * N];
//   double *x = new double[N];
//   double *b = new double[N];

//   Initialize(1.0, A, N * N);
//   InitializeDiagonal(2.0, A);

//   Initialize(N + 1, b, N);
//   Initialize(0.0, x, N);
//   struct timespec start, end;
//   clock_gettime(CLOCK_MONOTONIC_RAW, &start);
//   simple_iteration_method(A, x, b);
//   clock_gettime(CLOCK_MONOTONIC_RAW, &end);
//   std::cout << (float)(end.tv_sec - start.tv_sec) +
//                    (1e-9 * (end.tv_nsec - start.tv_nsec))
//             << std::endl;

//   // printf("%f\n", x[0]);
//   return 0;
// }
