
#ifndef S21_MATRIX_OOP_H_
#define S21_MATRIX_OOP_H_

#include <cmath>
#include <iostream>
#define E 1e-7

class Matrix {
 public:
  // МЕТОДЫ
  Matrix();   // консструктор
  ~Matrix();  // деструктор
  Matrix(const int rows, const int cols);
  Matrix(const Matrix& other);  // конструктор копирования
  Matrix(Matrix&& other);       // конструктор перемещения

  void set_rows(const int rows);
  void set_cols(const int cols);
  int get_rows();
  int get_cols();
  void fill_matrix(const char* str);

  bool eq_matrix(const Matrix& other);
  void sum_matrix(const Matrix& other);
  void sub_matrix(const Matrix& other);
  void mul_number(const double num);
  void mul_matrix(const Matrix& other);
  Matrix transpose();
  Matrix calc_complements();
  Matrix minor_matrix(const int row, const int col);
  double determinant();
  Matrix inverse_matrix();

  double& operator()(int row, int col);
  bool operator==(const Matrix& other);
  Matrix operator+(const Matrix& other);
  Matrix operator*(const Matrix& other);
  Matrix operator-(const Matrix& other);
  Matrix& operator=(const Matrix& other);
  Matrix& operator+=(const Matrix& other);
  Matrix& operator-=(const Matrix& other);
  Matrix& operator*=(const Matrix& other);
  Matrix& operator*=(const double num);
  void show_matrix() {
    for (int i = 0; i < _rows; ++i) {
      for (int j = 0; j < _cols; ++j) std::cout << _matrix[i][j] << " ";
      std::cout << std::endl;
    }
  }

 private:
  int _rows, _cols;
  double** _matrix;
  void copy_matrix(const Matrix& other);
  void copy_matrix(const int rows, const int cols, const Matrix& other);
  void destroy_matrix();
  void memory_allocation();
};

Matrix operator*(const double num, const Matrix& other);
Matrix operator*(const Matrix& other, const double num);

#endif
