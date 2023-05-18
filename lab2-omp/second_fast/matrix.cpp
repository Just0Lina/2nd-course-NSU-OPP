#include "../matrix.h"

Matrix::Matrix() : _rows(0), _cols(0), _matrix(nullptr) {}

Matrix::Matrix(int rows, int cols) : _rows(rows), _cols(cols) {
  memory_allocation();
}

Matrix::Matrix(const Matrix& other) : _rows(other._rows), _cols(other._cols) {
  memory_allocation();
  copy_matrix(other);
}

Matrix::Matrix(Matrix&& other)
    : _rows(other._rows), _cols(other._cols), _matrix(other._matrix) {
  other._matrix = nullptr;
  other._cols = 0;
  other._rows = 0;
}

void Matrix::set_rows(int rows) {
  Matrix tmp(rows, _cols);
  tmp.copy_matrix((rows > _rows) ? _rows : rows, _cols, *this);
  *this = tmp;
}

void Matrix::set_cols(int cols) {
  Matrix tmp(_rows, cols);
  tmp.copy_matrix(_rows, (cols > _cols) ? _cols : cols, *this);
  *this = tmp;
}

void Matrix::copy_matrix(const int rows, const int cols, const Matrix& other) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      _matrix[i][j] = other._matrix[i][j];
    }
  }
}

int Matrix::get_rows() { return _rows; }

int Matrix::get_cols() { return _cols; }

void Matrix::copy_matrix(const Matrix& other) {
  copy_matrix(_rows, _cols, other);
}

void Matrix::memory_allocation() {
  if (_rows <= 0 || _cols <= 0) {
    throw std::out_of_range("Error: out of range");
  }
  _matrix = new double*[_rows]();
  for (int i = 0; i < _rows; ++i) _matrix[i] = new double[_cols]();
}

Matrix::~Matrix() { destroy_matrix(); }

void Matrix::destroy_matrix() {
  for (int i = 0; i < _rows; ++i) {
    delete[] _matrix[i];
  }
  delete[] _matrix;
  _matrix = nullptr;
  _cols = 0;
  _rows = 0;
}

bool Matrix::eq_matrix(const Matrix& other) {
  bool result = true;
  if (other._cols == _cols && other._rows == _rows) {
    for (int i = 0; i < _rows && result; ++i) {
      for (int j = 0; j < _cols; ++j) {
        if (fabs(_matrix[i][j] - other._matrix[i][j]) >= E) {
          result = false;
          break;
        }
      }
    }
  } else {
    result = false;
  }
  return result;
}

void Matrix::sum_matrix(const Matrix& other) {
  if (_rows != other._rows || _cols != other._cols) {
    throw std::out_of_range(
        "Incorrect input, matrices should have the same size");
  }

  for (int i = 0; i < _rows; ++i) {
    for (int j = 0; j < _cols; ++j) {
      _matrix[i][j] += other._matrix[i][j];
    }
  }
}

void Matrix::sub_matrix(const Matrix& other) {
  Matrix tmp(other);
  tmp.mul_number(-1);
  sum_matrix(tmp);
}

void Matrix::mul_number(const double num) {
  for (int i = 0; i < _rows; ++i) {
    for (int j = 0; j < _cols; ++j) {
      _matrix[i][j] *= num;
    }
  }
}

void Matrix::mul_matrix(const Matrix& other) {
  if (_cols != other._rows) {
    throw std::out_of_range("Incorrect input, 1st cols != 2nd rows");
  }
  Matrix result(_rows, other._cols);

  for (int i = 0; i < _rows; ++i) {
    for (int j = 0; j < other._cols; ++j) {
      for (int k = 0; k < _cols; ++k) {
        result._matrix[i][j] += _matrix[i][k] * other._matrix[k][j];
      }
    }
  }
  *this = result;
}

Matrix Matrix::transpose() {
  Matrix transposed(_cols, _rows);
  for (int i = 0; i < _cols; ++i) {
    for (int j = 0; j < _rows; ++j) {
      transposed._matrix[i][j] = _matrix[j][i];
    }
  }
  return transposed;
}

Matrix Matrix::calc_complements() {
  if (_rows != _cols) {
    throw std::out_of_range("Incorrect input, matrix is not square");
  }
  Matrix result(_rows, _cols);
  if (_rows == 1) {
    result._matrix[0][0] = 1;
  } else {
    for (int i = 0; i < _rows; ++i) {
      for (int j = 0; j < _cols; ++j) {
        Matrix tmp = (*this).minor_matrix(i, j);
        int sign = ((i + j) % 2) ? -1 : 1;
        result._matrix[i][j] = tmp.determinant() * sign;
      }
    }
  }
  return result;
}

double Matrix::determinant() {
  if (_rows != _cols) {
    throw std::out_of_range("Incorrect input, matrix is not square");
  }
  double result;
  if (_rows == 1) {
    result = _matrix[0][0];
  } else if (_rows == 2) {
    result = _matrix[0][0] * _matrix[1][1] - _matrix[1][0] * _matrix[0][1];
  } else {
    for (int i = 0; i < _rows; ++i) {
      Matrix tmp = minor_matrix(0, i);
      int sign = (i % 2) ? -1 : 1;
      result += _matrix[0][i] * tmp.determinant() * sign;
    }
  }
  return result;
}

Matrix Matrix::minor_matrix(int row, int col) {
  Matrix result(_rows - 1, _cols - 1);
  for (int i = 0, x = 0; i < _rows; ++i) {
    if (i == row) continue;
    for (int j = 0, y = 0; j < _cols; ++j) {
      if (j == col) continue;
      result._matrix[x][y] = _matrix[i][j];
      y++;
    }
    x++;
  }
  return result;
}

Matrix Matrix::inverse_matrix() {
  double det = determinant();
  if (!det || _rows != _cols) {
    throw std::out_of_range("Error: determinant = 0");
  }
  Matrix temp = calc_complements();
  Matrix result = temp.transpose();
  result.mul_number(1 / det);
  return result;
}

void Matrix::fill_matrix(const char* str) {
  int n = 0;
  for (int i = 0; i < _rows; ++i) {
    for (int j = 0; j < _cols; ++j) {
      sscanf(str += n, "%lf%n", &_matrix[i][j], &n);
    }
  }
}

double& Matrix::operator()(int row, int col) {
  if (row >= _rows || col >= _cols || row < 0 || col < 0) {
    throw std::out_of_range("Incorrect input, row or col is incorrect");
  }
  return _matrix[row][col];
}

bool Matrix::operator==(const Matrix& other) { return eq_matrix(other); }

Matrix& Matrix::operator+=(const Matrix& other) {
  sum_matrix(other);
  return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
  sub_matrix(other);
  return *this;
}
Matrix& Matrix::operator*=(const Matrix& other) {
  mul_matrix(other);
  return *this;
}
Matrix& Matrix::operator*=(const double num) {
  mul_number(num);
  return *this;
}

Matrix Matrix::operator*(const Matrix& other) {
  Matrix result(*this);
  return result *= other;
}

Matrix operator*(const double num, const Matrix& other) {
  Matrix result(other);
  return result *= num;
}

Matrix operator*(const Matrix& other, const double num) {
  Matrix result(other);
  return result *= num;
}

Matrix Matrix::operator+(const Matrix& other) {
  Matrix result(*this);
  return result += other;
}

Matrix Matrix::operator-(const Matrix& other) {
  Matrix result(*this);
  return result -= other;
}

Matrix& Matrix::operator=(const Matrix& other) {
  if (this == &other) {
    throw std::out_of_range("Sent the same matrix");
  }
  if (_rows != other._rows || _cols != other._cols) {
    destroy_matrix();
    _rows = other._rows;
    _cols = other._cols;
    memory_allocation();
  }
  copy_matrix(other);
  return *this;
}
