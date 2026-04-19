#include "Matrix.h"
#include <iostream>
#include <iomanip>

Matrix::Matrix(size_t rows, size_t cols, double init)
    : rows_(rows), cols_(cols), data_(rows, std::vector<double>(cols, init)) {}

Matrix::Matrix(const std::vector<std::vector<double>>& vec) {
    if (vec.empty()) {
        rows_ = cols_ = 0;
        return;
    }
    rows_ = vec.size();
    cols_ = vec[0].size();
    for (const auto& row : vec) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
    data_ = vec;
}

double& Matrix::at(size_t i, size_t j) {
    if (i >= rows_ || j >= cols_)
        throw std::out_of_range("Matrix index out of range");
    return data_[i][j];
}

const double& Matrix::at(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_)
        throw std::out_of_range("Matrix index out of range");
    return data_[i][j];
}

std::vector<double> Matrix::getRow(size_t i) const {
    if (i >= rows_)
        throw std::out_of_range("Row index out of range");
    return data_[i];
}

std::vector<double> Matrix::getCol(size_t j) const {
    if (j >= cols_)
        throw std::out_of_range("Column index out of range");
    std::vector<double> col(rows_);
    for (size_t i = 0; i < rows_; ++i)
        col[i] = data_[i][j];
    return col;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            result.at(j, i) = data_[i][j];
    return result;
}

void Matrix::print() const {
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j)
            std::cout << std::setw(10) << data_[i][j] << " ";
        std::cout << "\n";
    }
}