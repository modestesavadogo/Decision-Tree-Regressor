#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstddef>
#include <stdexcept>

/**
 * @brief Simple 2D matrix class for storing numerical data.
 */
class Matrix {
private:
    std::vector<std::vector<double>> data_;
    size_t rows_;
    size_t cols_;

public:
    // Constructors
    Matrix() : rows_(0), cols_(0) {}
    Matrix(size_t rows, size_t cols, double init = 0.0);
    Matrix(const std::vector<std::vector<double>>& vec);

    // Element access
    double& at(size_t i, size_t j);
    const double& at(size_t i, size_t j) const;

    // Dimensions
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // Row/column extraction
    std::vector<double> getRow(size_t i) const;
    std::vector<double> getCol(size_t j) const;

    // Basic operations
    Matrix transpose() const;

    // Utility
    bool empty() const { return rows_ == 0 || cols_ == 0; }
    void print() const; // for debugging
};

#endif // MATRIX_H