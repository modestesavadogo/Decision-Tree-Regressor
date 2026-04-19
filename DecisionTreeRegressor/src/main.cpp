#include <iostream>
#include <iomanip>
#include <memory>
#include <random>
#include "CSVReader.h"
#include "DecisionTreeRegressor.h"
#include <algorithm>   // for std::shuffle
#include <numeric>     // for std::iota

/**
 * @brief Split a dataset into training and testing sets.
 * @param X Feature matrix.
 * @param y Target matrix.
 * @param train_ratio Fraction of data to use for training.
 * @return tuple (X_train, y_train, X_test, y_test)
 */
std::tuple<Matrix, Matrix, Matrix, Matrix> trainTestSplit(const Matrix& X, const Matrix& y,
                                                           double train_ratio = 0.8) {
    size_t n = X.rows();
    size_t n_train = static_cast<size_t>(n * train_ratio);

    // Create random permutation of indices
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    // Build train and test matrices
    Matrix X_train(n_train, X.cols());
    Matrix y_train(n_train, 1);
    Matrix X_test(n - n_train, X.cols());
    Matrix y_test(n - n_train, 1);

    for (size_t i = 0; i < n_train; ++i) {
        size_t idx = indices[i];
        for (size_t j = 0; j < X.cols(); ++j)
            X_train.at(i, j) = X.at(idx, j);
        y_train.at(i, 0) = y.at(idx, 0);
    }
    for (size_t i = n_train; i < n; ++i) {
        size_t idx = indices[i];
        for (size_t j = 0; j < X.cols(); ++j)
            X_test.at(i - n_train, j) = X.at(idx, j);
        y_test.at(i - n_train, 0) = y.at(idx, 0);
    }

    return {X_train, y_train, X_test, y_test};
}

int main() {
    try {
        // Load dataset (example: Auto MPG)
        // Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
        // Convert to CSV with header, e.g., auto-mpg.csv
        Matrix data = CSVReader::load("../data/preprocessed_insurance.csv", true);

        // Assume last column is target (MPG), all others are features
        size_t n_features = data.cols() - 1;
        Matrix X(data.rows(), n_features);
        Matrix y(data.rows(), 1);
        for (size_t i = 0; i < data.rows(); ++i) {
            for (size_t j = 0; j < n_features; ++j)
                X.at(i, j) = data.at(i, j);
            y.at(i, 0) = data.at(i, n_features);
        }

        // Split into train/test
        auto [X_train, y_train, X_test, y_test] = trainTestSplit(X, y, 0.8);

        // Create and train decision tree
        DecisionTreeRegressor tree(10, 5, 2, 0); // max_depth=10, min_samples_split=5, min_samples_leaf=2, all features
        tree.fit(X_train, y_train);

        // Evaluate
        double train_r2 = tree.score(X_train, y_train);
        double test_r2 = tree.score(X_test, y_test);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Training R² score: " << train_r2 << "\n";
        std::cout << "Testing R² score:  " << test_r2 << "\n";

        // Optional: predict and show some comparisons
        Matrix y_pred = tree.predict(X_test);
        std::cout << "\nSample predictions (true vs predicted):\n";
        for (size_t i = 0; i < std::min<size_t>(10, y_test.rows()); ++i) {
            std::cout << "  " << y_test.at(i, 0) << " -> " << y_pred.at(i, 0) << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}