#include "DecisionTreeRegressor.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <cassert>

DecisionTreeRegressor::DecisionTreeRegressor(size_t max_depth,
                                             size_t min_samples_split,
                                             size_t min_samples_leaf,
                                             size_t max_features)
    : max_depth_(max_depth),
      min_samples_split_(min_samples_split),
      min_samples_leaf_(min_samples_leaf),
      max_features_(max_features),
      n_features_(0),
      rng_(std::random_device{}()) {}

void DecisionTreeRegressor::fit(const Matrix& X, const Matrix& y) {
    // Basic input validation
    if (X.rows() == 0 || y.rows() == 0)
        throw std::invalid_argument("Training data cannot be empty");
    if (X.rows() != y.rows())
        throw std::invalid_argument("Number of samples in X and y must match");
    if (y.cols() != 1)
        throw std::invalid_argument("y must be a column vector (cols=1)");

    n_features_ = X.cols();

    // Convert Matrix to vectors for easier manipulation inside the tree builder
    std::vector<std::vector<double>> X_vec(X.rows());
    std::vector<double> y_vec(X.rows());
    for (size_t i = 0; i < X.rows(); ++i) {
        X_vec[i] = X.getRow(i);
        y_vec[i] = y.at(i, 0);
    }

    root_ = buildTree(X_vec, y_vec, 0);
}

std::unique_ptr<DecisionTreeRegressor::Node>
DecisionTreeRegressor::buildTree(const std::vector<std::vector<double>>& X,
                                 const std::vector<double>& y,
                                 size_t depth) {
    size_t n_samples = y.size();
    double current_var = computeVariance(y);

    // Stopping criteria
    if (n_samples < min_samples_split_ ||
        (max_depth_ > 0 && depth >= max_depth_) ||
        current_var < 1e-10) {  // pure node (variance almost zero)
        double mean = std::accumulate(y.begin(), y.end(), 0.0) / n_samples;
        return std::make_unique<Node>(mean);
    }

    // Prepare to find best split
    double best_var_red = -1.0;
    size_t best_feature = 0;
    double best_threshold = 0.0;
    std::vector<bool> best_left_mask;

    // Determine which features to consider
    std::vector<size_t> candidate_features = getCandidateFeatures();

    // Iterate over features
    for (size_t f : candidate_features) {
        // Create vector of (feature value, target) pairs and sort by feature value
        std::vector<std::pair<double, double>> sorted(n_samples);
        for (size_t i = 0; i < n_samples; ++i)
            sorted[i] = {X[i][f], y[i]};
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Cumulative sums for fast variance computation
        std::vector<double> cum_sum_y(n_samples + 1, 0.0);
        std::vector<double> cum_sum_y2(n_samples + 1, 0.0);
        for (size_t i = 0; i < n_samples; ++i) {
            cum_sum_y[i+1] = cum_sum_y[i] + sorted[i].second;
            cum_sum_y2[i+1] = cum_sum_y2[i] + sorted[i].second * sorted[i].second;
        }

        // Try all possible split points (between consecutive distinct values)
        for (size_t i = 0; i < n_samples - 1; ++i) {
            // Skip if feature values are equal (no split)
            if (sorted[i].first == sorted[i+1].first)
                continue;

            double threshold = (sorted[i].first + sorted[i+1].first) / 2.0;
            size_t left_count = i + 1;
            size_t right_count = n_samples - left_count;

            // Enforce min_samples_leaf
            if (left_count < min_samples_leaf_ || right_count < min_samples_leaf_)
                continue;

            // Compute variance of left and right using cumulative sums
            double sum_left = cum_sum_y[left_count];
            double sum2_left = cum_sum_y2[left_count];
            double sum_right = cum_sum_y[n_samples] - sum_left;
            double sum2_right = cum_sum_y2[n_samples] - sum2_left;

            double var_left = (sum2_left - sum_left * sum_left / left_count) / left_count;
            double var_right = (sum2_right - sum_right * sum_right / right_count) / right_count;

            double var_red = current_var -
                (static_cast<double>(left_count) / n_samples) * var_left -
                (static_cast<double>(right_count) / n_samples) * var_right;

            if (var_red > best_var_red) {
                best_var_red = var_red;
                best_feature = f;
                best_threshold = threshold;
            }
        }
    }

    // If no split improved variance, make a leaf
    if (best_var_red < 0) { // no valid split found
        double mean = std::accumulate(y.begin(), y.end(), 0.0) / n_samples;
        return std::make_unique<Node>(mean);
    }

    // Split the data according to best split
    std::vector<std::vector<double>> X_left, X_right;
    std::vector<double> y_left, y_right;
    X_left.reserve(n_samples);
    X_right.reserve(n_samples);
    y_left.reserve(n_samples);
    y_right.reserve(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        if (X[i][best_feature] <= best_threshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }

    // Recursively build children
    auto node = std::make_unique<Node>();
    node->is_leaf = false;
    node->feature_idx = best_feature;
    node->threshold = best_threshold;
    node->left = buildTree(X_left, y_left, depth + 1);
    node->right = buildTree(X_right, y_right, depth + 1);
    return node;
}

Matrix DecisionTreeRegressor::predict(const Matrix& X) {   // non‑const
    if (!root_)
        throw std::runtime_error("Model not fitted yet");

    Matrix result(X.rows(), 1);
    for (size_t i = 0; i < X.rows(); ++i) {
        result.at(i, 0) = predictSample(X.getRow(i));
    }
    return result;
}

double DecisionTreeRegressor::predictSample(const std::vector<double>& x) const {
    const Node* cur = root_.get();
    while (!cur->is_leaf) {
        if (x[cur->feature_idx] <= cur->threshold)
            cur = cur->left.get();
        else
            cur = cur->right.get();
    }
    return cur->value;
}

double DecisionTreeRegressor::score(const Matrix& X, const Matrix& y) { // non‑const
    if (X.rows() != y.rows())
        throw std::invalid_argument("X and y must have same number of rows");
    if (y.cols() != 1)
        throw std::invalid_argument("y must be a column vector");

    Matrix pred = predict(X);
    double sum_sq_err = 0.0;
    double sum_sq_total = 0.0;
    double mean_y = 0.0;

    // Compute mean of true values
    for (size_t i = 0; i < y.rows(); ++i)
        mean_y += y.at(i, 0);
    mean_y /= y.rows();

    for (size_t i = 0; i < y.rows(); ++i) {
        double diff = y.at(i, 0) - pred.at(i, 0);
        sum_sq_err += diff * diff;
        double diff_mean = y.at(i, 0) - mean_y;
        sum_sq_total += diff_mean * diff_mean;
    }

    if (sum_sq_total == 0.0) return 1.0; // perfect fit or constant target
    return 1.0 - (sum_sq_err / sum_sq_total);
}

double DecisionTreeRegressor::computeVariance(const std::vector<double>& y) const {
    if (y.empty()) return 0.0;
    double sum = 0.0, sum2 = 0.0;
    for (double v : y) {
        sum += v;
        sum2 += v * v;
    }
    double n = y.size();
    return (sum2 - sum * sum / n) / n;
}

double DecisionTreeRegressor::computeVarianceReduction(const std::vector<double>& y,
                                                       const std::vector<bool>& left_mask) const {
    // Not directly used in current implementation; kept for completeness.
    size_t n = y.size();
    size_t left_count = std::count(left_mask.begin(), left_mask.end(), true);
    size_t right_count = n - left_count;

    if (left_count == 0 || right_count == 0) return 0.0;

    double total_var = computeVariance(y);

    std::vector<double> y_left, y_right;
    y_left.reserve(left_count);
    y_right.reserve(right_count);
    for (size_t i = 0; i < n; ++i) {
        if (left_mask[i])
            y_left.push_back(y[i]);
        else
            y_right.push_back(y[i]);
    }

    double var_left = computeVariance(y_left);
    double var_right = computeVariance(y_right);

    return total_var -
           (static_cast<double>(left_count) / n) * var_left -
           (static_cast<double>(right_count) / n) * var_right;
}

std::vector<size_t> DecisionTreeRegressor::getCandidateFeatures() const {
    if (max_features_ == 0 || max_features_ >= n_features_) {
        // use all features
        std::vector<size_t> all(n_features_);
        std::iota(all.begin(), all.end(), 0);
        return all;
    } else {
        // randomly select max_features_ indices
        std::vector<size_t> indices(n_features_);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);
        indices.resize(max_features_);
        return indices;
    }
}