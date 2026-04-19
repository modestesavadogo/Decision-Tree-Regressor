#ifndef DECISIONTREEREGRESSOR_H
#define DECISIONTREEREGRESSOR_H

#include "Regressor.h"
#include <memory>
#include <vector>
#include <random>

/**
 * @brief Decision Tree for regression.
 *
 * Splits nodes by maximising variance reduction.
 * Supports early stopping via max_depth, min_samples_split, min_samples_leaf.
 */
class DecisionTreeRegressor : public Regressor {
private:
    // Internal tree node
    struct Node {
        bool is_leaf = false;
        double value = 0.0;                // predicted value (mean of targets) for leaf
        size_t feature_idx = 0;             // feature used for splitting
        double threshold = 0.0;              // threshold for split (feature <= threshold)
        std::unique_ptr<Node> left = nullptr;
        std::unique_ptr<Node> right = nullptr;

        Node() = default;
        explicit Node(double val) : is_leaf(true), value(val) {}
    };

    std::unique_ptr<Node> root_;
    size_t max_depth_;
    size_t min_samples_split_;
    size_t min_samples_leaf_;
    size_t max_features_;                // number of features to consider at each split (0 = all)
    size_t n_features_;
    mutable std::mt19937 rng_;            // for random feature selection

    // Helper functions
    std::unique_ptr<Node> buildTree(const std::vector<std::vector<double>>& X,
                                     const std::vector<double>& y,
                                     size_t depth);
    double predictSample(const std::vector<double>& x) const;  // const helper
    double computeVariance(const std::vector<double>& y) const;
    double computeVarianceReduction(const std::vector<double>& y,
                                    const std::vector<bool>& left_mask) const;
    std::vector<size_t> getCandidateFeatures() const;

public:
    /**
     * @brief Constructor with hyperparameters.
     * @param max_depth Maximum tree depth (0 = unlimited).
     * @param min_samples_split Minimum samples required to split a node.
     * @param min_samples_leaf Minimum samples required in a leaf.
     * @param max_features Number of features to consider for best split (0 = all).
     */
    DecisionTreeRegressor(size_t max_depth = 10,
                          size_t min_samples_split = 2,
                          size_t min_samples_leaf = 1,
                          size_t max_features = 0);

    void fit(const Matrix& X, const Matrix& y) override;
    Matrix predict(const Matrix& X) override;                // non‑const override
    double score(const Matrix& X, const Matrix& y) override; // non‑const override

    // Rule of five
    ~DecisionTreeRegressor() override = default;
    DecisionTreeRegressor(const DecisionTreeRegressor&) = delete;
    DecisionTreeRegressor& operator=(const DecisionTreeRegressor&) = delete;
    DecisionTreeRegressor(DecisionTreeRegressor&&) = default;
    DecisionTreeRegressor& operator=(DecisionTreeRegressor&&) = default;
};

#endif // DECISIONTREEREGRESSOR_H