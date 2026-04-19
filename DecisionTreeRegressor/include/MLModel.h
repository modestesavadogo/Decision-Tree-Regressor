#ifndef MLMODEL_H
#define MLMODEL_H

#include "Matrix.h"

/**
 * @brief Abstract base class for all machine learning models.
 */
class MLModel {
public:
    /**
     * @brief Train the model on given features and targets.
     * @param X Feature matrix (rows = samples, cols = features).
     * @param y Target matrix (rows = samples, cols = 1).
     */
    virtual void fit(const Matrix& X, const Matrix& y) = 0;

    /**
     * @brief Predict targets for new samples.
     * @param X Feature matrix.
     * @return Matrix of predictions (rows = samples, cols = 1).
     */
    virtual Matrix predict(const Matrix& X) = 0;          // non‑const (as per assignment)

    /**
     * @brief Compute a performance score (e.g., R² for regression).
     * @param X Features.
     * @param y True targets.
     * @return Score value.
     */
    virtual double score(const Matrix& X, const Matrix& y) = 0; // non‑const

    virtual ~MLModel() = default;
};

#endif // MLMODEL_H