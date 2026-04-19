#ifndef REGRESSOR_H
#define REGRESSOR_H

#include "SupervisedModel.h"

/**
 * @brief Abstract base class for regression models.
 */
class Regressor : public SupervisedModel {
public:
    /**
     * @brief Compute R² score (coefficient of determination).
     * @param X Features.
     * @param y True targets.
     * @return R² value.
     */
    double score(const Matrix& X, const Matrix& y) override = 0; // non‑const

    ~Regressor() override = default;
};

#endif // REGRESSOR_H