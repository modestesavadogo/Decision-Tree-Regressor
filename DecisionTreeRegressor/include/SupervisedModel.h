#ifndef SUPERVISEDMODEL_H
#define SUPERVISEDMODEL_H

#include "MLModel.h"

/**
 * @brief Tag class for supervised learning models.
 */
class SupervisedModel : public MLModel {
public:
    ~SupervisedModel() override = default;
};

#endif // SUPERVISEDMODEL_H