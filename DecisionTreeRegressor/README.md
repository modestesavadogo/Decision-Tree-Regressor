# DecisionTreeRegressor — C++ from Scratch

A clean, header-only-friendly implementation of a **Decision Tree Regressor** in modern C++17, built without any external ML libraries. Splits nodes by **variance reduction** (equivalent to MSE reduction) and exposes a scikit-learn-style API (`fit` / `predict` / `score`).

---

## Features

- Pure C++17, zero ML-library dependencies
- Variance-reduction splitting with **cumulative-sum optimisation** (O(n log n) per feature)
- Configurable hyperparameters: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- Random feature subsampling (enables Random-Forest-style ensembling)
- Abstract OOP hierarchy: `MLModel → SupervisedModel → Regressor → DecisionTreeRegressor`
- Lightweight `Matrix` class and `CSVReader` utility included
- R² score evaluation out of the box

---

## Project Structure

```
DecisionTreeRegressor/
├── CMakeLists.txt
├── include/
│   ├── MLModel.h               # Abstract base: fit / predict / score
│   ├── SupervisedModel.h       # Supervised learning interface
│   ├── Regressor.h             # Regressor interface
│   ├── DecisionTreeRegressor.h # Tree class declaration
│   ├── Matrix.h                # 2-D matrix class
│   └── CSVReader.h             # CSV loading utility
├── src/
│   ├── DecisionTreeRegressor.cpp
│   ├── Matrix.cpp
│   ├── CSVReader.cpp
│   └── main.cpp                # Demo: insurance charges prediction
├── data/
│   ├── insurance.csv           # Raw dataset
│   ├── preprocessed_insurance.csv
│   └── data_preprocessing.py   # Preprocessing script (pandas)
├── tests/                      # Unit tests (to be added)
└── uml/
    └── class_diagram.uml       # Class hierarchy diagram
```

---

## Build

**Requirements:** CMake ≥ 3.10, a C++17-capable compiler (GCC 7+, Clang 5+, MSVC 2017+).

```bash
git clone https://github.com/<your-username>/DecisionTreeRegressor.git
cd DecisionTreeRegressor
mkdir build && cd build
cmake ..
make
```

The executable is placed at `build/dtree_regressor`.

---

## Quick Start

```bash
./dtree_regressor
```

Expected output (values vary due to random train/test split):

```
Training R² score: 0.9821
Testing  R² score: 0.7043

Sample predictions (true vs predicted):
  12345.00 -> 11980.00
  ...
```

---

## API

```cpp
#include "DecisionTreeRegressor.h"

// Hyperparameters
DecisionTreeRegressor tree(
    /* max_depth        */ 10,
    /* min_samples_split*/ 5,
    /* min_samples_leaf */ 2,
    /* max_features     */ 0   // 0 = use all features
);

tree.fit(X_train, y_train);           // train
Matrix y_pred = tree.predict(X_test); // inference
double r2     = tree.score(X_test, y_test); // R²
```

### Hyperparameter guide

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | `10` | Maximum tree depth (`0` = unlimited) |
| `min_samples_split` | `2` | Minimum samples to attempt a split |
| `min_samples_leaf` | `1` | Minimum samples required in each leaf |
| `max_features` | `0` | Features considered per split (`0` = all) |

---

## Dataset

The demo uses the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (`insurance.csv`). The preprocessing script one-hot encodes categorical features (sex, smoker, region) and standardises numerical ones.

```bash
cd data
python data_preprocessing.py   # produces preprocessed_insurance.csv
```

---

## Class Hierarchy

```
MLModel  (abstract)
└── SupervisedModel  (abstract)
    └── Regressor  (abstract)
        └── DecisionTreeRegressor  ✓
```

---

## License

MIT