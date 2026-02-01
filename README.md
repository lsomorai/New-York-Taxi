# NYC Taxi Fare Prediction

A machine learning project that predicts taxi fare amounts in New York City using trip data. Implements and compares four regression models (SVM, Ridge, Random Forest, Gradient Boosting) using PySpark and scikit-learn.

[![CI](https://github.com/lsomorai/New-York-Taxi/actions/workflows/ci.yml/badge.svg)](https://github.com/lsomorai/New-York-Taxi/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/lsomorai/New-York-Taxi.git
cd New-York-Taxi

# Install dependencies
pip install -e ".[dev]"

# Run tests
make test

# Run the notebook
jupyter notebook New-York-Taxi.ipynb
```

### Data Setup

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/microize/newyork-yellow-taxi-trip-data-2020-2019) and place `yellow_tripdata_2019-06.csv` in the project root.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Results Summary](#results-summary)
3. [Project Structure](#project-structure)
4. [Data Pipeline](#data-pipeline)
5. [Models](#models)
6. [Usage](#usage)
7. [Contributing](#contributing)

---

## Problem Statement

**How can the cost of a taxi trip be predicted using the time, pickup location, and drop-off location?**

Accurate fare prediction enables:
- Upfront pricing for customer trust
- Optimized traffic predictions to reduce wait times
- Better resource allocation for vehicles and drivers

---

## Results Summary

| Model | R² Score | RMSE | Training Speed |
|-------|----------|------|----------------|
| **SVM (Linear)** | **0.810** | **6.47** | Fast |
| Ridge Regression | 0.767 | 7.46 | Fast |
| Gradient Boosting | 0.756 | 7.55 | Moderate |
| Random Forest | 0.709 | 8.29 | Slow |

**Recommendation**: SVM with linear kernel provides the best balance of accuracy and training speed.

**Key Finding**: Trip distance is the strongest predictor of fare amount, with temporal features (day of week, hour) providing secondary signals.

---

## Project Structure

```
New-York-Taxi/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration constants
│   ├── preprocessing.py   # Data cleaning and feature engineering
│   └── models.py          # Model training and evaluation
├── tests/
│   ├── test_preprocessing.py
│   └── test_models.py
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI pipeline
├── New-York-Taxi.ipynb    # Main analysis notebook
├── taxi+_zone_lookup.csv  # NYC zone reference data
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Data Pipeline

### Source
- **Dataset**: NYC Yellow Taxi Trip Data (June 2019)
- **Source**: [Kaggle](https://www.kaggle.com/datasets/microize/newyork-yellow-taxi-trip-data-2020-2019)
- **Sample**: 1% (~70,000 records) for development

### Features Used

| Feature | Description |
|---------|-------------|
| `pickup_hour` | Hour of pickup (0-23) |
| `pickup_day_of_week` | Day name (Monday-Sunday) |
| `passenger_count` | Number of passengers |
| `trip_distance` | Distance in miles |
| `pickup_location` | NYC taxi zone name |
| `dropoff_location` | NYC taxi zone name |

### Transformations

1. **Location Mapping**: Convert location IDs to zone names using lookup table
2. **Time Extraction**: Extract hour and day of week from timestamps
3. **Cyclic Encoding**: Transform day of week to sin/cos features to capture cyclical patterns
4. **Normalization**: StandardScaler applied to numerical features

---

## Models

### 1. Support Vector Machine (SVM)

- **Algorithm**: SVR with linear kernel
- **Hyperparameters**: Grid search over C values [0.5, 1.0]
- **Cross-validation**: 3-fold
- **Performance**: R² = 0.810, RMSE = 6.47

Best overall performance with fast training. Linear kernel captures the approximately linear relationship between distance and fare.

### 2. Ridge Regression

- **Algorithm**: Linear regression with L2 regularization
- **Hyperparameters**: α = 0.1
- **Performance**: R² = 0.767, RMSE = 7.46

Simple and interpretable. Accurately predicts typical fares but struggles with extreme values.

### 3. Random Forest

- **Algorithm**: Ensemble of decision trees
- **Hyperparameters**: 100 trees, max depth 7 (via grid search)
- **Cross-validation**: 3-fold
- **Performance**: R² = 0.709, RMSE = 8.29

Captures non-linear patterns but produces less smooth predictions. Long training time.

### 4. Gradient Boosting

- **Algorithm**: Sequential boosting of weak learners
- **Hyperparameters**: 50 iterations, max depth 6, learning rate 0.1
- **Performance**: R² = 0.756, RMSE = 7.55

Moderate performance with longer training time. Benefits from feature bucketization for location indices.

---

## Usage

### Using the Module

```python
from src.preprocessing import clean_dataframe, create_cyclic_features
from src.models import train_svm, compare_models

# Load and preprocess data
df = pd.read_csv("your_data.csv")
df_clean = clean_dataframe(df)
df_features = create_cyclic_features(df_clean)

# Train model
result = train_svm(X_train, y_train, X_test, y_test)
print(f"R² Score: {result['metrics']['r2']:.3f}")
```

### Development Commands

```bash
make install-dev  # Install with dev dependencies
make test         # Run test suite
make lint         # Check code style
make format       # Auto-format code
make clean        # Remove build artifacts
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Run tests (`make test`) and linting (`make lint`)
5. Commit your changes
6. Push to the branch
7. Open a Pull Request

---

## Future Improvements

- [ ] Hyperparameter optimization with Optuna
- [ ] Incorporate weather data as additional features
- [ ] Handle outliers more explicitly (negative fares, extreme values)
- [ ] Add model persistence with MLflow
- [ ] Deploy as REST API

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## References

- [NYC Taxi Trip Data - Kaggle](https://www.kaggle.com/datasets/microize/newyork-yellow-taxi-trip-data-2020-2019)
- [Apache Spark Documentation](https://spark.apache.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
