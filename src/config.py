"""Configuration constants for NYC Taxi Fare Prediction."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Data configuration
SAMPLE_FRACTION = 0.01
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.7

# Columns to drop from raw data
COLUMNS_TO_DROP = [
    "VendorID",
    "store_and_fwd_flag",
    "fare_amount",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "payment_type",
    "tip_amount",
]

# Feature columns for model training
FEATURE_COLUMNS = [
    "pickup_hour",
    "day_sin",
    "day_cos",
    "passenger_count",
    "trip_distance",
    "pickup_location_idx",
    "dropoff_location_idx",
]

TARGET_COLUMN = "total_amount"

# Model hyperparameters
SVM_CONFIG = {
    "kernel": "linear",
    "C_values": [0.5, 1.0],
    "cv_folds": 3,
}

RIDGE_CONFIG = {
    "max_iter": 10,
    "reg_param": 0.1,
    "elastic_net_param": 0.0,
}

RANDOM_FOREST_CONFIG = {
    "num_trees": [50, 100],
    "max_depth": [5, 7],
    "max_bins": 256,
    "cv_folds": 3,
}

GBT_CONFIG = {
    "max_depth": 6,
    "max_bins": 32,
    "step_size": 0.1,
    "max_iter": 50,
    "subsampling_rate": 0.8,
}
