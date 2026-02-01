"""Data preprocessing functions for NYC Taxi Fare Prediction."""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def clean_dataframe(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Remove non-informative columns from the dataframe.

    Args:
        df: Input pandas DataFrame
        columns_to_drop: List of column names to remove

    Returns:
        Cleaned DataFrame with specified columns removed
    """
    if columns_to_drop is None:
        columns_to_drop = [
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

    existing_cols = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=existing_cols)


def map_location_ids(
    df: pd.DataFrame,
    zone_lookup: pd.DataFrame,
    pickup_col: str = "PULocationID",
    dropoff_col: str = "DOLocationID",
) -> pd.DataFrame:
    """
    Map location IDs to zone names using a lookup table.

    Args:
        df: Input DataFrame with location IDs
        zone_lookup: DataFrame with LocationID to Zone mapping
        pickup_col: Name of pickup location column
        dropoff_col: Name of dropoff location column

    Returns:
        DataFrame with location names instead of IDs
    """
    location_dict = dict(zip(zone_lookup["LocationID"], zone_lookup["Zone"]))

    df = df.copy()
    df["pickup_location"] = df[pickup_col].map(location_dict).fillna("Unknown")
    df["dropoff_location"] = df[dropoff_col].map(location_dict).fillna("Unknown")

    return df.drop(columns=[pickup_col, dropoff_col])


def extract_time_features(
    df: pd.DataFrame,
    pickup_datetime_col: str = "tpep_pickup_datetime",
    dropoff_datetime_col: str = "tpep_dropoff_datetime",
) -> pd.DataFrame:
    """
    Extract hour and day of week from datetime columns.

    Args:
        df: Input DataFrame with datetime columns
        pickup_datetime_col: Name of pickup datetime column
        dropoff_datetime_col: Name of dropoff datetime column

    Returns:
        DataFrame with extracted time features
    """
    df = df.copy()

    # Ensure datetime type
    df[pickup_datetime_col] = pd.to_datetime(df[pickup_datetime_col])
    df[dropoff_datetime_col] = pd.to_datetime(df[dropoff_datetime_col])

    # Extract features
    df["pickup_date"] = df[pickup_datetime_col].dt.date
    df["pickup_hour"] = df[pickup_datetime_col].dt.hour
    df["pickup_day_of_week"] = df[pickup_datetime_col].dt.day_name()

    df["dropoff_date"] = df[dropoff_datetime_col].dt.date
    df["dropoff_hour"] = df[dropoff_datetime_col].dt.hour
    df["dropoff_day_of_week"] = df[dropoff_datetime_col].dt.day_name()

    return df.drop(columns=[pickup_datetime_col, dropoff_datetime_col])


def create_cyclic_features(
    df: pd.DataFrame,
    day_col: str = "pickup_day_of_week",
) -> pd.DataFrame:
    """
    Create cyclic (sin/cos) features for temporal data.

    Cyclic encoding prevents the model from treating Sunday and Monday
    as maximally distant when they are actually adjacent.

    Args:
        df: Input DataFrame with day of week column
        day_col: Name of the day of week column

    Returns:
        DataFrame with added cyclic features
    """
    df = df.copy()

    day_mapping = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    day_number = df[day_col].map(day_mapping)

    df["day_sin"] = np.sin(day_number * 2 * math.pi / 7)
    df["day_cos"] = np.cos(day_number * 2 * math.pi / 7)

    return df


def validate_dataframe(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate dataframe for common data quality issues.

    Args:
        df: Input DataFrame to validate

    Returns:
        Dictionary containing validation results
    """
    results = {
        "row_count": len(df),
        "null_counts": df.isnull().sum().to_dict(),
        "has_nulls": df.isnull().any().any(),
        "duplicate_count": df.duplicated().sum(),
    }

    if "total_amount" in df.columns:
        results["negative_fares"] = (df["total_amount"] < 0).sum()
        results["zero_fares"] = (df["total_amount"] == 0).sum()

    if "trip_distance" in df.columns:
        results["zero_distance"] = (df["trip_distance"] == 0).sum()
        results["negative_distance"] = (df["trip_distance"] < 0).sum()

    return results


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame based on percentile bounds.

    Args:
        df: Input DataFrame
        column: Column name to check for outliers
        lower_percentile: Lower percentile threshold (0-1)
        upper_percentile: Upper percentile threshold (0-1)

    Returns:
        DataFrame with outliers removed
    """
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
