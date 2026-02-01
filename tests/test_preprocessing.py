"""Tests for preprocessing functions."""

import math

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    clean_dataframe,
    create_cyclic_features,
    extract_time_features,
    map_location_ids,
    remove_outliers,
    validate_dataframe,
)


@pytest.fixture
def sample_raw_df():
    """Create a sample raw dataframe for testing."""
    return pd.DataFrame({
        "VendorID": [1, 2, 1],
        "tpep_pickup_datetime": [
            "2019-06-01 00:02:40",
            "2019-06-01 12:30:00",
            "2019-06-02 18:45:00",
        ],
        "tpep_dropoff_datetime": [
            "2019-06-01 00:18:37",
            "2019-06-01 12:45:00",
            "2019-06-02 19:00:00",
        ],
        "passenger_count": [1, 2, 3],
        "trip_distance": [2.0, 5.5, 3.2],
        "PULocationID": [162, 48, 148],
        "DOLocationID": [68, 239, 87],
        "payment_type": [1, 1, 2],
        "fare_amount": [11.5, 20.0, 15.0],
        "extra": [3.0, 0.5, 2.5],
        "mta_tax": [0.5, 0.5, 0.5],
        "tip_amount": [2.0, 4.0, 0.0],
        "tolls_amount": [0.0, 0.0, 5.0],
        "total_amount": [17.3, 25.5, 23.0],
        "store_and_fwd_flag": ["N", "N", "Y"],
        "improvement_surcharge": [0.3, 0.3, 0.3],
        "congestion_surcharge": [2.5, 2.5, 2.5],
    })


@pytest.fixture
def zone_lookup_df():
    """Create a sample zone lookup dataframe."""
    return pd.DataFrame({
        "LocationID": [48, 68, 87, 148, 162, 239],
        "Zone": [
            "Clinton East",
            "East Chelsea",
            "Financial District North",
            "Lower East Side",
            "Midtown East",
            "Upper West Side South",
        ],
    })


class TestCleanDataframe:
    """Tests for clean_dataframe function."""

    def test_drops_default_columns(self, sample_raw_df):
        """Test that default columns are dropped."""
        result = clean_dataframe(sample_raw_df)

        assert "VendorID" not in result.columns
        assert "store_and_fwd_flag" not in result.columns
        assert "fare_amount" not in result.columns
        assert "payment_type" not in result.columns

    def test_keeps_important_columns(self, sample_raw_df):
        """Test that important columns are retained."""
        result = clean_dataframe(sample_raw_df)

        assert "trip_distance" in result.columns
        assert "total_amount" in result.columns
        assert "passenger_count" in result.columns

    def test_custom_columns_to_drop(self, sample_raw_df):
        """Test dropping custom columns."""
        result = clean_dataframe(sample_raw_df, columns_to_drop=["VendorID"])

        assert "VendorID" not in result.columns
        assert "payment_type" in result.columns

    def test_handles_missing_columns(self, sample_raw_df):
        """Test that missing columns don't cause errors."""
        result = clean_dataframe(
            sample_raw_df,
            columns_to_drop=["VendorID", "nonexistent_column"],
        )

        assert "VendorID" not in result.columns
        assert len(result) == 3


class TestMapLocationIds:
    """Tests for map_location_ids function."""

    def test_maps_pickup_location(self, sample_raw_df, zone_lookup_df):
        """Test that pickup locations are mapped correctly."""
        cleaned = clean_dataframe(sample_raw_df)
        result = map_location_ids(cleaned, zone_lookup_df)

        assert "pickup_location" in result.columns
        assert result["pickup_location"].iloc[0] == "Midtown East"

    def test_maps_dropoff_location(self, sample_raw_df, zone_lookup_df):
        """Test that dropoff locations are mapped correctly."""
        cleaned = clean_dataframe(sample_raw_df)
        result = map_location_ids(cleaned, zone_lookup_df)

        assert "dropoff_location" in result.columns
        assert result["dropoff_location"].iloc[0] == "East Chelsea"

    def test_removes_original_id_columns(self, sample_raw_df, zone_lookup_df):
        """Test that original ID columns are removed."""
        cleaned = clean_dataframe(sample_raw_df)
        result = map_location_ids(cleaned, zone_lookup_df)

        assert "PULocationID" not in result.columns
        assert "DOLocationID" not in result.columns

    def test_handles_unknown_locations(self, zone_lookup_df):
        """Test that unknown location IDs are handled."""
        df = pd.DataFrame({
            "PULocationID": [999],
            "DOLocationID": [888],
            "trip_distance": [1.0],
        })
        result = map_location_ids(df, zone_lookup_df)

        assert result["pickup_location"].iloc[0] == "Unknown"
        assert result["dropoff_location"].iloc[0] == "Unknown"


class TestExtractTimeFeatures:
    """Tests for extract_time_features function."""

    def test_extracts_pickup_hour(self, sample_raw_df):
        """Test that pickup hour is extracted correctly."""
        result = extract_time_features(sample_raw_df)

        assert "pickup_hour" in result.columns
        assert result["pickup_hour"].iloc[0] == 0
        assert result["pickup_hour"].iloc[1] == 12

    def test_extracts_day_of_week(self, sample_raw_df):
        """Test that day of week is extracted correctly."""
        result = extract_time_features(sample_raw_df)

        assert "pickup_day_of_week" in result.columns
        assert result["pickup_day_of_week"].iloc[0] == "Saturday"

    def test_removes_datetime_columns(self, sample_raw_df):
        """Test that original datetime columns are removed."""
        result = extract_time_features(sample_raw_df)

        assert "tpep_pickup_datetime" not in result.columns
        assert "tpep_dropoff_datetime" not in result.columns


class TestCreateCyclicFeatures:
    """Tests for create_cyclic_features function."""

    def test_creates_sin_feature(self):
        """Test that sin feature is created."""
        df = pd.DataFrame({"pickup_day_of_week": ["Monday", "Thursday"]})
        result = create_cyclic_features(df)

        assert "day_sin" in result.columns
        assert len(result["day_sin"]) == 2

    def test_creates_cos_feature(self):
        """Test that cos feature is created."""
        df = pd.DataFrame({"pickup_day_of_week": ["Monday", "Thursday"]})
        result = create_cyclic_features(df)

        assert "day_cos" in result.columns
        assert len(result["day_cos"]) == 2

    def test_cyclic_values_are_bounded(self):
        """Test that cyclic values are between -1 and 1."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df = pd.DataFrame({"pickup_day_of_week": days})
        result = create_cyclic_features(df)

        assert result["day_sin"].between(-1, 1).all()
        assert result["day_cos"].between(-1, 1).all()

    def test_monday_values(self):
        """Test specific values for Monday (day 0)."""
        df = pd.DataFrame({"pickup_day_of_week": ["Monday"]})
        result = create_cyclic_features(df)

        expected_sin = math.sin(0 * 2 * math.pi / 7)
        expected_cos = math.cos(0 * 2 * math.pi / 7)

        assert abs(result["day_sin"].iloc[0] - expected_sin) < 1e-10
        assert abs(result["day_cos"].iloc[0] - expected_cos) < 1e-10


class TestValidateDataframe:
    """Tests for validate_dataframe function."""

    def test_returns_row_count(self, sample_raw_df):
        """Test that row count is returned."""
        result = validate_dataframe(sample_raw_df)

        assert result["row_count"] == 3

    def test_detects_no_nulls(self, sample_raw_df):
        """Test that absence of nulls is detected."""
        result = validate_dataframe(sample_raw_df)

        assert result["has_nulls"] == False  # noqa: E712

    def test_detects_nulls(self):
        """Test that nulls are detected."""
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = validate_dataframe(df)

        assert result["has_nulls"] == True  # noqa: E712
        assert result["null_counts"]["a"] == 1

    def test_detects_negative_fares(self):
        """Test that negative fares are detected."""
        df = pd.DataFrame({"total_amount": [10, -5, 20]})
        result = validate_dataframe(df)

        assert result["negative_fares"] == 1


class TestRemoveOutliers:
    """Tests for remove_outliers function."""

    def test_removes_extreme_values(self):
        """Test that extreme values are removed."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})
        result = remove_outliers(df, "value", lower_percentile=0.1, upper_percentile=0.9)

        assert 100 not in result["value"].values

    def test_preserves_normal_values(self):
        """Test that normal values are preserved."""
        df = pd.DataFrame({"value": list(range(1, 101))})
        result = remove_outliers(df, "value", lower_percentile=0.1, upper_percentile=0.9)

        assert len(result) < 100
        assert len(result) > 50
