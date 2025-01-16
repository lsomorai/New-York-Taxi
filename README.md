# New York Taxi Fare Prediction

## Table of Contents

1. [Introduction and Motivation](#Introduction-and-Motivation)  
   1.1 [Introduction](#Introduction)  
   1.2 [Problem Statement](#Problem-Statement)  
   1.3 [Motivation](#Motivation)  
2. [Data Collection](#Data-Collection)  
   2.1 [Dataset Overview](#Dataset-Overview)  
   2.2 [Data Import](#Data-Import)  
3. [Data Inspection and Validation](#Data-Inspection-and-Validation)  
   3.1 [Initial Inspection](#Initial-Inspection)  
   3.2 [Results](#Results)  
4. [Data Filtering](#Data-Filtering)  
5. [Data Transformation](#Data-Transformation)  
   5.1 [Convert Location IDs to Zone Names](#Convert-Location-IDs-to-Zone-Names)  
   5.2 [Time Features](#Time-Features)  
   5.3 [Normalization and Feature Engineering](#Normalization-and-Feature-Engineering)  
6. [Exploratory Data Analysis](#Exploratory-Data-Analysis)  
   6.1 [Scatter Plots](#Scatter-Plots)  
   6.2 [Correlation Matrix](#Correlation-Matrix)  
7. [Model Building and Results](#Model-Building-and-Results)  
   7.1 [Support Vector Machines (SVM)](#Support-Vector-Machines-SVM)  
   7.2 [Ridge Regression](#Ridge-Regression)  
   7.3 [Random Forest](#Random-Forest)  
   7.4 [Gradient Boosting](#Gradient-Boosting)  
8. [Conclusion](#Conclusion)  
9. [References](#References)  

---

## 1. Introduction and Motivation

### 1.1 Introduction

Big data has revolutionized industries by enabling the extraction of meaningful insights from massive datasets. In transportation, companies like Uber and Lyft leverage such insights to optimize services and enhance customer experiences. This project focuses on predicting the cost of taxi trips in New York City using the New York Taxi Trip Data, which includes details like trip time, pickup locations, and drop-off locations.

### 1.2 Problem Statement

The primary objective is to answer the question:  
**How can the cost of a taxi trip be predicted using the time, pickup location, and drop-off location?**

Accurate fare prediction is vital for ride-hailing companies to:  
- Provide customers with upfront pricing, building trust.  
- Ensure steady cash flow through advance payments.  
- Optimize traffic predictions to reduce wait times.  
- Improve resource allocation, such as vehicle and driver utilization.

### 1.3 Motivation

Enhanced fare prediction models improve operational efficiency and customer satisfaction. Additionally, understanding the factors influencing trip costs provides valuable insights into urban transportation dynamics in New York City.

---

## 2. Data Collection

### 2.1 Dataset Overview

The dataset used for this project is the New York Taxi Trip Data available on Kaggle:  
[Dataset Link](https://www.kaggle.com/datasets/microize/newyork-yellow-taxi-trip-data-2020-2019)  

Two files were selected:  
- `yellow_tripdata_2019_06.csv` (primary data file)  
- `taxi+_zone_lookup.csv` (zone lookup file)

### 2.2 Data Import

Data was imported into a Jupyter Notebook using Spark. The steps include:  
1. Initializing a Spark session.  
2. Importing the CSV files with `spark.read.csv()`.

---

## 3. Data Inspection and Validation

### 3.1 Initial Inspection

The dataset was checked for:  
- Column integrity.  
- Data types.  
- Missing values and distribution ranges.

### 3.2 Results

The dataset passed all validation checks. Columns and data types were verified, and no missing values were detected.

---

## 4. Data Filtering

To optimize resource usage and model training, irrelevant features were removed, including:  
- `VendorID`, `store_and_fwd_flag`, `fare_amount`, `extra`, `mta_tax`, `tolls_amount`, `improvement_surcharge`, `congestion_surcharge`, `payment_type`, and `tip_amount`.

---

## 5. Data Transformation

### 5.1 Convert Location IDs to Zone Names

Location IDs in the dataset were mapped to zone names using the `taxi+_zone_lookup.csv` file for better interpretability.

### 5.2 Time Features

Time-related columns were transformed to include:  
- **Hour of the day** (0-23).  
- **Day of the week** (0-6).  

### 5.3 Normalization and Feature Engineering

- Numerical features were normalized to improve model performance.  
- Time-series features were converted to cyclic representations to capture temporal patterns.

---

## 6. Exploratory Data Analysis

### 6.1 Scatter Plots

- **Passenger Count vs. Total Amount**: Minimal impact of passenger count on fare.  
- **Trip Distance vs. Total Amount**: Positive correlation, indicating trip distance as a key factor.

### 6.2 Correlation Matrix

- **Strong correlation**: Trip distance and fare amount.  
- **Weak correlation**: Passenger count and fare amount.

---

## 7. Model Building and Results

### 7.1 Support Vector Machines (SVM)

**Performance Metrics**:  
- R²: 0.810  
- RMSE: 6.473  

**Feature Importance**: Trip distance was the most significant predictor.

### 7.2 Ridge Regression

**Performance Metrics**:  
- R²: 0.767  
- RMSE: 7.455  

**Key Findings**: Accurately predicted typical fares but struggled with extreme values.

### 7.3 Random Forest

**Performance Metrics**:  
- R²: 0.715  
- RMSE: 8.512  

**Observations**: Captured general trends but showed deviations for higher fares.

### 7.4 Gradient Boosting

**Details Pending Completion**

---

## 8. Conclusion

This project demonstrates the application of machine learning techniques to predict taxi fares using real-world data. While SVM and Ridge Regression performed well, there is room for improvement in handling outliers and extreme fare values.  

**Future Work**:  
- Hyperparameter tuning.  
- Incorporating additional features, such as weather data.

---

## 9. References

- [New York Taxi Trip Data: Kaggle Dataset](https://www.kaggle.com/datasets/microize/newyork-yellow-taxi-trip-data-2020-2019)  
- [Spark Documentation: Apache Spark](https://spark.apache.org/)
