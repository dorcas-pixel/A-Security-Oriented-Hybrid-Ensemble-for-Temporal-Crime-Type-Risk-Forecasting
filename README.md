# A-Security-Oriented-Hybrid-Ensemble-for-Temporal-Crime-Type-Risk-Forecasting
# Time Series Ensemble Classification

This repository contains a Time Series Classification pipeline that combines a Gradient Boosting model and a Decision Tree using probability averaging. It is designed to evaluate the stability and performance of ensemble models on sequential data using TimeSeriesSplit cross-validation.  

## Features

- Ensemble of HistGradientBoostingClassifier and DecisionTreeClassifier
- Supports macro-F1 and False Alarm Rate (FAR) metrics
- Handles time series data with sequential cross-validation
- Optimized for speed with:
  - Histogram-based GBM
  - Early stopping
  - Parallel processing using `joblib`
- Easy to extend with other models

## Requirements

- Python 3.8+
- Libraries:
  ```bash
  pip install numpy pandas scikit-learn joblib

