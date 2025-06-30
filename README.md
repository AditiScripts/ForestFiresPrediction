# ForestFiresPrediction

This repository contains a Jupyter Notebook-based solution for Forest Fires Area Prediction. Using the Forest_Fires_Area.csv dataset (originally from UCI Machine Learning Repository), the purpose for this project was to build and evaluate supervised regression models that predicted the burned area of forest fires given meteorological and temporal features.

## Solution Summary
The notebook follows a structured pipeline:

- Data Loading & Exploration
    - Read the dataset with pandas, inspect summary statistics, and identify missing values.

- Data Cleaning
    - Drop rows with missing values.
    - Handle categorical variables (month, day) via one-hot encoding to convert them into numeric features.

- Outlier Detection & Treatment
    - Visualize feature distributions with matplotlib histograms.
    - Identify and manage extreme values to balance noise reduction and data preservation.

- Feature Scaling
    - Apply MinMaxScaler to normalize all input features to the [0,1] range, improving model convergence.

- Model 1: Linear Regression (Baseline)
    - Split data into training and test sets with train_test_split.
    - Train a LinearRegression model, print coefficients to interpret feature contributions.
    - Evaluate using Mean Squared Error (MSE) and R2.
    - Feature Selection: drop the least impactful feature, retrain, and compare performance.

- Model 2: Neural Network
    - Build an MLPRegressor with two hidden layers (10, 20 neurons) and train for up to 8000 iterations.
    - Compare MSE against the linear model to assess non-linear learning advantages.

## Technologies & Libraries
- Python 3.x
- Jupyter Notebook (.ipynb)
- pandas for data manipulation
- numpy for numerical operations
- matplotlib for visualizations
- scikit-learn for preprocessing and modeling:
- MinMaxScaler, train_test_split, LinearRegression, MLPRegressor

# Why These Models?

Linear Regression provides an interpretable baseline: coefficient values indicate each feature’s linear influence on burned area.

Neural Network (MLP) captures non-linear relationships between weather conditions, temporal factors, and fire size, often reducing error compared to linear methods.

# Usage
- Clone this repository.
- Install dependencies:
    "pip install pandas numpy matplotlib scikit-learn jupyter"
- Launch the Notebook:
    "jupyter notebook notebooks/forest_fires_prediction.ipynb"
- Run all cells to reproduce the analysis, training, and evaluation results.

## Results & Evaluation
- Linear Regression MSE: ~0.0561 on test set.
- MLPRegressor MSE: ~0.0548 on test set, showing a modest improvement.

Detailed plots, coefficient tables, and loss curves are available in the notebook.

Author: Aditi Bhat
