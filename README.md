# Boston Housing Price Prediction

## Overview
This project focuses on predicting Boston housing prices using various regression techniques. The analysis includes Exploratory Data Analysis (EDA), model building, hyperparameter tuning, and model evaluation.

## Motivation
I undertook this project to gain practical experience in applying machine learning techniques to a real-world dataset. It allowed me to explore different regression algorithms and understand the importance of EDA and model evaluation in achieving accurate predictions.

## Dataset
* **Name:** Boston House Price Prediction
* **Source:** Kaggle

## Project Highlights
* **Exploratory Data Analysis (EDA):**
    * Univariate analysis (histograms, density plots, box plots) to understand feature distributions.
    * Bivariate analysis (scatter plots, correlation heatmap) to explore relationships between variables.
    * Outlier detection and handling.
* **Regression Models:**
    * Linear Regression
    * Random Forest Regressor
    * Support Vector Regressor (SVR)
    * K-Nearest Neighbors Regressor (KNN)
    * Gradient Boosting Regressor
* **Hyperparameter Tuning:** GridSearchCV was used to optimize model parameters.
* **Evaluation Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared (R²), and Adjusted R-squared.
* **Feature Importance:** Feature importance was calculated and visualized for tree-based models.
* **Residual Analysis:** Residual plots and distribution analysis were performed to assess model assumptions.

## Files Description
* `boston.ipynb`: Jupyter Notebook containing the complete analysis, including data loading, EDA, model building, and evaluation.
* `boston.csv`: The Boston housing dataset in CSV format.

## Dependencies
* Python 3.13
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* xgboost

## Installation
1.  Clone the repository:
    ```bash
    git clone github.com/VRM20/Boston_house_price_prediction
    ```
2.  Navigate to the project directory:
    ```bash
    cd <project_directory>
    ```
3.  It is recommended to create a virtual environment:
    ```bash
    python -m venv venv
    ```
    * Activate the virtual environment:
        * Windows: `venv\Scripts\activate`
        * macOS/Linux: `source venv/bin/activate`
4.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```


## Usage
1.  Open the `boston.ipynb` notebook using Jupyter Notebook or JupyterLab.
2.  Run the cells in the notebook sequentially to reproduce the analysis and results.

## Results
The Gradient Boosting Regressor with hyperparameter tuning achieved the best performance on the test set:
* MAE: 1.9
* MSE: 6.2
* RMSE: 2.5
* R-squared: 0.92
* Adjusted R-squared: 0.91


## Improvements
Potential areas for future improvement include:
* Exploring other advanced regression models.
* Applying more sophisticated feature engineering techniques.
* Investigating different hyperparameter tuning methods (e.g., Bayesian Optimization).
* Collecting more data to improve model generalization.

## Author
Vanjul Rander
