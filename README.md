# Taking Aimes - Regression Analysis of Conspicuous Consumption Factors in the Ames, Iowa Housing Market

## Introduction

What is conspicuous consumption? It is the act of spending money on goods and services that are not necessary, but are used to show off wealth. This is a common practice in the United States, and is often seen in the housing market. In this project, we will be analyzing the housing market in Ames, Iowa, and determining which factors are most important in determining the price of a house. We will be using the Ames Housing dataset, which contains 2930 observations and 82 variables. The dataset can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

## Data Dictionary

| Variable | Description |
|----------|-------------|
| SalePrice | The property's sale price in dollars. This is the target variable that you're trying to predict. |
| MSSubClass | The building class |
| MSZoning | The general zoning classification |
| LotFrontage | Linear feet of street connected to property |
| LotArea | Lot size in square feet |
| Street | Type of road access |
| Alley | Type of alley access |
| LotShape | General shape of property |
| LandContour | Flatness of the property |
| Utilities | Type of utilities available |
| LotConfig | Lot configuration |
| LandSlope | Slope of property |
| Neighborhood | Physical locations within Ames city limits |
| Condition1 | Proximity to main road or railroad |
| Condition2 | Proximity to main road or railroad (if a second is present) |
| BldgType | Type of dwelling |

## Problem Statement -
An Airbnb investor approached our company with a proposal to purchase one or more houses to rent out in Ames, Iowa. They have their eye on several key neighborhoods in Ames which will form the basis for this study. Therefore, we postulate that an analysis of surrounding factors is in the client's best interest. With this in mind, we are performing a regression analysis to determine which conspicuous consumption [1] features (`CCFs`), commonly referred to as 'keeping up with the joneses', are most influential or noteworthy in determining the market value of a house (as of 2010) in these neighborhoods in Ames, Iowa.

## Preflight Checklist for Linear Regression
- [x] Is the target variable continuous?
- [x] Are the features continuous or categorical?
- [x] Are there any missing values?
- [x] Are there any outliers?
- [x] Are there any multicollinearity issues?
- [x] Is the data normally distributed?
- [x] Is the variance homoscedastic?
- [x] Are there any other issues that need to be addressed?

- We also want to check for a linear relationship between the target variable and the features. We can do this by plotting the target variable against each feature. If the relationship is linear, we should see a straight line. If the relationship is non-linear, we should see a curved line. We will do this in the EDA section.
- We have to see evenly distributed residuals (there must be multivariate normality). We can do this by plotting the residuals against the predicted values. If the residuals are evenly distributed, we should see a horizontal line. If the residuals are not evenly distributed, we should see a curved line. We will do this in the EDA section.
- As mentioned above, multicollinearity is something we must avoid if we are using linear regression models. Multiple regression assumes that the independent variables are not highly correlated with each other. We can check for multicollinearity by plotting a correlation matrix. If there are any features that are highly correlated with each other, we should remove one of them. We will do this in the EDA section.

# Defining our `success` criteria

## Success Criteria Metrics

* `R2` score (Coefficient of Determination) - This is the percentage of the variance in the target variable that is explained by the model.
* `RMSE` score (Root Mean Squared Error) - This is the average error of the model. The lower the RMSE, the better the model.

## Success Criteria Values
1. A model that can predict the sale price of a house in Ames, Iowa, with an `R2 score` of `0.7` or greater. *This would indicate that the model is able to explain 70% of the variance in the sale price of a house in Ames, Iowa.*
2. A model that can predict the sale price of a house in Ames, Iowa, with an `RMSE` of `30,000` or less. *This would indicate that the average error of the model is less than \\$30,000.00.*

The second option is clearly less desirable than the first option. However, considering both may be necessary to determine the best model.


## Data Cleaning and EDA (Exploratory Data Analysis)

- We will be using the `pandas` library to read in the data.
- We will be using the `pandas_profiling` library to generate a report of the data.

## Using Pandas Profiling to Generate a Report of the Data

When we use the `pandas_profiling` library to generate a report of the data, we get a lot of information about the data. We will be using the `ProfileReport` class from the `pandas_profiling` library to generate an html report which we can use to draw conclusions about the quality of this source. We will be using the `to_file()` method to save the report to a file.



### Missing Values

| Variable | Missing Values | % of Total Values |
|----------|----------------|-------------------|
Lot Frontage | 490 | 16.72%
Alley | 2732 | 93.22%
Mas Vnr Type | 22 | 0.75%
Mas Vnr Area | 22 | 0.75%
Bsmt Qual | 80 | 2.73%
Bsmt Cond | 80 | 2.73%
Bsmt Exposure | 82 | 2.80%
BsmtFin Type 1 | 80 | 2.73%
BsmtFin SF 1 | 1 | 0.03%
BsmtFin Type 2 | 81 | 2.74%
BsmtFin SF 2 | 1 | 0.03%
Bsmt Unf SF | 1 | 0.03%
Total Bsmt SF | 1 | 0.03%
Bsmt Full Bath | 2 | 0.07%
Bsmt Half Bath | 2 | 0.07%
Fireplace Qu | 1420 | 48.59%
Garage Type | 157 | 5.38%
Garage Yr Blt | 159 | 5.45%
Garage Finish | 159 | 5.45%
Garage Cars | 1 | 0.03%
Garage Area | 1 | 0.03%
Garage Qual | 159 | 5.45%
Garage Cond | 159 | 5.45%
Pool QC | 2917 | 99.59%
Fence | 2358 | 80.44%
Misc Feature | 2824 | 96.50%



### Action Plan for Missing Values

* We will be dropping columns with more than 50% missing values. These columns are `Alley`, `Pool QC`, `Fence`, and `Misc Feature`.
If a column has fewer than 50% missing values, we will be imputing the missing values, how we do this will be determined by the type of data in the column.

* We will be using the `SimpleImputer` class from `sklearn.impute` to impute the missing values.
* We will be using the `ColumnTransformer` class from `sklearn.compose` to apply the imputer to the appropriate columns.


### Positively Correlated Features with Sale Price

| Feature | Correlation with Sale Price |
|---------|-----------------------------|
Overall Qual | 0.80
Gr Liv Area | 0.70
Garage Cars | 0.64
Garage Area | 0.64
Total Bsmt SF | 0.63
1st Flr SF | 0.62
Year Built | 0.55
Full Bath | 0.54
Year Remod/Add | 0.53
Garage Yr Blt | 0.52
Mas Vnr Area | 0.51

### Negatively Correlated Features with Sale Price

| Feature | Correlation with Sale Price |
|---------|-----------------------------|
PID | -0.24
Enclosed Porch | -0.13
Kitchen AbvGr | -0.12
Overall Cond | -0.10
MS SubClass | -0.09
Low Qual Fin SF | -0.03
Bsmt Half Bath | -0.03

### Action Plan for Correlated Features

The best way to deal with correlated features is to remove one of them. We will be using the `VarianceThreshold` class from `sklearn.feature_selection` to remove features that have a variance below a certain threshold. We will be using the `ColumnTransformer` class from `sklearn.compose` to apply the `VarianceThreshold` to the appropriate columns.

### Action Plan for Outliers

We will be using the `IsolationForest` class from `sklearn.ensemble` to detect outliers. We will be using the `ColumnTransformer` class from `sklearn.compose` to apply the `IsolationForest` to the appropriate columns.

### Action Plan for Categorical Features

We will be using the `OneHotEncoder` class from `sklearn.preprocessing` to encode the categorical features. We will be using the `ColumnTransformer` class from `sklearn.compose` to apply the `OneHotEncoder` to the appropriate columns.

### Action Plan for Numerical Features

We will be using the `StandardScaler` class from `sklearn.preprocessing` to scale the numerical features. We will be using the `ColumnTransformer` class from `sklearn.compose` to apply the `StandardScaler` to the appropriate columns.

### Action Plan for Features with a Skewed Distribution

We will be using the `PowerTransformer` class from `sklearn.preprocessing` to transform the features with a skewed distribution. We will be using the `ColumnTransformer` class from `sklearn.compose` to apply the `PowerTransformer` to the appropriate columns.

### Action Plan for Features with a Log-Normal Distribution

We will be using the `PowerTransformer` class from `sklearn.preprocessing` to transform the features with a log-normal distribution. We will be using the `ColumnTransformer` class from `sklearn.compose` to apply the `PowerTransformer` to the appropriate columns.


## Feature Engineering

### Action Plan for Feature Engineering

We will be using the `PolynomialFeatures` class from `sklearn.preprocessing` to create polynomial features. We will be using the `ColumnTransformer` class from `sklearn.compose` to apply the `PolynomialFeatures` to the appropriate columns.

## Model Selection

### Action Plan for Model Selection

To select the right model for this problem, we will be using the `cross_val_score` function from `sklearn.model_selection` to evaluate the performance of different models. We will be using the `mean_squared_error` function from `sklearn.metrics` to evaluate the performance of the models. We will be using the `GridSearchCV` class from `sklearn.model_selection` to find the best hyperparameters for the models.

## Model Evaluation

### Action Plan for Model Evaluation

We will be using the `mean_squared_error` function from `sklearn.metrics` to evaluate the performance of the models. We will be using the `mean_absolute_error` function from `sklearn.metrics` to evaluate the performance of the models. We will be using the `r2_score` function from `sklearn.metrics` to evaluate the performance of the models.

**Reason:** The `mean_squared_error`, `mean_absolute_error`, and `r2_score` metrics are good to use when the target variable is continuous. The `mean_squared_error` metric is (by some accounts) the most commonly used metric for regression problems. The `mean_absolute_error` metric is less sensitive to outliers than the `mean_squared_error` metric. The `r2_score` metric is actually the more likely candidate for the most commonly used metric for regression problems.

## Model Deployment

### Action Plan for Model Deployment

We will be using the `pickle` module to save the model to a file. We will be using the `pickle` module to load the model from a file.

**Reason:** The `pickle` module is great for serializing and deserializing Python objects. It makes the process easy, and fast.

# Code Implementation of these Action Plans

| Feature | Number of Missing Values | Percentage of Missing Values |
|---------|--------------------------|------------------------------|
Fireplace Qu | 1420 | 48.59%
Alley | 2732 | 93.22%
Pool QC | 2917 | 99.59%
Fence | 2358 | 80.44%
Misc Feature | 2824 | 96.50%

The Fireplace Quality is too close to the threshold of 50% missing values. I am going to drop it.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pickle

def load_and_process_data(path):
    df = (
        pd.read_csv(path) # Load the data
    )
    # Drop columns with more than 50% missing values using pandas and list comprehension to create a list of columns to drop
    df = df.drop([col for col in df.columns if df[col].isnull().sum() > len(df) / 2], axis=1) # Drop columns with more than 50% missing values
    # dropping the columns below because they are not useful for the model and have too many missing values
    # Fireplace Qu | 1420 | 48.59%
    # Alley | 2732 | 93.22%
    # Pool QC | 2917 | 99.59%
    # Fence | 2358 | 80.44%
    # Misc Feature | 2824 | 96.50%
    df = df.drop(['Fireplace Qu', 'Alley', 'Pool QC', 'Fence', 'Misc Feature'], axis=1)
    # Drop the Id column
    df = df.drop(['Id'], axis=1)
    #* save a placeholder file to keep progress in case of a crash
    df.to_csv('data/current_data/placeholder.csv')
    # next, we will be dealing with the missing values
