from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# One hot encode the categorical features so that they can be used in the model.
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import (
    LassoCV,
    RidgeCV,
    ElasticNetCV,
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
)
from sklearn import metrics


def nan_inf_janitor(df):
    """
    Cleans a dataframe of NaNs and Infs.

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # This function takes a dataframe and removes the NaNs and Infs from it using Imputation.
    # if the column contains mostly integer/float values...
    # then replace the NaNs with the median of the column
    # if the column contains mostly string values...
    # then replace the NaNs with the mode of the column
    # if the column contains mostly boolean values...
    # then replace the NaNs with the mode of the column
    # if the column contains mostly datetime values...
    # then replace the NaNs with the mode of the column
    # if the column contains mostly categorical values...
    # then replace the NaNs with the mode of the column
    # if the column contains mostly ordinal values...
    # then replace the NaNs with the mode of the column
    # if the column contains mostly binary values...
    # then replace the NaNs with the mode of the column

    print("Janitor is cleaning the dataframe...")
    print("Cleaning the NaNs and Infs...")
    print(f"{df.columns}")
    print(f"-" * 50)

    # for each column in df...
    for col in tqdm(df.columns):
        try:
            # Determine the dtypes
            col_dtype_majority = df[col].dtypes
            # If the column is a float or an integer...
            if col_dtype_majority == "float64" or col_dtype_majority == "int64":
                # print(f"col: {col} is a float or an integer")
                # Replace the NaNs with the median of the column
                try:
                    df[col].fillna(df[col].median(), inplace=True)
                except Exception as e:
                    print(f"Error: {e}")
                    df[col].replace([np.inf, -np.inf], df[col].median(), inplace=True)
            elif col_dtype_majority == "object":
                # print(f"col: {col} is an object")
                # Replace the NaNs with the mode of the column
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif col_dtype_majority == "bool":
                # print(f"col: {col} is a bool")
                # Replace the NaNs with the mode of the column
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif col_dtype_majority == "datetime64[ns]":
                # print(f"col: {col} is a datetime64[ns]")
                # Replace the NaNs with the mode of the column
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif col_dtype_majority == "category":
                # print(f"col: {col} is a category")
                # Replace the NaNs with the mode of the column
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif col_dtype_majority == "int8":
                # print(f"col: {col} is a int8")
                # Replace the NaNs with the mode of the column
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif col_dtype_majority == "int16":
                # print(f"col: {col} is a int16")
                # Replace the NaNs with the mode of the column
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].replace([np.inf, -np.inf], df[col].median(), inplace=True)
                print("other")
            # Replace the Infs with the median of the column
            df[col].replace([np.inf, -np.inf], df[col].median(), inplace=True)
        except Exception as e:
            # print(f"col: {col} in dataframe")
            # print(f"Error: {e}")
            pass
    return df


import tqdm
from tqdm import tqdm
import datetime


def extract_feature_range(col):
    # Given a column, return the interquartile range
    # Determine the 1st and 3rd quartiles
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    # Determine the IQR
    iqr = q3 - q1
    return iqr


def detect_squarefootage_feature(col, feature_values):
    # given a series of values in a column, return True if the values are likely square footage values.
    # Determine the 1st and 3rd quartiles
    likely_sqft_range = range(
        100, 5000
    )  # Likely to be sqft if in this range of values.
    if (
        feature_values.min() in likely_sqft_range
        and feature_values.max() in likely_sqft_range
        or ("area" in col.lower() or "sq" in col.lower())
    ):
        return True
    else:
        return False


def detect_judgementfeature(col, feature_values):
    # given a series of values in a column, return True if the values are likely judgement features.
    # Determine the 1st and 3rd quartiles
    likely_judgement_range = range(
        1, 11
    )  # Likely to be judgement if in this range of values. The range will include 1 and 10.
    # todo --- the range above could cause issues with the left number.
    if (
        feature_values.min() in likely_judgement_range
        and feature_values.max() in likely_judgement_range
    ) or "qual" in col.lower():
        return True
    else:
        return False


def engineer_features(df, features=[]):
    # Given a dataframe, return a dataframe with new features.
    # Create a new feature called 'Total SF' that is the sum of the '1st Flr SF' and '2nd Flr SF' columns
    df["total_sqft"] = df["1st_flr_sf"] + df["2nd_flr_sf"]
    # Create a new feature called 'Total Bath' that is the sum of the 'Full Bath' and 'Half Bath' columns
    df["total_bath"] = df["full_bath"] + (df["half_bath"] * 0.5)
    # Create a new feature called 'Total Porch SF' that is the sum of the 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', and 'Screen Porch' columns
    df["total_porch_sf"] = (
        df["open_porch_sf"]
        + df["enclosed_porch"]
        + df["3ssn_porch"]
        + df["screen_porch"]
    )
    # Create a new feature called 'Total Home Quality' that is the sum of the 'Overall Qual' and 'Overall Cond' columns
    df["total_home_quality"] = df["overall_qual"] + df["overall_cond"]
    # New Feature: Ratio of square footage to lot size
    df["sqft_to_lot_ratio"] = df["total_sqft"] / df["lot_area"]
    # New Feature: Ratio of square footage to quality
    df["sqft_to_quality_ratio"] = df["total_sqft"] / df["total_home_quality"]
    # New Feature: Ratio of total bedrooms to total bathrooms
    df["bed_to_bath_ratio"] = df["bedroom_abvgr"] / df["total_bath"]
    # New Feature: Basement square footage to total square footage (how much of the house is a basement?)
    df["bsmt_to_totalsqft_ratio"] = df["total_bsmt_sf"] / df["total_sqft"]
    # New Feature: Ratio of total porch square footage to total square footage.
    df["porch_to_totalsqft_ratio"] = df["total_porch_sf"] / df["total_sqft"]

    # New Feature: the difference between general living area and total square footage
    df["diff_grnd_livarea_totalsqft"] = df["gr_liv_area"] - df["total_sqft"]

    # THE JONES EFFECT VARIABLES -
    # Size of the house in relation to houses in the same neighborhood
    df["size_relative_to_the_neighbors"] = df.groupby("neighborhood")[
        "neighborhood"
    ].transform("count")
    # New Feature: Lot frontage compared to other houses in the same neighborhood
    df["lot_frontage_relative_to_the_neighbors"] = df.groupby("neighborhood")[
        "lot_frontage"
    ].transform("mean")
    # New Feature: Lot area compared to other houses in the same neighborhood
    df["lot_area_relative_to_the_neighbors"] = df.groupby("neighborhood")[
        "lot_area"
    ].transform("mean")
    # New Feature: How does the house rank in the neighborhood relative to when it was remodeled? Relative to the median year the houses in their neighborhood were remodeled?
    df["remodeled_relative_to_the_neighbors"] = df.groupby("neighborhood")[
        "year_remod/add"
    ].transform(
        "median"
    )  # todo - check this
    # New Feature: How does the house rank in the neighborhood relative to when it was built? Relative to the median year the houses in their neighborhood were built?
    df["built_relative_to_the_neighbors"] = df.groupby("neighborhood")[
        "year_built"
    ].transform(
        "median"
    )  # todo - check this
    # Feature: how many houses in the same neighborhood have a basement?
    df["houses_in_neighborhood_with_basement"] = df.groupby("neighborhood")[
        "bsmt_qual"
    ].transform("count")
    # Feature: how many houses in the same neighborhood have a garage?
    df["houses_in_neighborhood_with_garage"] = df.groupby("neighborhood")[
        "garage_type"
    ].transform("count")
    # Feature: how many houses in the same neighborhood have a fireplace?
    df["houses_in_neighborhood_with_fireplace"] = df.groupby("neighborhood")[
        "fireplace_qu"
    ].transform("count")
    # Feature: how many houses in the same neighborhood have a pool?
    df["houses_in_neighborhood_with_pool"] = df.groupby("neighborhood")[
        "pool_area"
    ].transform("count")

    # New Feature: Ratio of total square footage to total number of rooms
    df["sqft_to_rooms_ratio"] = df["total_sqft"] / df["totrms_abvgrd"]
    # Lot Frontage to Lot Area
    df["lot_frontage_to_lot_area_ratio"] = df["lot_frontage"] / df["lot_area"]

    # The features that were added by this function are: 'total_sqft', 'total_bath', 'total_porch_sf', 'total_home_quality', 'sqft_to_lot_ratio', 'sqft_to_quality_ratio', 'bed_to_bath_ratio', 'bsmt_to_totalsqft_ratio', 'porch_to_totalsqft_ratio', 'diff_grnd_livarea_totalsqft', 'size_relative_to_the_neighbors', 'lot_frontage_relative_to_the_neighbors', 'lot_area_relative_to_the_neighbors', 'remodeled_relative_to_the_neighbors', 'built_relative_to_the_neighbors', 'houses_in_neighborhood_with_basement', 'houses_in_neighborhood_with_garage', 'houses_in_neighborhood_with_fireplace', 'houses_in_neighborhood_with_pool', 'sqft_to_rooms_ratio', 'lot_frontage_to_lot_area_ratio'

    features = features + [
        "total_sqft",
        "total_bath",
        "total_porch_sf",
        "total_home_quality",
        "sqft_to_lot_ratio",
        "sqft_to_quality_ratio",
        "bed_to_bath_ratio",
        "bsmt_to_totalsqft_ratio",
        "porch_to_totalsqft_ratio",
        "diff_grnd_livarea_totalsqft",
        "size_relative_to_the_neighbors",
        "lot_frontage_relative_to_the_neighbors",
        "lot_area_relative_to_the_neighbors",
        "remodeled_relative_to_the_neighbors",
        "built_relative_to_the_neighbors",
        "houses_in_neighborhood_with_basement",
        "houses_in_neighborhood_with_garage",
        "houses_in_neighborhood_with_fireplace",
        "houses_in_neighborhood_with_pool",
        "sqft_to_rooms_ratio",
        "lot_frontage_to_lot_area_ratio",
    ]

    # add the features created by the function 'engineer_features' to the list of features
    # features = features + features_created
    # features is a list of the features that were added by the function, 'engineer_features'

    # for each feature in the list of features, fill the null values with the mean of the column
    for feature in features:
        df[feature].fillna(df[feature].mean(), inplace=True)

    train.to_csv("datasets/train_addedfeatures.csv")
    print("done")
    print(f"All my features: {features}")
    # saving the features list to a file
    with open("datasets/features.txt", "w") as f:
        for item in features:
            f.write("%s, " % item)
    return df, features
