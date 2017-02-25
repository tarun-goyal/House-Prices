import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor


def _impute_missing_values(design_matrix, is_test):
    """Impute values for all the features having missing values."""
    missing_cols = {'mean': ['MasVnrArea'],
                    'mode': ['MasVnrType', 'Electrical', 'GarageYrBlt'],
                    'zero': ['BsmtUnfSF', 'BsmtFinSF2', 'TotalBsmtSF',
                             'BsmtFullBath','BsmtHalfBath', 'BsmtFinSF1',
                             'GarageArea']}
    design_matrix = _imputation_using_regression(design_matrix)
    design_matrix = _imputation_using_mean(design_matrix, missing_cols['mean'])
    if is_test:
        for col in missing_cols['zero']:
            design_matrix[col].fillna(0, inplace=True)
    for col in missing_cols['mode']:
        design_matrix[col] = design_matrix[col].fillna(
            design_matrix[col].value_counts().index[0])
    return design_matrix


def _imputation_using_regression(design_matrix):

    missing_col = 'LotFrontage'
    predictors = ['LotArea', 'LotShape', 'LotConfig']
    non_na_rows = design_matrix.dropna(subset=[missing_col])
    na_rows = design_matrix[design_matrix[missing_col].isnull()]

    non_na_rows = _create_dummies_for_categorical_features(non_na_rows)
    na_rows = _create_dummies_for_categorical_features(na_rows)

    predictors_after_dummy_creation = []
    # column names of categorical var have changed after dummy creation
    for pred in predictors:
        _ = [x for x in non_na_rows.columns if pred in x]
        predictors_after_dummy_creation.extend(_)

    for pred in predictors_after_dummy_creation:
        if pred not in na_rows.columns:  #na rows didn't contain LotConfig_FR3
            na_rows[pred] = 0.

    model = XGBRegressor(n_estimators=5000, learning_rate=0.01, subsample=0.8,
                         colsample_bytree=0.8)
    model.fit(non_na_rows[predictors_after_dummy_creation],
              non_na_rows['LotFrontage'])

    na_rows['LotFrontage'] = model.predict(
        na_rows[predictors_after_dummy_creation])

    # nom_na rows and na_rows now contain extra dummy variables, so can't
    # directly use their concatenated DF.
    design_matrix_helper = pd.concat([non_na_rows, na_rows], ignore_index=True)
    design_matrix = pd.merge(
        design_matrix, design_matrix_helper[['Id', 'LotFrontage']],
        how='left', on='Id')
    design_matrix.drop('LotFrontage_x', axis=1, inplace=True)
    design_matrix.rename(columns={'LotFrontage_y': 'LotFrontage'},
                         inplace=True)
    return design_matrix


def _imputation_using_mean(design_matrix, col_list):
    """Imputation by mean"""
    for col in col_list:
        design_matrix[col] = design_matrix[col].fillna(
            design_matrix[col].mean())
    return design_matrix


def _convert_data_types(design_matrix):
    """Conversion of categorical type continuous features into objects"""
    conversion_list = ['BedroomAbvGr', 'YrSold', 'MoSold', 'BsmtHalfBath',
                       'HalfBath']
        # 'OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath',
        # 'FullBath', 'HalfBath', 'MoSold', 'YrSold', 'BedroomAbvGr',
        # 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'Fireplaces']
    for column in conversion_list:
        design_matrix[column] = design_matrix[column].apply(str)
    return design_matrix


def _create_dummies_for_categorical_features(design_matrix):
    """Create dummies for categorical features"""
    feature_types = dict(design_matrix.dtypes)
    categorical_features = [feature for feature, type in feature_types
                            .iteritems() if type == 'object']
    design_matrix = pd.get_dummies(design_matrix, prefix=categorical_features,
                                   columns=categorical_features)
    return design_matrix


def clean_data(design_matrix, is_test=False, remove_outliers=False):
    """Cleaning raw data before model processing."""
    design_matrix = _impute_missing_values(design_matrix, is_test)
    design_matrix = _convert_data_types(design_matrix)
    design_matrix.fillna('None', inplace=True)
    design_matrix = _create_dummies_for_categorical_features(design_matrix)
    if remove_outliers:
        design_matrix = outlier_treatment(design_matrix)
    return design_matrix


def outlier_treatment(design_matrix, m=2.75):
    filtered_index = abs(design_matrix['SalePrice'] - np.mean(
        design_matrix['SalePrice'])) < m * np.std(design_matrix['SalePrice'])
    design_matrix = design_matrix[filtered_index]
    return design_matrix
