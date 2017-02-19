import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression


def _impute_missing_values(design_matrix):
    """Impute values for all the features having missing values."""
    missing_cols = {'cont': ['LotFrontage', 'MasVnrArea'],
                    'cat': ['MasVnrType', 'Electrical', 'GarageYrBlt']}
    design_matrix = _imputation_using_regression(design_matrix)
    design_matrix = _imputation_using_mean(design_matrix, missing_cols['cont'],
                                              strategy='mean')
    for col in missing_cols['cat']:
        design_matrix = design_matrix.fillna(design_matrix[col].value_counts()
                                             .index[0])
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

    model = LinearRegression(fit_intercept=True)
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


def _imputation_using_mean(design_matrix, col_list, strategy):
    """Strategy behind imputation"""
    for col in col_list:
        impute = Imputer(strategy=strategy, axis=0)
        imputed = impute.fit_transform(design_matrix[col])
        design_matrix[col] = pd.Series(imputed.tolist()[0])
    return design_matrix


def _normalize_continuous_features(design_matrix):
    """Normalize the continuous features"""
    feature_types = dict(design_matrix.dtypes)
    continuous_features = [feature for feature, type in feature_types
                           .iteritems() if type in ('int64', 'float64')]
    for feature in continuous_features:
        design_matrix[feature] = (design_matrix[feature] - np.mean(
            design_matrix[feature])) / np.std(design_matrix[feature])
    return design_matrix


def _create_dummies_for_categorical_features(design_matrix):
    """Create dummies for categorical features"""
    feature_types = dict(design_matrix.dtypes)
    categorical_features = [feature for feature, type in feature_types
                            .iteritems() if type == 'object']
    design_matrix = pd.get_dummies(design_matrix, prefix=categorical_features,
                                   columns=categorical_features)
    return design_matrix


def clean_data(design_matrix):
    """Cleaning raw data before model processing."""
    design_matrix = _impute_missing_values(design_matrix)
    # design_matrix = _normalize_continuous_features(design_matrix)
    design_matrix = _create_dummies_for_categorical_features(design_matrix)
    return design_matrix


# Features with missing values (Training data): 'NA' not specified
# 1. LotFrontage: 259/1460 (Important)
# 2. MasVnrType: 8/1460 (Important)
# 3. MasVnrArea: 8/1460 (Important)
# 4. Electrical: 1/1460 (Important)
# 5. GarageYrBlt: 81/1460 (Not important)
