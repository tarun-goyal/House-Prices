import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


def _impute_missing_values(design_matrix):
    """Impute values for all the features having missing values."""
    missing_cols = {'cont': ['LotFrontage', 'MasVnrArea'],
                    'cat': ['MasVnrType', 'Electrical', 'GarageYrBlt']}
    design_matrix = _imputation_strategy(design_matrix, missing_cols['cont'],
                                         strategy='mean')
    for col in missing_cols['cat']:
        design_matrix = design_matrix.fillna(design_matrix[col].value_counts()
                                             .index[0])
    return design_matrix


def _imputation_strategy(design_matrix, col_list, strategy):
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
    """Cleaning the raw data by removing missing values."""
    design_matrix = _impute_missing_values(design_matrix)
    design_matrix = _normalize_continuous_features(design_matrix)
    design_matrix = _create_dummies_for_categorical_features(design_matrix)
    return design_matrix


# Features with missing values (Training data): 'NA' not specified
# 1. LotFrontage: 259/1460 (Important)
# 2. MasVnrType: 8/1460 (Important)
# 3. MasVnrArea: 8/1460 (Important)
# 4. Electrical: 1/1460 (Important)
# 5. GarageYrBlt: 81/1460 (Not important)
