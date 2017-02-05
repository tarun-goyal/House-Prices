import pandas as pd


def _drop_features_with_missing_values(design_matrix, threshold=0.2):
    """Drop features from the data frame having missing values above a given
    threshold."""
    drop_cols = ['GarageYrBlt', 'LotFrontage']
    design_matrix.drop(drop_cols, axis=1, inplace=True)
    return design_matrix


def clean_data(design_matrix):
    """Cleaning the raw data by removing missing values."""
    house_prices = _drop_features_with_missing_values(design_matrix)
    return house_prices


# Features with missing values (Training data): 'NA' not specified
# 1. LotFrontage: 259/1460 (Important)
# 2. MasVnrType: 8/1460 (Important)
# 3. MasVnrArea: 8/1460 (Important)
# 4. Electrical: 1/1460 (Important)
# 5. GarageYrBlt: 81/1460 (Not important)
