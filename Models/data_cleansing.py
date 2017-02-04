

def _drop_features_with_missing_values(design_matrix, threshold=0.2):
    """Drop features from the data frame having missing values above a given
    threshold."""
    missing_perc = 1 - (design_matrix.count() / design_matrix.shape[0])
    missing_perc = missing_perc[missing_perc > threshold]
    drop_cols = list(missing_perc.index)
    design_matrix.drop(drop_cols, axis=1, inplace=True)
    return design_matrix


def _drop_rows_with_missing_values(design_matrix):
    """Drop rows from the data frame having missing values."""
    design_matrix = design_matrix.dropna()
    return design_matrix


def clean_data(design_matrix):
    """Cleaning the raw data by removing missing values."""
    house_prices = _drop_features_with_missing_values(design_matrix)
    house_prices = _drop_rows_with_missing_values(house_prices)
    return house_prices
