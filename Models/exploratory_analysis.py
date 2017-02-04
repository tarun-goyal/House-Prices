

def find_correlated_features(design_matrix):
    """Find features that are highly correlated with house prices."""
    corr_matrix = design_matrix.corr()
    corr_matrix = corr_matrix['SalePrice']
    corr_matrix = corr_matrix[corr_matrix >= 0.6]
    return list(corr_matrix.index)
