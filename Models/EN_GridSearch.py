from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
import pandas as pd
import numpy as np
import data_cleansing as dc


# Reading data
house_prices = pd.read_csv("../Data/train_imputed.csv")
test = pd.read_csv("../Data/test_imputed.csv")


def _create_dummies_for_categorical_features(design_matrix):
    feature_types = dict(design_matrix.dtypes)
    categorical_features = [feature for feature, type in feature_types
                            .iteritems() if type == 'object']
    design_matrix = pd.get_dummies(design_matrix, prefix=categorical_features,
                                   columns=categorical_features)
    return design_matrix


def _remove_any_more_null_rows(design_matrix):
    null_rows = pd.isnull(design_matrix).any(1).nonzero()[0]
    design_matrix.drop(design_matrix.index[null_rows], inplace=True)
    return design_matrix


class ElasticNetModel(object):

    def __init__(self):
        self.design_matrix = dc.clean_data(house_prices)
        self.design_matrix = _create_dummies_for_categorical_features(
            self.design_matrix)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in
                           ['Id', 'SalePrice']]
        _remove_any_more_null_rows(self.design_matrix)

    @staticmethod
    def _define_regressor_and_parameter_candidates():
        regressor = ElasticNet(
            fit_intercept=True, normalize=True, warm_start=True,
            max_iter=100000, tol=1e-7, selection='random')
        parameters = {
            'alpha': np.linspace(start=1e-3, stop=1e-1, num=10),
            'l1_ratio': [.95, .96, .97, .98, .99]}
        return regressor, parameters

    def grid_search_for_best_estimator(self):
        """Comprehensive search over provided parameters to find the best
        estimator"""
        regressor, parameters = self\
            ._define_regressor_and_parameter_candidates()
        model = GridSearchCV(regressor, parameters, cv=5, verbose=2, n_jobs=8)
        model.fit(self.design_matrix[self.predictors],
                  self.design_matrix['SalePrice'])
        cv_results = model.cv_results_
        results = DataFrame.from_dict(cv_results, orient='columns')
        results.to_csv('../Model_results/EN_GridSearch2_results.csv',
                       index=False)

ElasticNetModel().grid_search_for_best_estimator()
