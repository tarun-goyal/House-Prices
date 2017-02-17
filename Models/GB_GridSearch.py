from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
import pandas as pd
import data_cleansing as dc


# Reading data
house_prices = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/test.csv")


class GradientBoostingModel(object):

    def __init__(self):
        self.design_matrix = dc.clean_data(house_prices)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in
                           ['Id', 'SalePrice']]

    @staticmethod
    def _define_regressor_and_parameter_candidates():
        regressor = GradientBoostingRegressor(
            warm_start=True, max_features='auto', loss='ls', max_depth=3,
            verbose=2)
        parameters = {
            'n_estimators': [500, 1000],
            'learning_rate': [0.05, 0.07, 0.1],
            'min_samples_leaf': [1, 2, 5],
            'min_samples_split': [2, 3, 4, 5]}
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
        results.to_csv('../Model_results/GB_GridSearch_normalized_results.csv',
                       index=False)

GradientBoostingModel().grid_search_for_best_estimator()
