from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
import pandas as pd
import data_cleansing as dc


# Reading data
house_prices = pd.read_csv("../Data/train.csv")


class XGBoostingModel(object):

    def __init__(self):
        self.design_matrix = dc.clean_data(house_prices)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in
                           ['Id', 'SalePrice']]

    @staticmethod
    def _define_regressor_and_parameter_candidates():
        regressor = XGBRegressor(
            n_estimators=2066, learning_rate=0.01, objective='reg:linear',
            max_depth=5, min_child_weight=3, gamma=0)
        parameters = {'subsample': [i/10.0 for i in range(1, 10)],
                      'colsample_bytree': [i/10.0 for i in range(1, 10)]}
        return regressor, parameters

    def grid_search_for_best_estimator(self):
        """Comprehensive search over provided parameters to find the best
        estimator"""
        regressor, parameters = self\
            ._define_regressor_and_parameter_candidates()
        model = GridSearchCV(regressor, parameters, cv=5, verbose=2,
                             scoring='neg_mean_squared_error', iid=False)
        model.fit(self.design_matrix[self.predictors],
                  self.design_matrix['SalePrice'])
        print model.best_params_
        print model.best_score_
        # cv_results = model.cv_results_
        # results = DataFrame.from_dict(cv_results, orient='columns')
        # results.to_csv('../Model_results/XGB_GridSearch13_results.csv',
        #                index=False)

XGBoostingModel().grid_search_for_best_estimator()
