from sklearn.ensemble import RandomForestRegressor
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import data_cleansing as dc


# Reading data
house_prices = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/test.csv")


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


class RandomForestModel(object):

    def __init__(self):
        self.design_matrix = dc.clean_data(house_prices)
        self.design_matrix = _create_dummies_for_categorical_features(
            self.design_matrix)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in
                           ['Id', 'SalePrice']]
        _remove_any_more_null_rows(self.design_matrix)

    @staticmethod
    def _define_regressors():
        rf_regressors = [
            ("RandomForestRegressor, max_features='sqrt'",
             RandomForestRegressor(max_features='sqrt', warm_start=True,
                                   oob_score=True)),
            ("RandomForestRegressor, max_features='log2'",
             RandomForestRegressor(max_features='log2', warm_start=True,
                                   oob_score=True)),
            ("RandomForestRegressor, max_features='None'",
             RandomForestRegressor(max_features=None, warm_start=True,
                                   oob_score=True))
        ]
        return rf_regressors

    def calculate_oob_error_rate(self):
        """Calculate out-of-bag error rate for different number of
        max_features"""
        min_estimators = 1
        max_estimators = 50
        error_rate = OrderedDict((label, []) for label, _ in self.
                                 _define_regressors())
        for label, regressor in self._define_regressors():
            for trees in range(min_estimators, max_estimators + 1):
                print label, trees
                regressor.set_params(n_estimators=trees)
                regressor.fit(self.design_matrix[self.predictors],
                              self.design_matrix['SalePrice'])
                oob_error = 1 - regressor.oob_score_
                error_rate[label].append((trees, oob_error))
        return error_rate


for label, regressor_error in RandomForestModel().calculate_oob_error_rate()\
        .iteritems():
    xs, ys = zip(*regressor_error)
    plt.plot(xs, ys, label=label)

plt.xlim(1, 50)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
plt.savefig('../Model_results/rf_oob_error_rate.png')
