import pandas as pd
import numpy as np
import data_cleansing as dc
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_predict


# Reading data
house_prices = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/test.csv")


def _train_validation_split(design_matrix):
    """Split data into training and validation sets"""
    training, validation = train_test_split(design_matrix, test_size=0.3)
    return training, validation


class XGBoostingModel(object):

    def __init__(self):
        self.design_matrix = dc.clean_data(house_prices)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in
                           ['Id', 'SalePrice']]

    @staticmethod
    def _build_model(train_data, predictors):
        """Model: Extreme Gradient Boosting model using tuned parameters"""
        model = XGBRegressor(
            n_estimators=1530, learning_rate=0.01, max_depth=5,
            min_child_weight=1, gamma=0, subsample=0.7, colsample_bytree=0.65,
            reg_alpha=3.15, objective='reg:linear')
        model.fit(train_data[predictors], train_data['SalePrice'])
        return model

    def _calculate_evaluation_metric(self, iterations=10):
        """Calculate cross validation score based on RMSE"""
        rmse_group = []
        for itr in xrange(iterations):
            training, validation = _train_validation_split(self.design_matrix)
            model = self._build_model(training, self.predictors)
            predicted = cross_val_predict(model, validation[self.predictors],
                                          validation['SalePrice'], cv=5,
                                          verbose=2)
            rmse = np.sqrt(np.mean((np.log(predicted) -
                                    np.log(validation['SalePrice'])) ** 2))
            rmse_group.append(rmse)
        print rmse_group
        return np.mean(rmse_group)

    def _make_predictions(self):
        """Predict on provided test data"""
        test_data = dc.clean_data(test)
        predictors = [pred for pred in self.predictors if pred in list(
            test_data.columns.values)]
        model = self._build_model(self.design_matrix, predictors)
        test_data['SalePrice'] = model.predict(test_data[predictors])
        return test_data

    def submit_solution(self):
        """Submit the solution file"""
        eval_metric = self._calculate_evaluation_metric()
        predictions = self._make_predictions()
        submission = test[['Id']]
        submission['SalePrice'] = predictions['SalePrice']
        submission.to_csv(
            "../Submissions/submission_XGB_best_estimator3_" + str(
                eval_metric) + ".csv", index=False)

XGBoostingModel().submit_solution()
