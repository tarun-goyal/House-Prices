from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_predict
import pandas as pd
import numpy as np
import exploratory_analysis as ea
import data_cleansing as dc


# Reading data
house_prices = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/test.csv")


def _train_validation_split(design_matrix):
    """Split data into training and validation sets"""
    training, validation = train_test_split(design_matrix, test_size=0.3)
    return training, validation


def _get_correlated_predictors(design_matrix):
    """Get the list of predictors for algorithm."""
    predictors = ea.find_correlated_features(design_matrix)
    predictors.remove('SalePrice')
    return predictors


def _clean_test_data(test_data, cols):
    """Clean test data before making any predictions"""
    cols.append('Id')
    test_data = test_data[cols]
    test_data.dropna(inplace=True)
    return test_data


class LinearRegModel(object):

    def __init__(self):
        self.design_matrix = dc.clean_data(house_prices)
        self.training, self.validation = _train_validation_split(
            self.design_matrix)
        self.predictors = _get_correlated_predictors(self.training)

    @staticmethod
    def _build_model(train_data, predictors):
        """Model-2: Linear regression model using only highly correlated
        features with houses' sale prices"""
        model = LinearRegression(fit_intercept=True, normalize=True)
        model.fit(train_data[predictors], train_data['SalePrice'])
        return model

    def _calculate_evaluation_metric(self):
        """Calculate evaluation metrics - RMSE and relative RMSE"""
        model = self._build_model(self.training, self.predictors)
        predicted = cross_val_predict(model, self.validation[self.predictors],
                                      self.validation['SalePrice'], cv=10)
        rmse = np.sqrt(np.mean((predicted - self.validation['SalePrice']) ** 2))
        relative_rmse = rmse / np.mean(self.training['SalePrice'])
        return relative_rmse

    def _make_predictions(self):
        """Predict on provided test data"""
        predictors = _get_correlated_predictors(self.design_matrix)
        test_data = _clean_test_data(test, predictors)
        model = self._build_model(self.design_matrix, predictors)
        test_data['SalePrice'] = model.predict(test_data[predictors])
        return test_data

    def submit_solution(self):
        """Submit the solution file"""
        submission = self._make_predictions()
        submission = submission[['Id', 'SalePrice']]
        submission.to_csv("../Submissions/submission_1inear_reg_" + str(
            self._calculate_evaluation_metric()) + ".csv", index=False)


LinearRegModel().submit_solution()
