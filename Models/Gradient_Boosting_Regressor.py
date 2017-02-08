from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
import pandas as pd
import numpy as np
import data_cleansing as dc


# Reading data
house_prices = pd.read_csv("../Data/train_imputed.csv")
test = pd.read_csv("../Data/test_imputed.csv")


def _train_validation_split(design_matrix):
    """Split data into training and validation sets"""
    training, validation = train_test_split(design_matrix, test_size=0.3)
    return training, validation


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


class GradientBoostingModel(object):

    def __init__(self):
        self.design_matrix = dc.clean_data(house_prices)
        self.design_matrix = _create_dummies_for_categorical_features(
            self.design_matrix)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in
                           ['Id', 'SalePrice']]
        _remove_any_more_null_rows(self.design_matrix)

    @staticmethod
    def _build_model(train_data, predictors):
        """Model: Gradient Boosting model using the default parameters"""
        model = GradientBoostingRegressor(verbose=2)
        model.fit(train_data[predictors], train_data['SalePrice'])
        return model

    def _calculate_evaluation_metric(self, iterations=10):
        """Calculate evaluation metrics - RMSE and relative RMSE"""
        rmse_group = []
        for itr in xrange(iterations):
            training, validation = _train_validation_split(self.design_matrix)
            model = self._build_model(training, self.predictors)
            predicted = cross_val_predict(model, validation[self.predictors],
                                          validation['SalePrice'], cv=10)
            rmse = np.sqrt(np.mean((np.log(predicted) -
                                    np.log(validation['SalePrice'])) ** 2))
            rmse_group.append(rmse)
        print rmse_group
        return np.mean(rmse_group)

    def _make_predictions(self):
        """Predict on provided test data"""
        test_data = dc.clean_data(test)
        test_data = _create_dummies_for_categorical_features(test_data)
        test_data = _remove_any_more_null_rows(test_data)
        predictors = [pred for pred in self.predictors if pred in list(
            test_data.columns.values)]
        features = [i for i in predictors]
        model_coefficients = pd.DataFrame(columns=features)
        model = self._build_model(self.design_matrix, predictors)
        model_coefficients.loc[0] = model.feature_importances_
        test_data['SalePrice'] = model.predict(test_data[predictors])
        return test_data, model_coefficients

    def submit_solution(self):
        """Submit the solution file"""
        submission, model_coefficients = self._make_predictions()
        submission = submission[['Id', 'SalePrice']]
        submission.to_csv(
            "../Submissions/submission_GB_default_" + str(
                self._calculate_evaluation_metric()) + ".csv", index=False)
        model_coefficients.to_csv(
            "../Model_results/GB_default_coefficients.csv", index=False)

GradientBoostingModel().submit_solution()
