from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
import pandas as pd
import numpy as np
import data_cleansing as dc


# Reading data
house_prices = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/test.csv")


def _train_validation_split(design_matrix):
    """Split data into training and validation sets"""
    training, validation = train_test_split(design_matrix, test_size=0.3)
    return training, validation


class GradientBoostingModel(object):

    def __init__(self):
        self.mean = np.mean(house_prices['SalePrice'])
        self.stdev = np.std(house_prices['SalePrice'])
        self.design_matrix = dc.clean_data(house_prices)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in
                           ['Id', 'SalePrice']]

    @staticmethod
    def _build_model(train_data, predictors):
        """Model: Gradient Boosting model using the best estimator
        parameters"""
        model = GradientBoostingRegressor(
            n_estimators=1000, warm_start=True, max_features='auto',
            min_samples_split=4, min_samples_leaf=1, learning_rate=0.1,
            loss='ls', max_depth=3, verbose=2)
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
            # Un-normalizing the output before calculating error
            predicted = (predicted * self.stdev) + self.mean
            validation['SalePrice'] = (validation['SalePrice'] * self.stdev)\
                + self.mean
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
        features = [i for i in predictors]
        model_coefficients = pd.DataFrame(columns=features)
        model = self._build_model(self.design_matrix, predictors)
        model_coefficients.loc[0] = model.feature_importances_
        test_data['SalePrice'] = model.predict(test_data[predictors])
        test_data['SalePrice'] = (test_data['SalePrice'] * self.stdev)\
            + self.mean
        return test_data, model_coefficients

    def submit_solution(self):
        """Submit the solution file"""
        predictions, model_coefficients = self._make_predictions()
        submission = test[['Id']]
        submission['SalePrice'] = predictions['SalePrice']
        submission.to_csv(
            "../Submissions/submission_GB_normalized_" + str(
                self._calculate_evaluation_metric()) + ".csv", index=False)
        #model_coefficients.to_csv(
        #    "../Model_results/GB_best_estimator_coefficients.csv", index=False)

GradientBoostingModel().submit_solution()
