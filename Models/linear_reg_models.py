from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import exploratory_analysis as ea
import data_cleansing as dc


# Reading data
house_prices = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/test.csv")


def _get_clean_design_matrix(design_matrix):
    """Get the cleaned design matrix"""
    design_matrix = dc.clean_data(design_matrix)
    return design_matrix


def _train_validation_split(design_matrix):
    """Split data into training and validation sets"""
    training, validation = train_test_split(design_matrix, test_size=0.3)
    return training, validation


def _get_correlated_predictors(design_matrix):
    """Get the list of predictors for algorithm."""
    predictors = ea.find_correlated_features(design_matrix)
    predictors.remove('SalePrice')
    return predictors


def _get_elements_for_building_model():
    """Accumulate elements required for building the model."""
    design_matrix = _get_clean_design_matrix(house_prices)
    training, validation = _train_validation_split(design_matrix)
    predictors = _get_correlated_predictors(training)
    return training, validation, predictors


def _build_model(training, predictors):
    """Model-1: Linear regression model using only highly correlated features
    with houses' sale prices"""
    model = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    model.fit(training[predictors], training['SalePrice'])
    return model


def _make_predictions(model, test_data, predictors):
    """Make predictions on the provided data"""
    predictions = model.predict(test_data[predictors])
    return predictions


def _calculate_evaluation_metric():
    """Calculate evaluation metrics - RMSE and relative RMSE"""
    training, validation, predictors = _get_elements_for_building_model()
    model = _build_model(training, predictors)
    rmse = np.sqrt(np.mean((_make_predictions(model, validation, predictors)
                            - validation['SalePrice']) ** 2))
    relative_rmse = rmse / np.mean(training['SalePrice'])
    return relative_rmse


def _clean_test_data(test_data, cols):
    """Clean test data before making any predictions"""
    cols.append('Id')
    test_data = test_data[cols]
    test_data.dropna(inplace=True)
    return test_data


def _predict_using_test_data():
    """Predict on provided test data"""
    train_data = _get_clean_design_matrix(house_prices)
    predictors = _get_correlated_predictors(train_data)
    test_data = _clean_test_data(test, predictors)
    model = _build_model(train_data, predictors)
    test_data['SalePrice'] = model.predict(test_data[predictors])
    return test_data


def submit_solution():
    """Submit the solution file"""
    submission = _predict_using_test_data()
    submission = submission[['Id', 'SalePrice']]
    submission.to_csv("../Submissions/submission_1inear_reg_" + str(
        _calculate_evaluation_metric()) + ".csv", index=False)


submit_solution()
