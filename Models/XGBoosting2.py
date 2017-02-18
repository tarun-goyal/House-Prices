import pandas as pd
import data_cleansing as dc
import xgboost as xgb
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from xgboost.sklearn import XGBRegressor


# Reading data
house_prices = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/test.csv")
rcParams['figure.figsize'] = 18, 12


class XGBoostingModel(object):

    def __init__(self):
        self.design_matrix = dc.clean_data(house_prices)
        self.predictors = [ele for ele in list(
            self.design_matrix.columns.values) if ele not in
                           ['Id', 'SalePrice']]

    @staticmethod
    def _build_model():
        """Model: Extreme Gradient Boosting model using tuned parameters"""
        model = XGBRegressor(
            n_estimators=5000, learning_rate=0.01, max_depth=5,
            min_child_weight=1, gamma=0, subsample=0.7, colsample_bytree=0.65,
            reg_alpha=3.15, objective='reg:linear')
        return model

    def _execute_cross_validation(self, model, cv_folds=5,
                                  early_stopping_rounds=50):
        """Get cross validated results based on RMSE"""
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(self.design_matrix[self.predictors].values,
                              label=self.design_matrix['SalePrice'].values)
        cv_results = xgb.cv(xgb_param, xgtrain, nfold=cv_folds, metrics='rmse',
                            num_boost_round=model.get_params()['n_estimators'],
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=True)
        model.set_params(n_estimators=cv_results.shape[0])

        # Fit the algorithm on the data
        model.fit(self.design_matrix[self.predictors],
                  self.design_matrix['SalePrice'], eval_metric='rmse')

        # Print model report:
        print "\nCV Results"
        print cv_results
        cv_results.to_csv("../Model_results/XGB_tuned_CV2_results.csv")
        feat_imp = pd.Series(model.booster().get_fscore()).sort_values(
            ascending=False)
        feat_imp = feat_imp[:50]
        feat_imp.plot(kind='bar', title='Feature Importance')
        plt.ylabel('Feature Importance Score')
        plt.savefig('../Model_results/XGB_Top50_feat2_imp.png')

    def _make_predictions(self):
        """Predict on provided test data"""
        test_data = dc.clean_data(test)
        predictors = [pred for pred in self.predictors if pred in list(
            test_data.columns.values)]
        model = self._build_model()
        model.fit(self.design_matrix[predictors],
                  self.design_matrix['SalePrice'])
        test_data['SalePrice'] = model.predict(test_data[predictors])
        return test_data

    def submit_solution(self):
        """Submit the solution file"""
        self._execute_cross_validation(model=self._build_model())
        # predictions = self._make_predictions()
        # submission = test[['Id']]
        # submission['SalePrice'] = predictions['SalePrice']
        # submission.to_csv(
        #     "../Submissions/submission_XGB_CV_tuned2.csv", index=False)

XGBoostingModel().submit_solution()
