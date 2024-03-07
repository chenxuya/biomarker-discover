import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import random
from skopt.space import Real
from sklearn.metrics import roc_auc_score



class CustomLogModel():
    def __init__(self, times, method):
        self.random_seed_list =  random.sample(range(10, 10001), times)
        self.models = []
        self.method = method
        self.feature_names_in_ = None
        self.coef_ = None
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        for seed in self.random_seed_list:
            np.random.seed(seed)
            if self.method=="LASSO":
                # 使用L1正则化
                logistic_regression = LogisticRegression(penalty='l1', solver='saga', random_state=seed, max_iter=1000)

                # 设置GridSearchCV参数
                params = {'C': np.logspace(-4, 4, 20)}
            elif self.method=="LR":
                logistic_regression = LogisticRegression(solver="saga", random_state=seed, max_iter=1000)
                params = {"C":np.logspace(-3,3,10), "penalty":["l2"]}# l1 lasso l2 ridge
            elif self.method =="ELASTICNET":
                logistic_regression = LogisticRegression(penalty="elasticnet",solver="saga", random_state=seed, max_iter=1000)
                params = {"C":np.logspace(-3,3,7),
                            "l1_ratio":np.linspace(0, 1, 20)}
            else:
                raise ValueError("valid method in [LASSO, LR, ELASTICNET]")
            
            grid_search = GridSearchCV(logistic_regression, params, cv=5, n_jobs=20)

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            # print(grid_search.get_params())
            
            self.feature_names_in_ = best_model.feature_names_in_
            self.models.append(best_model)
        self.get_coef()

    def predict(self, X_test):
        predictions = []
        for model in self.models:
            prediction = model.predict(X_test)
            predictions.append(prediction)

        # 将所有模型的输出值取平均值
        average_prediction = np.mean(predictions, axis=0)
        # 将连续值转换为类别(四舍五入)
        average_prediction = np.round(average_prediction).astype(int)

        return average_prediction

    def predict_proba(self, X_test):
        predict_probability = []
        for model in self.models:
            prediction = model.predict_proba(X_test)
            predict_probability.append(prediction)

        # 将所有模型的输出值取平均值
        average_probability = np.mean(predict_probability, axis=0)
        return average_probability

    def get_coef(self):
        coefs = []
        for model in self.models:
            coef = model.coef_
            coefs.append(coef)
        # 将所有模型的输出值取平均值
        average_coef = np.mean(coefs, axis=0)
        self.coef_ = average_coef

    def get_params(self, deep=True):
        best_model = None
        best_auc = -1
        for model in self.models:
            y_pred_proba = model.predict_proba(self.X_train)[:, 1]
            auc = roc_auc_score(self.y_train, y_pred_proba)
            if auc > best_auc:
                best_auc = auc
                best_model = model

        return best_model.get_params() if best_model else None

