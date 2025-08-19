import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

class UnivariateFeatureSelector:
    def __init__(self, X, y):
        """
        初始化特征选择器。
        :param X: 特征数据，类型为 pandas DataFrame。
        :param y: 目标变量，类型为 pandas Series。
        """
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X should be a pandas DataFrame and y should be a pandas Series.")
        self.X = X
        self.y = y
        self.scores_ = None

    def select_features(self, method='chi2', k='all'):
        """
        根据指定的方法选择特征。
        :param method: 字符串，指定使用的方法（'chi2', 'mutual_info', 'f_classif', 'pearson', 'logistic'）。
        :param k: 选择的特征数量，'all' 表示选择所有特征。
        :return: 选择的特征 DataFrame。
        """
        if method in ['chi2', 'mutual_info', 'f_classif']:
            if method == 'chi2':
                selector = SelectKBest(chi2, k=k)
            elif method == 'mutual_info':
                selector = SelectKBest(mutual_info_classif, k=k)
            elif method == 'f_classif':
                selector = SelectKBest(f_classif, k=k)
            self.X_new = selector.fit_transform(self.X, self.y)
            self.scores_ = selector.scores_
            selected_features = self.X.columns[selector.get_support(indices=True)]
            return pd.DataFrame(self.X_new, columns=selected_features)

        elif method == 'pearson':
            return self.pearson_corr(k)
        elif method == 'logistic':
            return self.logistic_coeffs(k)
        else:
            raise ValueError("Unsupported method.")

    def pearson_corr(self, k='all'):
        """
        计算特征与目标变量之间的皮尔森相关系数。
        :param k: 选择的特征数量，'all' 表示选择所有特征。
        :return: 选择的特征 DataFrame。
        """
        corr_coeffs = self.X.apply(lambda col: col.corr(self.y))
        self.scores_ = np.abs(corr_coeffs)  # 取绝对值
        sorted_features = self.scores_.sort_values(ascending=False).index
        
        if k == 'all':
            return self.X[sorted_features]
        return self.X[sorted_features[:k]]

    def logistic_coeffs(self, k='all'):
        """
        使用逻辑回归的系数作为特征重要性的指标。
        :param k: 选择的特征数量，'all' 表示选择所有特征。
        :return: 选择的特征 DataFrame。
        """
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X, self.y)
        self.scores_ = pd.Series(np.abs(model.coef_[0]), index=self.X.columns)
        sorted_features = self.scores_.sort_values(ascending=False).index
        
        if k == 'all':
            return self.X[sorted_features]
        return self.X[sorted_features[:k]]

# 使用示例
from sklearn.datasets import load_iris
data = load_iris()
X, y = pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)

selector = UnivariateFeatureSelector(X, y)
X_new = selector.select_features(method='chi2', k=2)
print("Selected features using Chi-squared test:\n", X_new)
