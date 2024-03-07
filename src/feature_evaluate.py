import pandas as pd
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
import shap
import pickle
import os

class RFEFeatureSelector:
    def __init__(self, model):
        self.model = model
        self.selector = None
        self.feature_name = None

    def fit(self, X, y):
        """
        Fits the model to the training data using recursive feature elimination to select the most important features.
        
        Parameters:
            X: DataFrame
                The input features for training.
            y: Series
                The target variable for training.
        
        Returns:
            None
        """
        self.feature_name = X.columns
        try:
            self.selector = RFECV(estimator=self.model, step=1, cv=StratifiedKFold(5), scoring='accuracy')
            self.selector = self.selector.fit(X, y)
        except Exception as e:
            print(f"RFECV failed with error {e}, falling back to RFE")
            self.selector = RFE(estimator=self.model, n_features_to_select=1)
            self.selector = self.selector.fit(X, y)

    def rank_features(self):
        """
        Rank the features based on the fitted selector and return the rankings as a pandas Series.
        """
        if self.selector is None:
            raise ValueError("Selector not fitted, please call fit first.")
        rankings = pd.Series(1/self.selector.ranking_, index=self.feature_name).sort_values(ascending=False)
        rankings.name = "Importance"
        return rankings

    def transform(self, X, n_features):
        """
        Transform the input data X using the top n_features selected features.
        
        Parameters:
            X: array-like
                The input data to transform.
            n_features: int
                The number of top features to select.
        
        Returns:
            array-like
                The transformed data with only the selected top features.
        """
        if self.selector is None:
            raise ValueError("Selector not fitted, please call fit first.")
        top_features = self.rank_features().nlargest(n_features).index
        return X[top_features]

    def load_model(self, filename):
        if not os.path.exists(filename):
            raise ValueError("File %s not exists, Please train model first" % filename)
        # 导入整个对象
        with open(filename, 'rb') as file:
            loaded_model = pickle.load(file)
            self.model = loaded_model['model']
            self.selector = loaded_model['selector']
            self.feature_name = loaded_model['feature_name']
        
    def save_model(self, filename):
        # 将selector的重要属性打包成一个字典
        model_to_save = {
            'model': self.model,
            'selector': self.selector,
            'feature_name': self.feature_name
        }
        # 使用pickle保存模型到文件
        with open(filename, 'wb') as file:
            pickle.dump(model_to_save, file)

class SHAPFeatureSelector:
    def __init__(self, model, refitmodel=False):
        self.model = model
        self.explainer = None
        self.shap_values = None
        self.refitmodel = refitmodel
        self.feature_name = None

    def fit(self, X, y):
        """
        Fit the model to the training data and calculate SHAP values for feature interpretation.

        Parameters:
            X: pandas DataFrame
                The input features for training the model.
            y: pandas Series
                The target variable for training the model.

        Returns:
            None
        """
        self.feature_name = X.columns
        # 训练模型
        if self.refitmodel:
            self.model.fit(X, y)
        # 创建解释器
        self.explainer = shap.Explainer(self.model, X)
        # 计算SHAP值
        self.shap_values = self.explainer(X)

    def rank_features(self):
        """
        Calculate the feature importance rankings based on SHAP values.

        Returns:
            pandas.Series: A series containing the feature importance rankings.
        """
        # 检查是否已计算SHAP值
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated, please call fit first.")
        # 计算每个特征的平均绝对SHAP值
        if len(self.shap_values.shape)==2:
            shap_sum = np.abs(self.shap_values.values).mean(axis=0)
        elif len(self.shap_values.shape)==3:
            shap_sum = np.abs(self.shap_values.values).mean(axis=(0,2))
        rankings = pd.Series(shap_sum, index=self.feature_name).sort_values(ascending=False)
        rankings.name = "Importance"
        return rankings

    def transform(self, X, n_features):
        """
        Transform the input features by selecting the top n_features based on feature importance rankings.

        Parameters:
        - X: The input feature matrix
        - n_features: The number of top features to select

        Returns:
        - The input feature matrix with only the top selected features
        """
        # 获取特征重要性排名
        rankings = self.rank_features()
        # 选择最重要的n个特征
        top_features = rankings.nlargest(n_features).index
        return X.loc[:, top_features]

    def save_model(self, filename):
        # 检查解释器和SHAP值是否已计算
        if self.explainer is None or self.shap_values is None:
            raise ValueError("Model not fitted or SHAP values not calculated. Please call fit first.")
        
        # 将对象的重要属性保存为字典
        model_to_save = {
            'model': self.model,
            'explainer': self.explainer,
            'shap_values': self.shap_values,
            'refitmodel': self.refitmodel,
            'feature_name': self.feature_name
        }
        
        # 使用pickle保存模型到文件
        with open(filename, 'wb') as file:
            pickle.dump(model_to_save, file)
            
    def load_model(self, filename):
        # 使用pickle从文件加载模型
        with open(filename, 'rb') as file:
            loaded_data = pickle.load(file)
            self.model = loaded_data['model']
            self.explainer = loaded_data['explainer']
            self.shap_values = loaded_data['shap_values']
            self.refitmodel = loaded_data['refitmodel']
            self.feature_name = loaded_data['feature_name']

class FeatureSelectorIndividual:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def cumulative_importance_selection(self, cumulative_threshold=0.9):
        """
        Function to perform cumulative importance feature selection.

        :param cumulative_threshold: float, default 0.9
            The threshold for cumulative importance.
        :return: pandas DataFrame
            DataFrame containing the selected features for each model.
        """
        # 创建一个空的DataFrame用于存储每个模型筛选后的特征
        selected_features_ali = []

        for model in self.dataframe.columns:
            # 对当前模型的特征重要性进行排序
            sorted_features = self.dataframe[model].sort_values(ascending=False)
            # 计算累计重要性
            cumulative_importance = sorted_features.cumsum()
            sum_importance = sorted_features.sum()
            # 筛选累计重要性在阈值内的特征
            selected_indices = pd.Series(cumulative_importance[cumulative_importance/sum_importance <= cumulative_threshold].index.to_list())
            selected_features_ali.append(selected_indices)

        selected_features = pd.concat(selected_features_ali, axis=1)
        selected_features.columns = self.dataframe.columns
        return selected_features

    def top_n_selection(self, top_n=10):
        """
        This function selects the top N features for each model and stores them in a DataFrame.
        
        :param top_n: The number of top features to select (default is 10)
        :return: A DataFrame containing the selected top features for each model
        """
        # 创建一个空的DataFrame用于存储每个模型筛选后的特征
        selected_features = pd.DataFrame(index=range(top_n))
        for model in self.dataframe.columns:
            # 对当前模型的特征重要性进行排序并选择前Top N个
            top_features = self.dataframe[model].sort_values(ascending=False).head(top_n)
            selected_features[model] = top_features.index
        # 移除全为NaN的行
        selected_features.dropna(axis=0, how='all', inplace=True)
        return selected_features



if __name__ == "__main__":

    # # 测试FeatureSelector类
    # # 生成示例数据
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=20, n_informative=5, n_redundant=5, n_clusters_per_class=1, random_state=42)
    X = pd.DataFrame(X, columns=[f'特征{i+1}' for i in range(20)])
    y = pd.Series(y)
    # model = RandomForestClassifier()
    # # 初始化FeatureSelector并进行训练
    # selector = RFEFeatureSelector(model)
    # selector.fit(X, y)

    # # 计算每个特征的rank得分
    # rankings = selector.rank_features()

    # # 进行特征筛选，选择最重要的5个特征
    # X_transformed = selector.transform(X, 5)

    # print(rankings, X_transformed.head())

    # 测试SHAPFeatureSelector类
    # 使用逻辑回归作为模型示例
    # model = LogisticRegression(max_iter=1000)
    # model = RandomForestClassifier()
    model = XGBClassifier(eval_metric='logloss')
    # model = SVC(kernel="linear") # 其他的核函数需要修改SHAP的接口
    selector = SHAPFeatureSelector(model, refitmodel=True)

    # 使用之前生成的示例数据
    selector.fit(X, y)

    # 计算每个特征的SHAP重要性排名
    rankings = selector.rank_features()

    # 进行特征筛选，选择最重要的5个特征
    X_transformed = selector.transform(X, 5)

    print(rankings.head(), X_transformed.head())
