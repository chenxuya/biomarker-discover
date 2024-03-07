import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import numpy as np

class FeatureSelectorIndividual:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def cumulative_importance_selection(self, cumulative_threshold=0.9):
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
        # 创建一个空的DataFrame用于存储每个模型筛选后的特征
        selected_features = pd.DataFrame(index=range(top_n))

        for model in self.dataframe.columns:
            # 对当前模型的特征重要性进行排序并选择前Top N个
            top_features = self.dataframe[model].sort_values(ascending=False).head(top_n)
            selected_features[model] = top_features.index
        # 移除全为NaN的行
        selected_features.dropna(axis=0, how='all', inplace=True)
        return selected_features
# 生成示例数据
np.random.seed(0)  # 为了可复现性
features = [f"Feature_{i}" for i in range(1, 101)]
models = ["Model_A", "Model_B", "Model_C", "Model_D"]
data = np.random.rand(100, 4)  # 生成100个特征在4个模型中的重要性

# 创建DataFrame
example_df = pd.DataFrame(data, index=features, columns=models)

# 使用修改后的类
selector_individual = FeatureSelectorIndividual(example_df)

# 策略1：累计重要性选择
cumulative_selected_features = selector_individual.cumulative_importance_selection(cumulative_threshold=0.96)

# 策略2：Top N选择
top_n_selected_features = selector_individual.top_n_selection(top_n=10)

print(cumulative_selected_features,"\n", top_n_selected_features.head())
