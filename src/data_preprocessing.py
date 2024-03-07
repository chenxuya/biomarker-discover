import pandas as pd
from .utils.str_param import CommonParam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import numpy as np
from sklearn import preprocessing
# Python code for data_preprocessing

class DataGroup():
    def __init__(self, data:pd.DataFrame, info:pd.DataFrame) -> None:
        """
        初始化DataGroup对象。
        :param data: 包含特征数据的DataFrame，行为样本，列为特征。
        :param info: 包含分组信息的DataFrame。
        """
        self.data = data # 行为样本，列为特征
        self.info = info
        self.str_param = CommonParam()
    
    def get_compares_data(self,
        compares:str # gp1_vs_gp2;gp1_vs_gp3
        ):
        """
        根据提供的比较字符串获取对应组的数据。
        :param compares: 比较字符串，格式为"gp1_vs_gp2;gp1_vs_gp3"。
        :yield: 每一对比较组的数据和信息。
        """
        ali_compares = compares.strip().split(self.str_param.semi_sep)
        compares = [i.strip() for i in ali_compares]
        for compare in compares:
            cur_gp_data_info = self.get_compare_data(compare)
            yield compare,cur_gp_data_info

    def get_compare_data(self, 
            compare:str # gp1_vs_gp2
            ):
        """
        获取单个比较组合的数据和信息。
        :param compare: 单个比较字符串，格式为"gp1_vs_gp2"。
        :return: 包含当前比较组数据和信息的DataFrame。
        """
        gps = compare.split(self.str_param.vs_sep)
        cur_gp_info = self.info[self.info[self.str_param.group].isin(gps)]
        cur_gp_data = self.data.loc[cur_gp_info[self.str_param.sample],:] # 提取当前比较组的数据
        cur_gp_data_info = pd.concat([cur_gp_data, cur_gp_info.set_index(self.str_param.sample)], axis=1)
        return cur_gp_data_info


class DataSplit():
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the class with the given DataFrame.

        Parameters:
            data (pd.DataFrame): The DataFrame to be used for initialization.

        Returns:
            None
        """
        self.split_seed = [958, 798, 221, 421, 620, 358, 521, 670, 588, 526, 617, 594, 515, 326, 267, 519, 255, 863, 662, 511]
        self.data = data

    def split_data(self, split_times: int, train_size: float, bygroup=True, kfold=False, train_sample=None):
        """
        按照train_size 所赞比拆分数据为训练集和验证集，拆分split_times次。
        如果kfold=True, 则按照kfold的方式，拆分split_times次。否则根据split_seed 的随机种子随机拆分split_time次。
        如果bygroup=True, 则根据分组等比例拆分。
        如果指定train_sample, 则按照给定的train_sample拆分
        """
        # 按照train_size 所赞比拆分数据为训练集和验证集，拆分split_times次。
        # 如果kfold=True, 则按照kfold的方式，拆分split_times次。否则根据split_seed 的随机种子随机拆分split_time次。
        # 如果bygroup=True, 则根据分组等比例拆分。
        if train_sample is None:
            if kfold:
                # 检查最小类别数量
                min_class_count = self.data.iloc[:, -1].value_counts().min()
                if min_class_count < split_times *2:
                    raise ValueError("最小类别的数量小于 KFold 折数的一半，无法进行有效的 KFold 分割。")
                if not bygroup:
                    kf = KFold(n_splits=split_times, shuffle=True, random_state=self.split_seed[0])
                    for train_index, test_index in kf.split(self.data):
                        yield self.data.iloc[train_index], self.data.iloc[test_index]
                else:
                    skf = StratifiedKFold(n_splits=split_times, shuffle=True, random_state=self.split_seed[0])  # 需要指定一个随机种子
                    for train_index, test_index in skf.split(self.data.iloc[:, :-1], self.data.iloc[:, -1]):
                        yield self.data.iloc[train_index].copy(deep=True), self.data.iloc[test_index].copy(deep=True)
            else:
                for seed in self.split_seed[:split_times]:
                    if train_size>=1:
                            x_train, y_train = self.data.iloc[:,:-1], self.data.iloc[:, -1]
                            x_test, y_test = x_train.copy(deep=True), y_train.copy(deep=True)
                            train_index = test_index = self.data.index
                    else:
                        if bygroup:
                            x_train, x_test, y_train, y_test, train_index, test_index = train_test_split(self.data.iloc[:,:-1], self.data.iloc[:,-1],
                            self.data.index,train_size=train_size, stratify=self.data.iloc[:,-1], random_state=seed)
                        else:
                            x_train, x_test, y_train, y_test, train_index, test_index = train_test_split(self.data.iloc[:,:-1], self.data.iloc[:,-1],
                            self.data.index, train_size=train_size,
                                                random_state=seed)
                    train = pd.concat([x_train, y_train], axis=1)
                    test = pd.concat([x_test, y_test], axis=1)
                    train.index = train_index 
                    test.index = test_index
                    yield train, test
        else:
            train = self.data.loc[train_sample, :]
            test = self.data.loc[self.data.index.difference(train_sample), :]
            yield train, test

class MissingValueAnalyzer:
    def __init__(self, data: pd.DataFrame) -> None:
        """
        初始化缺失值分析器。
        :param data: DataFrame，包含待分析的数据。
        """
        self.data = data

    def summarize_missing(self) -> pd.DataFrame:
        """
        统计每列的缺失值数量和百分比。
        :return: 一个DataFrame，包含每列的缺失值统计信息。
        """
        missing_count = self.data.isnull().sum()  # 缺失值计数
        missing_percent = (missing_count / len(self.data)) * 100  # 缺失值百分比
        missing_summary = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percent': missing_percent
        })
        return missing_summary[missing_summary['Missing Count'] > 0]  # 只返回有缺失值的列

    def plot_missing_distribution(self, outfile) -> None:
        """
        绘制缺失值分布图。
        """
        missing_count = self.data.isnull().sum()
        missing_count = missing_count[missing_count > 0]
        missing_count.sort_values(inplace=True)
        missing_count.plot.barh()
        plt.title('Distribution of Missing Values')
        plt.xlabel('Number of Missing Values')
        plt.ylabel('Columns')
        plt.savefit(outfile)
        plt.close()

    def identify_high_missing_features(self, threshold: float = 50.0) -> pd.Series:
        """
        标识缺失值超过给定阈值的特征。
        :param threshold: 缺失值百分比的阈值。
        :return: 缺失值百分比超过阈值的特征名称。
        """
        summary = self.summarize_missing()
        high_missing = summary[summary['Missing Percent'] > threshold]
        return high_missing.index


class BasicMissingValueImputer:
    def __init__(self, data: pd.DataFrame) -> None:
        """
        初始化缺失值填充器。
        :param data: DataFrame，包含待填充的数据。
        """
        self.data = data

    def fill_mean(self, columns=None):
        """
        使用列的均值填充缺失值。
        :param columns: 指定需要填充的列名列表，如果为None，则填充所有数值型列。
        """
        if columns is None:
            columns = self.data.select_dtypes(include='number').columns
        self.data[columns] = self.data[columns].fillna(self.data[columns].mean())

    def fill_median(self, columns=None):
        """
        使用列的中位数填充缺失值。
        :param columns: 指定需要填充的列名列表，如果为None，则填充所有数值型列。
        """
        if columns is None:
            columns = self.data.select_dtypes(include='number').columns
        self.data[columns] = self.data[columns].fillna(self.data[columns].median())

    def fill_mode(self, columns=None):
        """
        使用列的众数填充缺失值。
        :param columns: 指定需要填充的列名列表，如果为None，则填充所有列。
        """
        if columns is None:
            columns = self.data.columns
        for column in columns:
            mode_value = self.data[column].mode()[0]
            self.data[column] = self.data[column].fillna(mode_value)

    def fill_specific_value(self, value, columns=None):
        """
        使用指定值填充缺失值。
        :param value: 用于填充的指定值。
        :param columns: 指定需要填充的列名列表，如果为None，则填充所有列。
        """
        if columns is None:
            columns = self.data.columns
        self.data[columns] = self.data[columns].fillna(value)

    def fill_forward_or_backward(self, method='ffill', columns=None):
        """
        使用前向填充或后向填充缺失值。
        :param method: 填充方法，'ffill'为前向填充，'bfill'为后向填充。
        :param columns: 指定需要填充的列名列表，如果为None，则填充所有列。
        """
        if columns is None:
            columns = self.data.columns
        self.data[columns] = self.data[columns].fillna(method=method)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd

class MLMissingValueImputer:
    def __init__(self, data: pd.DataFrame) -> None:
        """
        初始化机器学习缺失值填充器。
        :param data: DataFrame，包含待填充的数据。
        """
        self.data = data

    def fill_knn(self, n_neighbors=5):
        """
        使用KNN算法填充缺失值。
        :param n_neighbors: 用于KNN算法的邻居数。
        """
        imputer = KNNImputer(n_neighbors=n_neighbors)
        self.data = pd.DataFrame(imputer.fit_transform(self.data),
                                 columns=self.data.columns,
                                 index=self.data.index)

    def fill_random_forest(self, max_iter=10, random_state=0):
        """
        使用随机森林填充缺失值。
        :param max_iter: 最大迭代次数。
        :param random_state: 随机状态，确保可重复性。
        """
        imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=random_state),
                                   max_iter=max_iter,
                                   random_state=random_state)
        self.data = pd.DataFrame(imputer.fit_transform(self.data),
                                 columns=self.data.columns,
                                 index=self.data.index)

    def fill_gradient_boosting(self, max_iter=10, random_state=0):
        """
        使用梯度提升树填充缺失值。
        :param max_iter: 最大迭代次数。
        :param random_state: 随机状态，确保可重复性。
        """
        imputer = IterativeImputer(estimator=GradientBoostingRegressor(random_state=random_state),
                                   max_iter=max_iter,
                                   random_state=random_state)
        self.data = pd.DataFrame(imputer.fit_transform(self.data),
                                 columns=self.data.columns,
                                 index=self.data.index)


class Preprocess():
    def __init__(self, x_train, x_test, y_train,y_test,
        transform:str,
        scale:str,
        category:list,
        norminal:list,
        norminal_scale,
        ) -> None:
        # cfgs = Param(cfg_path=cfg_path)
        self.transform = transform
        self.scale = scale
        self.norminal_scale = norminal_scale
        self.x_train = x_train.copy(deep=True)
        self.x_test = x_test.copy(deep=True)
        self.y_train = y_train
        self.y_test = y_test
        self.category = category
        self.norminal = norminal
        self.train_index = self.x_train.index # 训练集样本编号
        self.test_index = self.x_test.index # test set 样本编号
        print(f"""preprocess data...
                    1 metabolite(protein) transform:{self.transform}
                    2 metabolite(protein) scale:{self.scale}
                    3 numeric clinical index transform: z-score
                    4 category clinical index transform: one-hot encoding
                    """)

    def meta_trans(self, meta_var)->pd.DataFrame:
        # 代谢物数据的log2，z-score等转换
        meta_train = self.x_train.loc[:,meta_var] # shallow copy
        meta_test = self.x_test.loc[:, meta_var]
        if self.transform=="log2":
            min_value = min(self.x_train.min().min(), self.x_test.min().min())
            if min_value < 0.00000001/100:
                adjustment_value = min_value / 10
            else:
                adjustment_value = 0.00000001
            self.x_train.loc[:,meta_var] = np.log2(meta_train+adjustment_value) # 这里的修改在x_train上直接修改。
            self.x_test.loc[:,meta_var] = np.log2(meta_test+adjustment_value)
        if self.scale=="z-score":
            scaler = preprocessing.StandardScaler().fit(self.x_train.loc[:,meta_var])
            self.x_train.loc[:, meta_var] = scaler.transform(self.x_train.loc[:,meta_var])
            self.x_test.loc[:, meta_var] = scaler.transform(self.x_test.loc[:, meta_var])
        elif self.scale =="min-max":
            scaler = preprocessing.MinMaxScaler().fit(self.x_train.loc[:,meta_var])
            self.x_train.loc[:, meta_var] = scaler.transform(self.x_train.loc[:,meta_var])
            self.x_test.loc[:, meta_var] = scaler.transform(self.x_test.loc[:, meta_var])
        
    def norminal_trans(self, norminal_var)->pd.DataFrame:
        # 正态型变量的转换，比如年龄等其他特征
        norminal_var = self.x_train.columns.intersection(norminal_var)
        norm_train = self.x_train.loc[:,norminal_var]
        norm_test = self.x_test.loc[:, norminal_var]
        if self.norminal_scale == "z-score":
            scaler = preprocessing.StandardScaler().fit(norm_train)
        elif self.norminal_scale == "min-max":
            scaler = preprocessing.MinMaxScaler().fit(norm_train)
        self.x_train.loc[:,norminal_var] = scaler.transform(norm_train)
        self.x_test.loc[:, norminal_var] = scaler.transform(norm_test)

    def categorical_trans(self, categorical_var)->pd.DataFrame:
        # 类别型变量的编码
        categorical_var = self.x_train.columns.intersection(categorical_var)
        cat_train = self.x_train.loc[:, categorical_var]
        cat_test = self.x_test.loc[:,categorical_var]
        encoder = ce.OneHotEncoder(use_cat_names=True).fit(cat_train, self.y_train)
        coded_train = encoder.transform(cat_train)
        coded_test = encoder.transform(cat_test)
        self.x_train.drop(categorical_var, axis=1, inplace=True)
        self.x_test.drop(categorical_var, axis=1, inplace=True)
        self.x_train = pd.concat([self.x_train,coded_train], axis=1)
        self.x_test = pd.concat([self.x_test, coded_test], axis=1)

    def process(self):
        # 当数据集中纯在类别型变量，正态型变量时，用此函数处理。
        if self.category is not None:
            meta_var = self.x_train.columns.difference(self.category)
        if self.norminal is not None:
            meta_var = meta_var.difference(self.norminal)
        if self.norminal is None and self.category is None:
            meta_var = self.x_train.columns
        if meta_var.to_list():
            self.meta_trans(meta_var)
        if self.norminal is not None:
            self.norminal_trans(self.norminal)
        if self.category is not None:
            self.categorical_trans(self.category)
        return self.x_train, self.x_test, self.y_train, self.y_test

