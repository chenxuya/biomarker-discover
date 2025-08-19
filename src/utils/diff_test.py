import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

class MetaboliteRawFilter():
    def __init__(self, X:pd.DataFrame, y:pd.Series, pair:bool):
        self.X = X # 行为样本，列为代谢物
        self.y = y
        self.y.index = X.index
        self.p_cutoff = None
        self.adjust_p_cutoff = None
        self.fold_change_cutoff = None
        self.data = pd.concat([self.X, self.y], axis=1)
        self.pair = pair # 是否为配对样本
        self.diff_stats = None
    @staticmethod
    def difference_testing(df1:pd.DataFrame, df2:pd.DataFrame, df1_group:str, df2_group:str, pair=False)->pd.DataFrame:
        """
        runing test between two group.p值检验方式:
        当两组样本独立时：
            两组数据都服从正态分布,且方差齐次时,用标准ttest。
            两组数据都服从正态分布,且方差不齐次时,用Welch's t-test。
            有一组数据不服从正态分布,用Mann-Whitney rank test。
        当两组样本相关时：
            两组数据都服从正态分布,且方差齐次时,用pair ttest。
            否则用Wilcox检验, 成对数据。
        Parameters:
        ---------
        df1
            group1,row represent metabolity, col represent samples. (num_metabolity, num_samples), normal samples.
        df2
            group2,the shape is same with df1, case samples
        df1_group
            group name of df1
        df2_group
            group name of df2
        ---------
        return:
        dataframe with columns data1_norm_P,data2_norm_P,var_lev_P,final_P,FDR,mean_FDC, median_FDC
        """
        row1, row2 = df1.shape[0], df2.shape[0]
        col1, col2 = df1.shape[1], df2.shape[1]
        assert row1 == row2, f"the row of df1 {df1.shape[0]} not equal to df2 {df2.shape[0]}"
        assert col1>0 and col2>0 , f"df1{col1}, df2{col2}"
        df1 = df1.astype(np.float64);df2=df2.astype(np.float64)
        # 正态分布检验
        def _norm(df:pd.DataFrame):
            norm1 = df.apply(stats.shapiro, axis=1)
            df1_p = [i[1] for i in norm1]
            df1_t = [True if i>0.05 else False for i in df1_p]
            return df1_p, df1_t
        df1_p, df1_t = _norm(df1)
        df2_p, df2_t = _norm(df2)
        shap_t = [True if (i and j) else False for i,j in zip(df1_t, df2_t)]
        # 方差齐次检验
        lev_p = []
        for i in range(row1):
            x ,y = df1.iloc[i],df2.iloc[i]
            if min(x)==max(x) and min(y)==max(y):
                lev_p.append(1.0)
            else:
                lev_p.append(stats.levene(x,y)[1]) 
        lev_t = [True if i>0.05 else False for i in lev_p]
        f_p_list = []
        for i in range(row1):
            x, y = df1.iloc[i], df2.iloc[i]
            # 正态且方差齐
            if shap_t[i] and lev_t[i]:
                if min(x)==max(x) and min(y)==max(y):
                    f_p = 1.0
                elif pair: # pair ttest
                    f_p = stats.ttest_rel(x, y)[1]
                else:
                    f_p = stats.ttest_ind(x, y)[1]
            # 正态方差不齐
            elif shap_t[i] and (not lev_t[i]):
                if pair:
                    f_p = stats.wilcoxon(x, y, zero_method='wilcox', correction=False)[1]
                else:
                    f_p = stats.ttest_ind(x, y, equal_var=False)[1]
            # 不服从正态分布
            else:
                if pair:
                    f_p = stats.wilcoxon(x, y, zero_method='wilcox', correction=False)[1]
                else:
                    f_p = stats.mannwhitneyu(x, y, alternative="two-sided")[1]
            f_p_list.append(f_p)
        fdr = fdrcorrection(f_p_list)[1]
        data1_mean = df1.mean(axis=1)
        data2_mean = df2.mean(axis=1)
        data1_median = df1.median(axis=1)
        data2_median = df2.median(axis=1)
        mean_fold_change = data2_mean/data1_mean # 这里如果出现0除的情况index的顺序会改变
        median_fold_change = data2_median/data1_median
        
        res_df = pd.DataFrame({f"{df1_group}_norm_P":list(df1_p),
                            f"{df2_group}_norm_P":list(df2_p),
                            "var_lev_P":list(lev_p),
                            "final_P":f_p_list,
                            "FDR":list(fdr),
        }, index=df1.index)
        res_df2 = pd.concat([res_df,
                    mean_fold_change, median_fold_change, data1_mean, data2_mean,
                    data1_median, data2_median], axis=1)
        res_df2.columns = [f"{df1_group}_norm_P",f"{df2_group}_norm_P","var_lev_P","final_P","FDR",
        f"mean_FDC({df2_group}/{df1_group})", f"median_FDC({df2_group}/{df1_group})",f"{df1_group}_mean",f"{df2_group}_mean",
        f"{df1_group}_median",f"{df2_group}_median"]
        return res_df2

    def difference_stat(self, outfile):
        gps = sorted(self.y.unique())
        gp0, gp1 = gps[0], gps[1]
        df0, df1 = self.X[self.y==gp0].T, self.X[self.y==gp1].T
        res = MetaboliteRawFilter.difference_testing(df1=df0,df2=df1, df1_group=str(gp0), df2_group=str(gp1), pair=self.pair)
        self.diff_stats = res
        res.to_csv(outfile, sep="\t", index="Index")

    def difference_filter(self, p_cutoff, adjust_p_cutoff, fold_change_cutoff)->pd.DataFrame:
        # 目前仅支持两组比较
        self.p_cutoff = p_cutoff
        self.adjust_p_cutoff = adjust_p_cutoff
        self.fold_change_cutoff = fold_change_cutoff
        gps = sorted(self.y.unique())
        gp0, gp1 = gps[0], gps[1]
        df0, df1 = self.X[self.y==gp0].T, self.X[self.y==gp1].T
        if self.diff_stats is None:
            res = MetaboliteRawFilter.difference_testing(df1=df0,df2=df1, df1_group=str(gp0), df2_group=str(gp1), pair=self.pair)
            self.diff_stats = res
        else:
            res = self.diff_stats
        filtered_X = self.X.T[(res["final_P"]<self.p_cutoff)&
                            (res["FDR"]<self.adjust_p_cutoff)&
                            ((res[f"mean_FDC({gp1}/{gp0})"]>self.fold_change_cutoff)|
                            (res[f"mean_FDC({gp1}/{gp0})"]<1/self.fold_change_cutoff))].T
        filtered_X.index.rename("Index", inplace=True)
        print("difference filter done")
        return filtered_X


class DelongTest():
    def __init__(self,preds1,preds2,label,threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1=preds1
        self._preds2=preds2
        self._label=label
        self.threshold=threshold
        

    def _auc(self,X, Y)->float:
        return 1/(len(X)*len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self,X, Y)->float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y==X else int(Y < X)

    def _structural_components(self,X, Y)->list:
        V10 = [1/len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self,V_A, V_B, auc_A, auc_B)->float:
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
    
    def _z_score(self,var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB )**(.5)+ 1e-8)

    def _group_preds_by_label(self,preds, actual)->list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)+ self._get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = stats.norm.sf(abs(z))*2

        return z,p

    def show_result(self):
        z,p=self._compute_z_p()
        return p

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, f_regression,mutual_info_regression
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
        :param method: 字符串，指定使用的方法（'chi2', 'mutual_info_classif', 'f_classif','f_regression', 'pearson', 'logistic', 'mutual_info_regression'更多参考https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html）。
        :param k: 选择的特征数量，'all' 表示选择所有特征。
        :return: 选择的特征 DataFrame。
        """
        if method in ['chi2', 'mutual_info_classif', 'f_classif', 'f_regression', 'mutual_info_regression']:
            if method == 'chi2':
                selector = SelectKBest(chi2, k=k)
            elif method == 'mutual_info_classif':
                selector = SelectKBest(mutual_info_classif, k=k)
            elif method == 'f_classif':
                selector = SelectKBest(f_classif, k=k)
            elif method == 'f_regression':
                selector = SelectKBest(f_regression, k=k)
            elif method == 'mutual_info_regression':
                selector = SelectKBest(mutual_info_regression, k=k)
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
