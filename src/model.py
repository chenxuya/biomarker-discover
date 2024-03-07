import pandas as pd
import numpy as np
from os.path import join
import os
from .utils.str_param import CommonParam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.utils import shuffle
# from imblearn.over_sampling import SMOTE
from scipy import stats
from .custom_model import CustomLogModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import svm  # SVM算法
from sklearn.linear_model import Lasso, LogisticRegression
import xgboost as xgb
import pickle
from .utils.diff_test import DelongTest
# Python code for model

class Modeling():
    def __init__(self,x_train, y_train, k_fold, scoring, 
        random_seed, opti_method, njobs, lasso_times) -> None:
        # cfgs = Param(cfg_path=cfg_path)
        self.x_train = x_train
        self.y_train = y_train
        self.k_fold = k_fold
        self.scoring = scoring
        self.random_seed = random_seed
        self.opti_method = opti_method
        self.n_jobs = njobs
        self.num_class = len(y_train.unique())
        self.lasso_times = lasso_times


    def calculate_scale_pos_weight(self):
        """
        计算二分类问题中的scale_pos_weight。
        """
        positive_count = sum(self.y_train)
        negative_count = len(self.y_train) - positive_count
        scale_pos_weight = negative_count / positive_count
        return scale_pos_weight

    def calculate_sample_weights(self):
        """
        为多分类问题计算样本权重。
        """
        class_weights = self.y_train.value_counts().max() / self.y_train.value_counts()
        return self.y_train.map(class_weights)

    def select_basic_model(self, method):
        if method.upper() == "SVM":
            return svm.SVC(random_state=self.random_seed, probability=True, kernel="linear", class_weight="balanced")
        elif method.upper() =="RFC": # RandomForestClassifier
            return RandomForestClassifier(random_state=self.random_seed,class_weight="balanced_subsample")
        elif method.upper() == "LASSO":
            if self.lasso_times >1:
                return CustomLogModel(self.lasso_times, "LASSO") # 运行lasso1000次，取结果的均值
            else:
                if self.num_class >2:
                    return LogisticRegression(random_state=self.random_seed, solver="saga", penalty="l1", max_iter=1000, multi_class='multinomial',class_weight="balanced")
                else:
                    return LogisticRegression(random_state=self.random_seed, solver="liblinear", penalty="l1", max_iter=1000, class_weight="balanced")
        elif method.upper() =="LR": # logistic regression
            if self.num_class > 2:
                return LogisticRegression(solver="saga", random_state=self.random_seed, max_iter=1000, multi_class='multinomial', class_weight="balanced")
            else:
                return LogisticRegression(solver="saga", random_state=self.random_seed, max_iter=1000, class_weight="balanced")
        elif method.upper() == "ELASTICNET":
            if self.lasso_times >1:
                return CustomLogModel(self.lasso_times, "ELASTICNET") # 运行lasso1000次，取结果的均值
            else:
                return LogisticRegression(penalty="elasticnet",solver="saga", random_state=self.random_seed, max_iter=1000)
        elif method.upper() == "XGBOOST":
            if self.num_class >2:
                return xgb.XGBClassifier(random_state=self.random_seed)
            else:
                return xgb.XGBClassifier(random_state=self.random_seed, scale_pos_weight=self.calculate_scale_pos_weight())
        elif method.upper() == "DNN_FNN":
            dnn = Classifier(self.x_train.shape[1])
            return BasicDnn(dnn)
        else:
            raise ValueError("invalid model method, check model method in config file")

    def get_grid_params(self, method):
        if method =="SVM":
            param_grid = {# 'C': [0.001,0.025,0.05,0.1, 1, 10, 100, 1000], 
                            'C': [0.001,0.01,0.1, 1], 
                            # 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                            # 'kernel': ['linear']
                            }
        elif method == "RFC":
            param_grid = {'n_estimators': [20,50,100,200],
                            'max_features': ['sqrt'],
                            'max_depth' : [4,5,6,7,8],
                            'criterion' :['gini', 'entropy']
                            }
        elif method == "LASSO":
            if self.lasso_times>1:
                param_grid = {}
            else:
                param_grid = {"C":np.logspace(-3,3,7)}
        elif method.upper() in ["CUSTOMLASSO", "CUSTOMLR", "CUSTOMELASTICNET"]:
            param_grid = {}
        elif method == "LR":
            param_grid = {"C":np.logspace(-3,3,7), "penalty":["l2"]}# l1 lasso l2 ridge
        elif method == "ELASTICNET":
            param_grid = {"C":np.logspace(-3,3,7),
                            "l1_ratio":np.linspace(0, 1, 20)}
        elif method =="XGBOOST":
            param_grid = {
                        'max_depth': [3, 4, 5],
                        'learning_rate': [0.1, 0.01, 0.001],
                        'n_estimators': [50, 100, 200],
                        "lambda":[0.1,1, 10, 100],
                        "alpha":[0.1,1, 10, 100],
                    }
        else:
            raise ValueError("invalid model method, check model method in config file")
        return param_grid
    
    def get_bayes_params(self, method):
        if method == "SVM":
            param_space = {'C': Real(1e-6, 1e+6, prior='log-uniform'),
                            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                            'degree': Integer(1,8),
                            'kernel': Categorical(['linear']),
                            "probability":Categorical([True])
                            }
        elif method == "RFC":
            param_space = {'n_estimators': Integer(2,500),
                            'max_features': Categorical(['sqrt', "log2", None]),
                            'max_depth' : Integer(2,8),
                            'criterion' :Categorical(['gini', 'entropy'])
                            }
        elif method == "LASSO":
            param_space = {"C":Real(1e-6, 1e+6, prior='log-uniform')}

        elif method == "LR":
            param_space = {"C": Real(1e-6, 1e+6, prior='log-uniform'),
                             "penalty":Categorical(["l2"])}# l1 lasso l2 ridge
        elif method == "ELASTICNET":
            param_space = {"C":Real(1e-6, 1e+6, prior='log-uniform'),
                            "l1_ratio":Real(0,1)}
        else:
            raise ValueError("invalid model method, check model method in config file")
        return param_space

    def get_scoring_method(self):
        # 交叉验证时的评价指标
        if self.scoring == "roc_auc":
            if len(self.y_train.unique())==2:
                metric = "roc_auc"
            else:
                raise ValueError("roc_auc score only support binary classification, for multi-class classification, please use other metrics")
        elif self.scoring == "accuracy":
            metric = "accuracy"
        elif self.scoring =="f1":
            if len(self.y_train.unique())==2:
                metric = "f1"
            else:
                raise ValueError("f1 score only support binary classification, for multi-class classification, please use [f1_micro, f1_macro, f1_weighted]")
        elif self.scoring == "f1_micro":
            metric = "f1_micro"
        elif self.scoring == "f1_macro":
            metric = "f1_macro"
        elif self.scoring == "recall":
            if len(self.y_train.unique())==2:
                metric = "recall"
            else:
                raise ValueError("recall score only support binary classification, for multi-class classification, please use [recall_micro, recall_macro, recall_weighted]")
        elif self.scoring == "recall_micro":
            metric = "recall_micro"
        elif self.scoring == "recall_macro":
            metric = "recall_macro"
        elif self.scoring == "precision_micro":
            metric = "precision_micro"
        elif self.scoring == "precision_macro":
            metric = "precision_macro"
        elif self.scoring == "precision_weighted":
            metric = "precision_weighted"

        elif self.scoring == "precision":
            if len(self.y_train.unique()) == 2:
                metric = "precision"
            else:
                raise ValueError("precision score only support binary classification, for multi-class classification, please use [precision_micro, precision_macro, precision_weighted]")
        elif self.scoring == "balanced_accuracy":
            # 平衡准确率适用于处理类别不平衡的情况
            metric = "balanced_accuracy"
        elif self.scoring == "neg_log_loss":
            # 适用于概率估计的负对数损失
            metric = "neg_log_loss"
        elif self.scoring == "average_precision":
            # 平均精确率适用于二分类和多标签问题
            metric = "average_precision"

        else:
            raise ValueError("invalid scoring method, check scoring method in config file")
        return metric

    def construct(self, method)->dict:
        basic_model = self.select_basic_model(method=method)
        scoring_method = self.get_scoring_method()
        if self.opti_method in ["grid", "bayes"]:
            if self.lasso_times >1 and (method in ["LASSO", "LR", "ELASTICNET"]):
                basic_model.fit(self.x_train, self.y_train)
            elif "DNN" in method:
                basic_model.fit(self.x_train, self.y_train)
            else:
                if self.opti_method =="grid":
                    param_grid = self.get_grid_params(method=method)
                    optimizer = GridSearchCV(basic_model, param_grid, refit=True, 
                                    verbose=0, cv=int(self.k_fold), scoring=scoring_method, n_jobs=self.n_jobs)
                elif self.opti_method =="bayes":
                    param_space = self.get_bayes_params(method=method)
                    optimizer = BayesSearchCV(basic_model,param_space, n_iter=100, scoring=scoring_method, n_jobs=self.n_jobs)
                try:
                    optimizer.fit(self.x_train, self.y_train)
                    print("tuned hpyerparameters :(best parameters) ",optimizer.best_params_)
                    basic_model = optimizer.best_estimator_
                except ValueError:
                    print("k fold Gridsearch failed,it could be caused by sample unbalance")
                    basic_model.fit(self.x_train, self.y_train)
        else:
            basic_model.fit(self.x_train, self.y_train)
        return basic_model
    
    def save_model(self,model, outfile_without_suffix):
        save_dir = os.path.dirname(outfile_without_suffix)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        model_param = model.get_params(deep=True)
        model_param_df = pd.DataFrame(model_param, index=["best"]).T
        model_param_df.to_csv(f"{outfile_without_suffix}.txt", sep="\t", index_label="HyperParam")
        pickle.dump(model, open(join(save_dir, f"{outfile_without_suffix}.pkl"),"wb"))

    def load_model(self, model_pick):
        if os.path.exists(model_pick):
            model = pickle.load(open(model_pick, "rb"))
        else:
            raise ValueError("Please train model first.")
        return model


class Testing():
    def __init__(self,test_x, test_y, model, method):
        self.test_x = test_x
        self.test_y = test_y
        self.model = model
        self.method = method
        self.predict = None
        self.probality = None
        self.fpr, self.tpr, self.thresholds = None, None, None
        self.roc_auc = None
        self.macro_roc_auc = None
        self.auc_ci_left = None
        self.auc_ci_right = None
        self.auc_pvalue = None
        self.f1_score, self.precision, self.recall, self.specificity = None, None, None, None
        self.macro_f1_score, self.macro_precision, self.macro_recall, self.macro_specificity = None, None, None, None
        self.confusion_matrix = None
        self.var_importance = None
        self.feature_name = None
        self.sparse_coef = None
        self.rocDisplay = None
        self.confusion_matrix_display = None
        self.alphas = None
        self.accuracy = None
        self.brier_score = None
        self.optimal_threshold = None
        self.common_param = CommonParam()

    def bootAUC(self, true_value, probability):
        n_bootstraps = 1000
        rng_seed = 42
        bootstrapped_scores = []

        rng = np.random.default_rng(rng_seed)
        i = 0
        while i < n_bootstraps:
            if i==100000:exit("all labels seems to be the same")
            indices = rng.integers(0, len(probability), len(probability))
            if len(np.unique(true_value[indices]))==1:continue # 如果抽到同样的标签，直接进入下一个循环。
            score = metrics.roc_auc_score(true_value[indices], probability[indices])
            bootstrapped_scores.append(score)
            i += 1
        auc_confidence_interval = [np.percentile(bootstrapped_scores, 5), np.percentile(bootstrapped_scores, 95)]
        return auc_confidence_interval[0], auc_confidence_interval[1]


    def test(self, optimal_threshold=None):
        # 二分类测试
        self.probality = self.model.predict_proba(self.test_x)
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.test_y, self.probality[:,1])
        self.roc_auc = metrics.auc(self.fpr, self.tpr)
        self.rocDisplay = metrics.RocCurveDisplay(fpr=self.fpr, tpr=self.tpr, roc_auc=self.roc_auc, estimator_name = self.method)
        
        self.auc_ci_left, self.auc_ci_right = self.bootAUC(self.test_y, self.probality[:,1])
        np.random.seed(123)
        try:
            self.auc_pvalue = DelongTest(np.random.rand(len(self.test_y)), self.probality[:,1], self.test_y).show_result()
        except ZeroDivisionError:
            self.auc_pvalue = 1.
        except Exception as e:
            print(e)
        # Calculate Youden's J statistic for each threshold
        youden_j = self.tpr - self.fpr
        # Find the optimal threshold that maximizes Youden's J statistic
        optimal_idx = np.argmax(youden_j)
        if optimal_threshold is None:
            self.optimal_threshold = self.thresholds[optimal_idx]
        else:
            self.optimal_threshold = optimal_threshold
        # determine the predict label according optimal threshold probability
        self.predict = (self.probality[:,1] >=self.optimal_threshold).astype(int)
        self.precision = metrics.precision_score(self.test_y, self.predict)
        self.recall = metrics.recall_score(self.test_y, self.predict)
        self.f1_score = metrics.f1_score(self.test_y, self.predict)
        self.accuracy = metrics.accuracy_score(self.test_y, self.predict)
        self.confusion_matrix = metrics.confusion_matrix(self.test_y, self.predict)
        TP = self.confusion_matrix[1, 1]
        TN = self.confusion_matrix[0, 0]
        FP = self.confusion_matrix[0, 1]
        FN = self.confusion_matrix[1, 0]
        self.specificity = TN / float(TN+FP)
        self.brier_score = metrics.brier_score_loss(self.test_y, self.probality[:,1])

        self.confusion_matrix_display = metrics.ConfusionMatrixDisplay(self.confusion_matrix)
        self.feature_name = self.model.feature_names_in_
        if self.method =="SVM":
            params_dict = self.model.get_params(deep=True)
            if params_dict["kernel"] =="linear":
                self.var_importance = np.abs(self.model.coef_)
        if self.method in ["LR", "LASSO"]:
            self.var_importance = np.abs(self.model.coef_)
        if self.method in ["RFC", "XGBOOST"] or ("DNN" in self.method):
            self.var_importance = self.model.feature_importances_

    def multi_class_test(self):
        def _macro_average_specificity(confusion_matrix):
            # 提取矩阵大小，即类别数量
            num_classes = confusion_matrix.shape[0]
            
            # 计算每个类别的TN和FP
            TN = np.zeros(num_classes)
            FP = np.zeros(num_classes)
            for i in range(num_classes):
                TN[i] = np.sum(confusion_matrix) - np.sum(confusion_matrix[i, :]) - np.sum(confusion_matrix[:, i]) + confusion_matrix[i, i]
                FP[i] = np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
            
            # 计算每个类别的特异性
            specificity_per_class = TN / (TN + FP)
            
            # 计算宏平均特异性
            macro_specificity = np.nanmean(specificity_per_class)
            return macro_specificity

        def _calculate_micro_average_specificity(conf_matrix):
            """
            Correctly calculate micro-average specificity.
            
            Args:
            conf_matrix -- Confusion matrix, a matrix for each category

            Returns:
            correct_micro_avg_specificity -- Correctly calculated micro-average specificity
            """
            # Total number of instances
            total_instances = np.sum(conf_matrix)

            # Calculate TN and FP for each class
            TN = []
            FP = []
            num_classes = conf_matrix.shape[0]
            for i in range(num_classes):
                # TN: Sum all the values excluding the current row and column
                TN_i = total_instances - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
                TN.append(TN_i)

                # FP: Sum of the current column excluding the diagonal (current class)
                FP_i = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
                FP.append(FP_i)

            # Calculate micro-average specificity
            correct_micro_avg_specificity = np.sum(TN) / (np.sum(TN) + np.sum(FP))
            return correct_micro_avg_specificity


        mirco = "micro"
        macro = "macro"
        # 多分类的测试
        self.predict = self.model.predict(self.test_x)
        self.probality = self.model.predict_proba(self.test_x)
        test_y_bin = LabelBinarizer().fit_transform(self.test_y)
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(test_y_bin.ravel(), self.probality.ravel()) # 微平均
        self.auc_ci_left, self.auc_ci_right = self.bootAUC(test_y_bin.ravel(), self.probality.ravel())
        self.auc_pvalue = DelongTest(np.random.rand(len(test_y_bin)), self.probality.ravel(), test_y_bin.ravel()).show_result()

        self.roc_auc = metrics.auc(self.fpr, self.tpr)
        self.macro_roc_auc = metrics.roc_auc_score(test_y_bin, self.probality, average=macro)  #宏平均

        self.rocDisplay = metrics.RocCurveDisplay(fpr=self.fpr, tpr=self.tpr, roc_auc=self.roc_auc, estimator_name = method)
        self.precision = metrics.precision_score(self.test_y, self.predict, average=mirco)
        self.recall = metrics.recall_score(self.test_y, self.predict, average=mirco)
        self.f1_score = metrics.f1_score(self.test_y, self.predict, average=mirco)
        self.accuracy = metrics.accuracy_score(self.test_y, self.predict)
        # 宏平均
        self.macro_precision = metrics.precision_score(self.test_y, self.predict, average=macro)
        self.macro_recall = metrics.recall_score(self.test_y, self.predict, average=macro)
        self.macro_f1_score = metrics.f1_score(self.test_y, self.predict, average=macro)


        self.confusion_matrix = metrics.confusion_matrix(self.test_y, self.predict)
        self.optimal_threshold = np.nan

        self.specificity = _calculate_micro_average_specificity(self.confusion_matrix) # 微平均
        self.macro_specificity = _macro_average_specificity(self.confusion_matrix) # 宏平均
        self.confusion_matrix_display = metrics.ConfusionMatrixDisplay(self.confusion_matrix)

        self.feature_name = self.model.feature_names_in_
        if method =="SVM":
            params_dict = self.model.get_params(deep=True)
            if params_dict["kernel"] =="linear":
                self.var_importance = np.abs(self.model.coef_)
        if method in ["LR", "LASSO"]:
            self.var_importance = np.abs(self.model.coef_)
        if method in ["RFC", "XGBOOST"] or ('DNN' in method.upper()):
            self.var_importance = self.model.feature_importances_


    def output(self, outdir, split, prefix):
        split = f"{self.common_param.split}{split}"
        def _makedir(dir):
            if not os.path.exists(dir): os.makedirs(dir)
        if not os.path.exists(outdir): os.makedirs(outdir)
        _makedir(join(outdir, self.common_param.roc))
        _makedir(join(outdir, self.common_param.confusion))
        _makedir(join(outdir, self.common_param.metric))
        _makedir(join(outdir, self.common_param.importance))
        y_label = sorted(self.test_y.unique())
        res1_dict = dict()
        res1_dict["y_True"] = self.test_y
        res1_dict["y_Predict"] = self.predict
        for i in range(len(y_label)):
            res1_dict[f"{y_label[i]}_Probability"] = self.probality[:,i]
        # res1 = pd.DataFrame({"y_True":self.test_y,"1_Probability":self.probality[:,1],"0_Probability":self.probality[:,0], "y_Predict":self.predict}, index=self.test_x.index)
        res1 = pd.DataFrame(res1_dict, index=self.test_x.index)
        res1.to_csv(join(outdir, self.common_param.roc,f"{prefix}_{split}_{self.method}_TrueAndPredict.txt"), sep="\t")
        ## 
        res2 = pd.DataFrame({self.common_param.fpr:self.fpr, self.common_param.tpr:self.tpr, self.common_param.threshold:self.thresholds})
        res2.to_csv(join(outdir, self.common_param.roc,f"{prefix}_{split}_{self.method}_roc_curve.txt"), sep="\t")
        ##
        res3 = pd.DataFrame(self.confusion_matrix, index=[["True label"]*len(y_label),y_label], columns=[["Predicted label"]*len(y_label),y_label])
        res3.to_csv(join(outdir, self.common_param.confusion,f"{prefix}_{split}_{self.method}_confusion_matrix.txt"), sep="\t")
        ##二分类
        if self.confusion_matrix.shape[0]==2:
            com_metrics = [self.roc_auc, self.auc_ci_left, self.auc_ci_right, self.auc_pvalue,self.accuracy,self.f1_score, self.precision, self.recall, self.specificity,self.brier_score, self.optimal_threshold]
            com_metrics_name = [self.common_param.auc,"AUC_CI_left", "AUC_CI_right", "AUC_pvalue", "Accuracy","F1_socre", "Precision", "Recall(Sensitivity)","Specificity","Brier_score", "Optimal_threshold"]
        else:
            com_metrics = [self.roc_auc, self.auc_ci_left, self.auc_ci_right, self.auc_pvalue,self.accuracy,self.f1_score, self.precision, self.recall, self.specificity, self.optimal_threshold,
                            self.macro_roc_auc, self.macro_f1_score, self.macro_precision, self.macro_recall, self.macro_specificity]
            com_metrics_name = [self.common_param.auc,"AUC_CI_left", "AUC_CI_right", "AUC_pvalue", "Accuracy","F1_socre", "Precision", "Recall(Sensitivity)","Specificity", "Optimal_threshold",
                            "macro_AUC", "macro_F1_socre", "macro_Precision", "macro_Recall(Sensitivity)","macro_Specificity"]
        res4 = pd.DataFrame({"Value":com_metrics}, index=com_metrics_name)
        res4.to_csv(join(outdir, self.common_param.metric,f"{prefix}_{split}_{self.method}_metrics.txt"), sep="\t", index_label=self.common_param.metric)
        ## 
        if self.var_importance is not None:
            res5 = pd.DataFrame(self.var_importance.squeeze(), index=self.feature_name)
            res5.to_csv(join(outdir, self.common_param.importance,f"{split}_{self.method}_variableImportance.txt"), sep="\t")
        fig, ax = plt.subplots(figsize=(10,10),dpi=100)
        self.rocDisplay.plot(ax=ax, lw=2, color="crimson")
        ax.plot([0, 1], [0, 1], '--', lw=2, color = 'grey')
        ax.axis('square')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        fig.savefig(join(outdir,self.common_param.roc, f"{prefix}_{split}_{self.method}_roc_curve.jpg"))
        fig.savefig(join(outdir,self.common_param.roc, f"{prefix}_{split}_{self.method}_roc_curve.pdf"))
        plt.close(fig=fig)
        fig2, ax2 = plt.subplots(figsize=(10,10), dpi=100)
        self.confusion_matrix_display.plot(ax=ax2)
        ax2.set_xticklabels(sorted(self.test_y.unique(), reverse=False))
        ax2.set_yticklabels(sorted(self.test_y.unique(), reverse=False))
        fig2.savefig(join(outdir, self.common_param.confusion, f"{prefix}_{split}_{self.method}_confusion_matrix.jpg"))
        fig2.savefig(join(outdir, self.common_param.confusion, f"{prefix}_{split}_{self.method}_confusion_matrix.pdf"))
        plt.close(fig=fig2)
        return res1, res2, res3, res4, res5

