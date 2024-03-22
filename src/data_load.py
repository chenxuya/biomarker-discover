import pandas as pd
import configparser
from typing import Tuple
from src.utils.str_param import CommonParam
class Param():
    def __init__(self, cfg_path) -> None:
        self.str_param = CommonParam()
        all_par = self.loadCfg(cfg_path)
        self.datafile = all_par["datafile"]
        self.info_file = all_par["info_file"]
        self.test_data_file = all_par["test_data_file"]
        self.marker_file = all_par["marker_file"]
        self.splited_data_dir = all_par["splited_data_dir"]
        self.outdir = all_par["outdir"]
        self.compare = all_par["compare"]
        self.transform = all_par["transform"]
        self.scale = all_par["scale"]
        self.method = all_par["method"].upper().strip(";").strip()
        self.rfe = all_par["rfe"].upper()
        self.shap = all_par["shap"].upper()
        self.lasso_times = int(all_par["lasso_times"])
        self.k_fold = int(all_par["k_fold"])
        self.opti_method = all_par["opti_method"]
        self.scoring = all_par["scoring"]
        self.train_size = float(all_par["train_size"])
        self.split_times = int(all_par["split_times"])
        self.random_seed = int(all_par["random_seed"])
        self.category = all_par["category"].strip(";")
        self.norminal = all_par["norminal"].strip(";")
        self.norminal_scale = all_par["norminal_scale"]
        self.ordinal = all_par["ordinal"].strip(";")
        self.n_jobs = int(all_par["n_jobs"])
        self.refit = all_par["refit"].upper()
        self.min_features_to_select = int(all_par["min_features_to_select"])
        self.p_cutoff = float(all_par["p_value"])
        self.adjust_p_cutoff = float(all_par["adjusted_p"])
        self.fold_change = float(all_par["fold_change"])
        self.check()

    def loadCfg(self, cfg_path):
        conf = configparser.ConfigParser()
        conf.read(cfg_path, encoding="utf-8")
        file_par = dict(conf.items("file"))
        compare_par = dict(conf.items("compare"))
        preprocess_par = dict(conf.items("preprocess"))
        model_par = dict(conf.items("model"))
        feature_select_par = dict(conf.items("feature_select"))
        all_par = {**file_par, **compare_par, 
                    **preprocess_par, **model_par,
                    **feature_select_par}
        return all_par


    def check(self):
        for m in self.method.split(self.str_param.semi_sep):
            if m not in ["SVM", "RFC", "LASSO", "LR","ELASTICNET","XGBOOST", "DNN_FNN"]:
                raise ValueError(f"Expected [SVM, RFC, LASSO, LR, DNN_FNN], but got {m}")
        if self.transform not in ["no", "log2"]:
            raise ValueError(f"Expected [no log2], but got {self.transform}")
        if self.scale not in ["min-max", "z-score", "no"]:
            raise ValueError(f"Expected [min-max, z-score, no], but got {self.scale}")
        if self.scoring not in ["roc_auc", "accuracy", "f1", "f1_micro", "f1_macro", "f1_weighted", "recall", "recall_micro", "recall_macro", "recall_weighted", "precision", "precision_micro", "precision_macro", "precision_weighted"]:
            raise ValueError(f"Expected [roc_auc, accuracy, f1, f1_micro, f1_macro, f1_weighted, recall, recall_micro, recall_macro, recall_weighted, precision, precision_micro, precision_macro, precision_weighted], but got {self.scoring}")
        if self.train_size<0.5 or self.train_size>1:
            raise ValueError(f"Expected train_size between [0.5,1], but got {self.train_size}")
        if self.opti_method not in ["grid", "bayes", "no"]:
            raise ValueError(f"Expected [grid, bayes, no], but got {self.opti_method}")
        if self.refit not in ["TRUE", "FALSE"]:
            raise ValueError(f"Expected [TRUE, FALSE], but got {self.refit}")
        if self.refit == "TRUE":
            self.refit = True
        else:
            self.refit = False
        if self.rfe not in ["TRUE", "FALSE"]:
            raise ValueError(f"Expected [TRUE, FALSE], but got {self.rfe}")
        if self.rfe == "TRUE":
            self.rfe = True
        else:
            self.rfe = False
        if self.shap not in ["TRUE", "FALSE"]:
            raise ValueError(f"Expected [TRUE, FALSE], but got {self.shap}")
        if self.shap == "TRUE":
            self.shap = True
        else:
            self.shap = False

def checkColumn(df:pd.DataFrame, tcolums:list, datapath:str):
    if not isinstance(tcolums, list):
        raise TypeError(f"{tcolums} must be list, but get {type(tcolums)}")
    for c in tcolums:
        if c not in df.columns:
            raise ValueError(f"column {c} not found in {datapath}({tcolums})")

class DataLoader():
    def __init__(self, Meta_data_path:str, info_file:str, groups:str, category:list=None, norminal:list=None) -> None:
        self.Meta_data_path = Meta_data_path
        self.info_path = info_file
        self.category = category
        self.norminal = norminal
        self.info = None
        self.meta_data = None
        self.val_sample = None
        self.groups = groups
        self.str_param = CommonParam()
    

    def loadMeta(self)->pd.DataFrame:
        data = read_file(self.Meta_data_path, index_col=0)
        if self.info is None:
            self.info = self.loadInfo()
        self.val_sample = data.columns.intersection(self.info[self.str_param.sample])
        self.meta_data = data.loc[:, self.val_sample]
        return self.meta_data # 行为代谢物，列为样本


    def loadData(self)->pd.DataFrame:
        if self.info is None:
            self.info = self.loadInfo()
        if self.meta_data is None:
            self.meta_data = self.loadMeta()
        data = self.meta_data
        if self.norminal is not None:
            normianl_df = pd.DataFrame(self.info.loc[self.val_sample,self.norminal]).T
            data = pd.concat([data, normianl_df], axis=0)
        if self.category is not None:
            category_df = pd.DataFrame(self.info.loc[self.val_sample, self.category]).T
            data = pd.concat([data, category_df], axis=0)
        return data # data 行为代谢物或其他临床指标，列为样本。

    def loadInfo(self)->pd.DataFrame:
        info = read_file(self.info_path)
        info = info.astype(str)
        info_col = [self.str_param.sample, self.str_param.group]
        gps = self.groups.split(self.str_param.vs_sep)
        info = info[info[self.str_param.group].isin(gps)]
        info[self.str_param.sample] = info[self.str_param.sample].astype(str)
        if self.category is not None:
            info_col = info_col + self.category 
        if self.norminal is not None:
            info_col = info_col + self.norminal
        checkColumn(info, info_col, self.info_path)
        info.index = info[self.str_param.sample]
        return info

def read_file(file_path:str, index_col:int=None)->pd.DataFrame:
    """
    Read a file and return its content as a pandas DataFrame.

    Args:
        file_path (str): The path to the file to be read.
        index_col (int, optional): The column to be used as the row labels of the DataFrame.

    Returns:
        pd.DataFrame: The content of the file as a pandas DataFrame.
    """
    if file_path.endswith(".txt") or file_path.endswith('.xls'):
        data = pd.read_csv(file_path, sep="\t", index_col=index_col)
    elif file_path.endswith(".csv"):
        data = pd.read_csv(file_path, sep=",", index_col=index_col)
    elif file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path, index_col=index_col, header=0)
    else:
        raise ValueError(f"file_path must be .txt(sep by tab) or .csv(sep by comma) or .xlsx or .xls(sep by tab), but get {file_path}")
    return data