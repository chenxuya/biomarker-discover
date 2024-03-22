import pandas as pd
import os
from os.path import join
import glob
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from str_param import CommonParam
from plot_frame import plot_roc,plot_horizontal_barplot
from ..feature_evaluate import FeatureSelectorIndividual
from ..data_load import read_file
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy import interpolate
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, binarize


def add_to_dict(dictionary, model_method, obj):
    if model_method in dictionary:
        dictionary[model_method].append(obj)
    else:
        dictionary[model_method] = [obj]

def shape_tprfpr(tpr_fpr_df_ali):
    fpr_list, tpr_list,thr_list = [], [],[]
    for roc in tpr_fpr_df_ali:
        fpr_list.append(roc[CommonParam.fpr])
        tpr_list.append(roc[CommonParam.tpr])
        thr_list.append(roc[CommonParam.threshold])
    cols = [f"{CommonParam.split}{i}" for i in range(1, len(tpr_fpr_df_ali)+1)]
    fpr, tpr, thr = pd.concat(fpr_list, axis=1), pd.concat(tpr_list, axis=1), pd.concat(thr_list, axis=1)
    fpr.columns = cols
    tpr.columns = cols
    thr.columns = cols
    return fpr, tpr, thr


def shape_metric(metric_ali, sort=False):
    """
    Calculate a shape metric based on the input metrics. 

    Args:
        metric_ali (list): List of input metrics.
        sort (bool, optional): Whether to sort the resulting dataframe. Defaults to False.

    Returns:
        pandas.DataFrame: Dataframe containing the calculated shape metric.
    """
    df = pd.concat(metric_ali, axis=1)
    cols = [f"{CommonParam.split}{i}" for i in range(1, len(metric_ali)+1)]
    df.columns = cols
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    df.insert(0, CommonParam.mean, mean)
    df.insert(1, CommonParam.std, std)
    if sort:
        df.sort_values(by=[CommonParam.mean, CommonParam.std], ascending=[False, True], inplace=True)
    return df

def shape_metric_dict(dict4):
    def _shape_dict_value(ali):
        df = pd.concat(ali, axis=1)
        cols = [f"{CommonParam.split}{i}" for i in range(1, len(ali)+1)]
        df.columns = cols
        df = df.T
        df.index.name = CommonParam.dataset
        df = df.reset_index()
        return df
    dataframes_dict = {k:_shape_dict_value(v) for k,v in dict4.items()}
    combined_dataframe = pd.concat(dataframes_dict, axis=0).reset_index(level=0)
    combined_dataframe = combined_dataframe.rename(columns={'level_0': CommonParam.method})
    return combined_dataframe

def interaction(df:pd.DataFrame):
    """
    df 每一列为一个模型赛选的的marker
    该函数用来对每一列两两取交集。
    """
    # df 每一列为一个模型赛选的的marker
    # 该函数用来对每一列两两取交集。
    cols = df.columns
    all_combination = []
    for i in range(2, len(cols)+1):
        all_combination.extend(list(combinations(cols, i)))
    if not all_combination:return df # 如果只有一个模型，则没有交集，直接返回。
    all_combination = list(dict.fromkeys(all_combination)) # 列表去除，同时尽量保留原始顺序
    inter_df_list = []
    for combination in all_combination:
        inter_col = "+".join(list(combination)) + " intersection"
        ali = []
        for col in combination:
            ali.append(df[col][df[col].notna()].to_list())
        inter_v = list(set(ali[0]).intersection(*ali[1:]))
        inter_df = pd.DataFrame({inter_col:inter_v})
        inter_df_list.append(inter_df)
    inter_df_res = pd.concat(inter_df_list, axis=1)
    return pd.concat([df, inter_df_res], axis=1)


def weight_score(model_metrics_df:pd.DataFrame, model_varImp_df:pd.DataFrame, weight_metric:str)->pd.Series:
    """
    Calculate the weighted feature importance score based on the given model metrics and variable importance dataframes.

    Parameters:
        model_metrics_df (pd.DataFrame): The dataframe containing model metrics.
        model_varImp_df (pd.DataFrame): The dataframe containing variable importance scores.
        weight_metric (str): The specific metric to be used for weighting.

    Returns:
        pd.Series: The weighted feature importance scores.
    """
    model_metrics_df = model_metrics_df.loc[weight_metric,:]
    weighted_model_metric = (model_metrics_df-0.5)/((model_metrics_df-0.5).values.sum())
    model_varImp_df = model_varImp_df[weighted_model_metric.index]
    minmax_model_varImp_df = pd.DataFrame(MinMaxScaler().fit_transform(model_varImp_df),
                            index=model_varImp_df.index, columns=model_varImp_df.columns)
    weight_varImp_df = minmax_model_varImp_df * weighted_model_metric.values
    weighted_varImp = weight_varImp_df.sum(axis=1)
    weighted_varImp.rename("weighted_featureImportance", inplace=True)
    weighted_varImp.sort_values(ascending=False, inplace=True)
    return weighted_varImp

def batch_out(out_without_suffix,model_method, train_dict2, train_dict4):
    fpr, tpr, thr = shape_tprfpr(train_dict2[model_method])
    metric = shape_metric(train_dict4[model_method])
    roc_fig = plot_roc(tpr, fpr)
    fpr.to_csv(out_without_suffix+f"_{CommonParam.fpr}.txt", sep="\t", index=False)
    tpr.to_csv(out_without_suffix+f"_{CommonParam.tpr}.txt", sep="\t", index=False)
    thr.to_csv(out_without_suffix+f"_{CommonParam.threshold}.txt", sep="\t", index=False)
    metric.to_csv(out_without_suffix+f"_{CommonParam.metric}.txt", sep="\t", index=True, index_label=CommonParam.metric)
    roc_fig.savefig(out_without_suffix+f"_{CommonParam.roc}.jpg")
    return fpr, tpr, thr,metric, roc_fig


def batch_imp(outdir, model_method, train_dict5, evaluation_method=CommonParam.model):
    imp = shape_metric(train_dict5[model_method], True)
    imp.to_csv(os.path.join(outdir,f"{evaluation_method}_{CommonParam.importance}.txt"), sep="\t", index=True, index_label=CommonParam.index)
    fig = plot_horizontal_barplot(imp)
    fig.savefig(os.path.join(outdir,f"{evaluation_method}_{CommonParam.importance}.jpg"))
    plt.close(fig)
    return imp


def extract_one_model(model_folder, split):
    model = os.path.basename(model_folder)
    train_file = join(model_folder, f"{CommonParam.train_prefix}_{CommonParam.metric}.txt")
    test_file = join(model_folder, f"{CommonParam.test_prefix}_{CommonParam.metric}.txt")
    train = read_file(train_file, index_col=0).T
    train.insert(0, CommonParam.dataset, train.index)
    train.insert(1, CommonParam.type, [CommonParam.train_prefix]*train.shape[0])
    train.insert(2, CommonParam.method, [model]*train.shape[0])
    test = read_file(test_file, index_col=0).T
    test.insert(0, CommonParam.dataset, test.index)
    test.insert(1, CommonParam.type, [CommonParam.test_prefix]*test.shape[0])
    test.insert(2, CommonParam.method, [model]*test.shape[0])
    cross_list = []
    for i in range(1,split+1):
        loc = f"{CommonParam.split}{i}"
        temp_cross = pd.concat([train.loc[loc,:], test.loc[loc,:]], axis=1).T
        cross_list.append(temp_cross)
    cross = pd.concat(cross_list, axis=0)
    return cross

def extract(merged_res, split):
    model_folders = glob.glob(join(merged_res,"**"))
    multi_cross_ali = []
    for mf in model_folders:
        if not os.path.isdir(mf):continue
        md_cross = extract_one_model(mf, split)
        multi_cross_ali.append(md_cross)
    # multi_cross = pd.concat(multi_cross_ali, axis=0).sort_index(key=lambda x:pd.Series(sorted([i.split("Split")[1] for i in x])))
    multi_cross = pd.concat(multi_cross_ali, axis=0)
    multi_cross.sort_values(by=[CommonParam.dataset, CommonParam.method], inplace=True)
    return multi_cross

def merge_multi_panel(merged_res_dict:dict, split):
    res_ali = []
    for panel, merged_dir in merged_res_dict.items():
        cur_res = extract(merged_res=merged_dir, split=split)
        cur_res.insert(0, CommonParam.panel, [panel]*cur_res.shape[0])
        res_ali.append(cur_res)
    res = pd.concat(res_ali, axis=0).sort_values(by=[CommonParam.dataset,CommonParam.panel, CommonParam.method])
    return res
