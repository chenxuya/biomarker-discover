import pandas as pd
import os
from os.path import join
import glob
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from str_param import CommonParam
from ..feature_evaluate import FeatureSelectorIndividual
from ..data_load import read_file
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn2, venn2_circles, venn3
import venn
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

def plot_roc(tpr, fpr):
    assert tpr.shape[1] == fpr.shape[1]
    fig, ax = plt.subplots(figsize=(10,10), dpi=100)
    aucs = list()
    mean_fpr = np.linspace(0,1,max(fpr.shape[0], 100))
    new_tpr = pd.DataFrame(index=range(len(mean_fpr)), columns=tpr.columns)
    for i in range(tpr.shape[1]):
        cur_fpr, cur_tpr = fpr.iloc[:,i], tpr.iloc[:,i]
        f = interpolate.interp1d(cur_fpr, cur_tpr)
        new_tpr.iloc[:,i] = f(mean_fpr)
        auc = metrics.auc(cur_fpr[cur_fpr.notna()], cur_tpr[cur_tpr.notna()])
        aucs.append(auc)
        # ax.plot(cur_fpr, cur_tpr, lw=1, alpha=0.3, label="ROC split %d (AUC = %0.3f)"%(i, auc))
        ax.plot(cur_fpr, cur_tpr, lw=1, alpha=0.3)
    mean_tpr = new_tpr.mean(axis=1)
    mean_tpr[0] = 0.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='crimson',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    std_tpr = new_tpr.std(axis=1)
    tprs_upper = np.array(np.minimum(mean_tpr + std_tpr, 1), dtype=float)
    tprs_lower = np.array(np.maximum(mean_tpr - std_tpr, 0),dtype=float)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
    ax.plot([0, 1], [0, 1], '--', lw=3, color = 'grey')
    ax.axis('square')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    ax.set_xlabel('False Positive Rate',fontsize=20)
    ax.set_ylabel('True Positive Rate',fontsize=20)
    ax.set_title('ROC Curve',fontsize=24)
    ax.legend(loc='lower right',fontsize=12)
    return fig


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

def venn_plot(df:pd.DataFrame):
    """
    Function for creating a Venn diagram plot based on the input DataFrame.

    Args:
    df (pd.DataFrame): The input DataFrame for creating the Venn diagram.

    Returns:
    fig: The generated figure object for the Venn diagram plot.
    """
    if df.shape[1]==1:return
    fig, ax = plt.subplots(figsize=(10,10), dpi=100)
    subsets = []
    set_labels = []
    set_colors = []
    colors = ['#00FFFF','#15B01A','#DDA0DD','#069AF3', '#FF4500', '#4B0082']

    for i in range(df.shape[1]):
        subsets.append(set(df.iloc[:,i].dropna()))
        set_labels.append(df.columns[i])
        set_colors.append(colors[i])
    if df.shape[1]==2:
        venn2(ax=ax, subsets=subsets, set_labels=set_labels, set_colors=set_colors, alpha=0.8, normalize_to=1.0)
    elif df.shape[1]==3:
        venn3(ax=ax, subsets=subsets, set_labels=set_labels, set_colors=set_colors, alpha=0.8, normalize_to=1.0)
    else:
        venn.venn(dict(zip(set_labels, subsets)), ax=ax)
    return fig

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
