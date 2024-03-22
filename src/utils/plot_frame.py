import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from sklearn import metrics
from matplotlib_venn import venn2, venn2_circles, venn3
import venn

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
    std_tpr = new_tpr.std(axis=1).fillna(0)
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


def plot_metric(data, filter_key:dict, group_col="Type", hue_col="Metric",
     x_col="Method", ncol=1, fig_cell_width=7, fig_cell_height=7, share_x=True, share_y=False,
      x_ticklabel_rotation=60):
    if filter_key:
        for key, value in filter_key.items():
            data = data[data[key].isin(value.split(";"))]

    metric_col = ["AUC", "Accuracy","F1_socre","Precision","Recall(Sensitivity)","Specificity","Brier_score"]

    data = data.drop(["AUC_CI_left","AUC_CI_right","AUC_pvalue", "Optimal_threshold"], axis=1)
    data.rename(columns={"Recall(Sensitivity)":"Sensitivity"})

    if group_col:
        grouped_data = data.groupby(group_col)
        gp_len = len(data[group_col].unique())
    else:
        grouped_data = [(None, data)]
        gp_len = 1
    nrow = gp_len//ncol if gp_len%ncol==0 else gp_len//ncol+1
    if gp_len < ncol:
        ncol =1
        nrow = gp_len

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(fig_cell_width*ncol,nrow*fig_cell_height), sharex=share_x, sharey=share_y)
    i = 0
    for gp, gp_data in grouped_data:
        data_long = pd.melt(gp_data, id_vars=data.columns.difference(metric_col), value_vars=metric_col, var_name="Metric", value_name="Value")
        if nrow==1 and ncol==1:
            cur_axes = axes
        else:
            cur_axes = axes[i] if gp_len==nrow else  axes[i//ncol][i%ncol] 
        sns.barplot(x=x_col, y="Value", hue=hue_col, data=data_long, ax=cur_axes)
        cur_axes.set_ylim(0, 1.1)
        cur_axes.set_xticklabels(cur_axes.get_xticklabels(), rotation=x_ticklabel_rotation)
        # Create the legend for the current axes and place it outside the plot
        handles, labels = cur_axes.get_legend_handles_labels()
        cur_axes.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        i +=1
    # 处理子图数量不是完全填满最后一行的情况
    if len(grouped_data) % ncol:
        for j in range(i, nrow * ncol):
            fig.delaxes(axes[j//ncol][j%ncol])

    plt.subplots_adjust(right=0.85)  # You may need to adjust this value to fit your legend.
    plt.tight_layout()
    return fig

# 定义绘图函数
def plot_horizontal_barplot(df, top_n=10):
    """
    函数接收一个DataFrame，绘制每个化合物的Mean值的横向条形图。
    """
    df.fillna(0, inplace=True)
    df = df.sort_values(by="Mean", ascending=False)
    df = df.head(top_n).sort_values(by="Mean", ascending=True)
    fig = plt.figure(figsize=(10, 6))
    df["Mean"].plot(kind='barh', xerr=df["Std"], color='skyblue')
    plt.xlabel('Mean Importance')
    plt.title('Mean Importance of Feature with Standard Deviation')
    plt.tight_layout()
    return fig




if __name__ == "__main__":
    data = pd.read_csv(r"/share2/users/chenxu/project/mGWAS/medicine2/MWY-22-2897.WHUH_75/BIOINF-242682_2.word/analysis_emphysema/ml2/metric_summary.txt", sep="\t")
    fig = plot_metric(data, filter_key=None)
    fig.savefig("/share2/users/chenxu/project/mGWAS/medicine2/MWY-22-2897.WHUH_75/BIOINF-242682_2.word/code/temp.jpg")