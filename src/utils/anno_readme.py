import markdown
import os
from pathlib import Path
import re
from os.path import join

def pipeline_summary(cfgs):
    lines_to_insert = f"""本次建模(筛选)方法: 

1.经过客户指定筛选后, 剩余的变量用不同的机器学习模型再做建模或筛选。
2.指定训练集和测试集。
3.训练集和测试集中的数值型变量(代谢物,菌群,蛋白等组学)做先做{cfgs.transform}转换,随后{cfgs.scale}处理。如果有额外添加的指标(比如临床指标),则其中, 数值型变量做{cfgs.norminal_scale}处理。类别型变量one-hot编码。
4.选择一个机器学习模型, 在训练集中进行训练。并用{cfgs.k_fold}折交叉验证的方法找到最佳模型。
5.利用最佳模型在测试集中测试, 获得模型的各种评价指标。
6.重复步骤2-5 {cfgs.split_times}次, 获得模型在{cfgs.split_times}次数据拆分中的成绩。获得每个特征的在{cfgs.split_times}次数据拆分中的平均重要性。
7.挑选可能的biomarker。挑选方法: 按平均特征重要性, 从大到小排列。特征重要性依次累计相加, 直到累计值大于所有特征重要性总和的75%(95%)时停止累计相加。用于累计相加的特征即为我们选定的可能marker。(如果仅需要建模验证, 则该步的结果可以忽略)
8.换用不同的模型, 重复步骤4-7。获得每个模型挑选出的可能的biomarker。(如果仅建模, 则该的结果可以忽略)
9.对不同模型挑选出的biomarker取交集。同时根据测试集中模型的平均AUC值, 对每个模型中各特征的重要性进行权重相加, 得到weighted 特征重要性, 供客户后续特征挑选参考。
10.需要根据客户的要求, 去掉或再添加某些特征再做机器学习, 检验这些biomarker的有效性。(如果仅建模, 则不需要继续进行该步)

accumulation_75(95)percent计算方法:
按平均特征重要性,从大到小排列。特征重要性依次累计相加, 直到累计值大于所有特征重要性总和的75%(95%)时停止累计相加。用于累计相加的特征即为我们选定的可能marker。如果仅需要建模, 可以忽略该结果。
\n\n\n\n"""
    return lines_to_insert


def txt2html(infile, outfile):
    md_file = os.path.splitext(infile)[0] + ".md"
    md_hander = open(md_file, "w", encoding="utf-8")
    txt_hander = open(infile, "r", encoding="utf-8")
    text = txt_hander.readlines()
    for i in range(len(text)):
        l = text[i].replace("\n", "").replace("    ", "&emsp;") + "<br>\n" # html专用格式
        md_hander.write(l)
    txt_hander.close()
    md_hander.close()

    md_hander = open(md_file, "r", encoding="utf-8")
    html_hander = open(outfile, "w", encoding="utf-8")
    text2html = md_hander.read()
    html = markdown.markdown(text2html)
    html_hander.write(html)
    md_hander.close()
    html_hander.close()
    os.remove(md_file)

def tree_readme(adir):
    # 以tree的形式展示adir下所有文件
    def generate_tree(tree_str, pathname, n=0):
        num = 0 if n==0 else 4
        if pathname.is_file():
            tree_str += '    |' * n + '-' * num + pathname.name + '\n'
        elif pathname.is_dir():
            tree_str += '    |' * n + '-' * num + \
                str(pathname.relative_to(pathname.parent)) + '\\' + '\n'
            for cp in sorted(pathname.iterdir(), key = lambda x:str(x).lower()):
                tree_str = generate_tree(tree_str,cp, n + 1)
        return tree_str
    tree_str = generate_tree("",Path(adir), 0)
    return tree_str

def anno_tree(tree_readme):
    # 文件注释
    anno_dict = {
        "Split1_.*_vs_.*_Train\.txt":" # 第1次分割的训练数据。",
        "Split1_.*_vs_.*_Test\.txt":" # 第1次分割的测试数据。",
        "MergedRes":" # 重要文件夹",
        "RFE_Importance.txt":" # 根据RFE算法获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。",
        "SavedModel_Importance.txt": " # 根据模型自身特性获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。",
        "SHAP_Importance.txt": " # 根据SHAP算法获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。",
        "Test_ROC.jpg": " # 测试集中的ROC曲线图",
        "RFE_cumulative_75percent_venn":" (重要图片,如果只是构建模型可以忽略该文件) # 利用RFE算法评价特征重要性。根据累计值75%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图",
        "RFE_cumulative_95percent_venn":" (重要图片,如果只是构建模型可以忽略该文件) # 利用RFE算法评价特征重要性。根据累计值95%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图",
        "SavedModel_cumulative_75percent_venn":" (重要图片,如果只是构建模型可以忽略该文件) # 利用模型自身特性评价特征重要性。根据累计值75%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图",
        "SavedModel_cumulative_95percent_venn":" (重要图片,如果只是构建模型可以忽略该文件) # 利用模型自身特性评价特征重要性。根据累计值95%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图",
        "SHAP_cumulative_75percent_venn":" (重要图片,如果只是构建模型可以忽略该文件) # 利用SHAP算法评价特征重要性。根据累计值75%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图",
        "SHAP_cumulative_95percent_venn":" (重要图片,如果只是构建模型可以忽略该文件) # 利用SHAP算法评价特征重要性。根据累计值95%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图",
        "Test_Metric.txt": " # 模型在测试集中获得的各种评价指标,包括AUC,AUC_CI_left(auc 置信区间的下区间), AUC_CI_right(auc 置信区间的上区间),AUC_pvalue, specificity,f1-score, precision,recall(Sensitivity), Optimal_threshold,以及它们在不同数据拆分中的平均值和标准差",
        "Test_FPR.txt" :"# 模型在测试集中绘制ROC曲线所需的false positive rate",
        "Test_Thresholds.txt": " # 测试集中,绘制ROC曲线时所需的threshold",
        "Test_TPR.txt": " # 模型在测试集中绘制ROC曲线所需的true positive rate",
        "summary.xlsx" :"(重要结果) # 根据不同的累计值(75%,95%)筛选出的marker汇总,以及它们之间的交集汇总。weighted_feature_importance这个sheet用于后续客户特征挑选参考(参考筛选或建模步骤)。",
        "Test_Split1_[A-Z]*_confusion_matrix.jpg": " # 测试集中的混淆矩阵可视化",
        "Test_Split1_[A-Z]*_confusion_matrix.txt": " # 测试集中的混淆矩阵",
        "Train_Split1_[A-Z]*_confusion_matrix.jpg": " # 训练集中的混淆矩阵可视化",
        "Train_Split1_[A-Z]*_confusion_matrix.txt": " # 训练集中的混淆矩阵",
        "Split1_[A-Z]*_RFEImportance.txt": " # RFE特征重要性。值越大,表明该特征越重要。",
        "Split1_[A-Z]*_SHAPImportance.txt": " # SHAP特征重要性。值越大,表明该特征越重要。",
        "Test_Split1_[A-Z]*_metrics.txt": " # 模型在测试集中的各种评价指标。",
        "Train_Split1_[A-Z]*_metrics.txt": " # 模型在训练集中的各种评价指标。",
        "Test_Split1_[A-Z]*_roc_curve.jpg": " # 测试集中的ROC曲线",
        "Test_Split1_[A-Z]*_roc_curve.txt":" # 测试集中用于做ROC曲线的数据。",
        "Test_Split1_[A-Z]*_TrueAndPredict.txt": "# 在测试集中模型预测为某一类的概率,以及模型实际预测的值和真值。",
        "Train_Split1_[A-Z]*_roc_curve.jpg":" # 训练集中的ROC曲线",
        "Train_Split1_[A-Z]*_roc_curve.txt":" # 训练集中用于做ROC曲线的数据。",
        "Train_Split1_[A-Z]*_TrueAndPredict.txt": " # 在训练集中模型预测为某一类的概率,以及模型实际预测的值和真值。",
        "Split1_(LASSO|LR|RFC|XGBOOST|SVM)\.txt":" # 模型的最佳超参数。",
        "Split1_.*RFE.pkl":" # 保存的RFE模型。",
        "Split1_.*SHAP.pkl":" # 保存的SHAP模型。",
        "Split1_.*.pkl":" # 保存的最佳模型。",
        "Test_Split1_multi_models_roc.jpg": " # 测试集中,多种不同模型的ROC绘制于同一图中。",
        "Train_Split1_multi_models_roc.jpg": " # 训练集中,多种不同模型的ROC绘制于同一图中。"
    }
    # 文件夹注释
    anno_dict2 = {
        "LASSO":" # 利用LASSO模型建模(筛选)的结果。",
        "LR":" # 利用logistic regression 建模(筛选)的结果。",
        "RFC": "  # 利用随机森林建模(筛选)的结果。",
        "Split\d+":" # 某一次数据拆分的结果",
        "Confusion": " # 混淆矩阵",
        "Importance": " # 特征重要性",
        "Metric":" # 评价指标",
    }
    lines = tree_readme.split("\n")
    # 先注释文件夹
    new_lines = []
    for l in lines:
        if (".jpg" in l) or (".txt" in l) or (".pkl" in l) or (".pdf" in l):
            new_lines.append(l)
            continue
        matched = False
        for pattern in anno_dict2.keys():
            res1 = re.search(pattern, l)
            if res1 is not None:
                matched = True
                anno = anno_dict2[pattern]
                break
        if matched:
            new_lines.append(l.replace("\n", "")+anno)
        else:
            new_lines.append(l)

    new_lines2 = []
    for l in new_lines:
        matched = False
        for pattern in anno_dict.keys(): # 遍历key,看这一行是否需要注释,以及注释哪一个
            result = re.search(pattern, l)
            if result is not None:
                matched = True
                anno = anno_dict[pattern] # 注释上的文本内容
                break
        if matched:
            new_lines2.append(l.replace("\n", "") + anno + "\n")
        else:
            new_lines2.append(l+"\n")
    return new_lines2
    
def annoed_readme(adir, cfgs):
    basic_tree = tree_readme(adir=adir)
    annoed_tree = anno_tree(basic_tree)
    pipeline_summary_str = pipeline_summary(cfgs)
    annoed_tree.insert(0, pipeline_summary_str)
    txt_hander = open(os.path.join(adir, "readme.txt"), "w", encoding="utf-8")
    txt_hander.writelines(annoed_tree) 
    txt_hander.close()
    txt2html(os.path.join(adir, "readme.txt"),os.path.join(adir, "readme.html"))

if __name__ == "__main__":
    class cfgs:
        transform = "no"
        scale = "scale"
        norminal_scale = "minmax"
        k_fold = "k_fold"
        split_times = 5
    
    annoed_readme("/share2/users/chenxu/code/tools2/biomarker-discovery/tests/Mild_vs_moderate-severe", cfgs)