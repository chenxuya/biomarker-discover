本次建模(筛选)方法: 

1.经过客户指定筛选后, 剩余的变量用不同的机器学习模型再做建模或筛选。
2.指定训练集和测试集。
3.训练集和测试集中的数值型变量(代谢物,菌群,蛋白等组学)做先做log2转换,随后min-max处理。如果有额外添加的指标(比如临床指标),则其中, 数值型变量做None处理。类别型变量one-hot编码。
4.选择一个机器学习模型, 在训练集中进行训练。并用5折交叉验证的方法找到最佳模型。
5.利用最佳模型在测试集中测试, 获得模型的各种评价指标。
6.重复步骤2-5 2次, 获得模型在2次数据拆分中的成绩。获得每个特征的在2次数据拆分中的平均重要性。
7.挑选可能的biomarker。挑选方法: 按平均特征重要性, 从大到小排列。特征重要性依次累计相加, 直到累计值大于所有特征重要性总和的75%(95%)时停止累计相加。用于累计相加的特征即为我们选定的可能marker。(如果仅需要建模验证, 则该步的结果可以忽略)
8.换用不同的模型, 重复步骤4-7。获得每个模型挑选出的可能的biomarker。(如果仅建模, 则该的结果可以忽略)
9.对不同模型挑选出的biomarker取交集。同时根据测试集中模型的平均AUC值, 对每个模型中各特征的重要性进行权重相加, 得到weighted 特征重要性, 供客户后续特征挑选参考。
10.需要根据客户的要求, 去掉或再添加某些特征再做机器学习, 检验这些biomarker的有效性。(如果仅建模, 则不需要继续进行该步)

accumulation_75(95)percent计算方法:
按平均特征重要性,从大到小排列。特征重要性依次累计相加, 直到累计值大于所有特征重要性总和的75%(95%)时停止累计相加。用于累计相加的特征即为我们选定的可能marker。如果仅需要建模, 可以忽略该结果。




Moderate_vs_Severe\
    |----data\
    |    |----Split1_Moderate_vs_Severe_Test.txt # 模型的最佳超参数。
    |    |----Split1_Moderate_vs_Severe_Train.txt # 模型的最佳超参数。
    |    |----Split2_Moderate_vs_Severe_Test.txt
    |    |----Split2_Moderate_vs_Severe_Train.txt
    |----MergedResult\ # 重要文件夹
    |    |----LASSO\ # 利用LASSO模型建模(筛选)的结果。
    |    |    |----RFE_Importance.txt # 根据RFE算法获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----SavedModel_Importance.jpg
    |    |    |----SavedModel_Importance.txt # 根据模型自身特性获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----SHAP_Importance.txt # 根据SHAP算法获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----Test_FPR.txt# 模型在测试集中绘制ROC曲线所需的false positive rate
    |    |    |----Test_Metric.txt # 模型在测试集中获得的各种评价指标,包括AUC,AUC_CI_left(auc 置信区间的下区间), AUC_CI_right(auc 置信区间的上区间),AUC_pvalue, specificity,f1-score, precision,recall(Sensitivity), Optimal_threshold,以及它们在不同数据拆分中的平均值和标准差
    |    |    |----Test_ROC.jpg # 测试集中的ROC曲线图
    |    |    |----Test_Thresholds.txt # 测试集中,绘制ROC曲线时所需的threshold
    |    |    |----Test_TPR.txt # 模型在测试集中绘制ROC曲线所需的true positive rate
    |    |    |----Train_FPR.txt
    |    |    |----Train_Metric.txt
    |    |    |----Train_ROC.jpg
    |    |    |----Train_Thresholds.txt
    |    |    |----Train_TPR.txt
    |    |----LR\ # 利用logistic regression 建模(筛选)的结果。
    |    |    |----RFE_Importance.txt # 根据RFE算法获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----SavedModel_Importance.jpg
    |    |    |----SavedModel_Importance.txt # 根据模型自身特性获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----SHAP_Importance.txt # 根据SHAP算法获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----Test_FPR.txt# 模型在测试集中绘制ROC曲线所需的false positive rate
    |    |    |----Test_Metric.txt # 模型在测试集中获得的各种评价指标,包括AUC,AUC_CI_left(auc 置信区间的下区间), AUC_CI_right(auc 置信区间的上区间),AUC_pvalue, specificity,f1-score, precision,recall(Sensitivity), Optimal_threshold,以及它们在不同数据拆分中的平均值和标准差
    |    |    |----Test_ROC.jpg # 测试集中的ROC曲线图
    |    |    |----Test_Thresholds.txt # 测试集中,绘制ROC曲线时所需的threshold
    |    |    |----Test_TPR.txt # 模型在测试集中绘制ROC曲线所需的true positive rate
    |    |    |----Train_FPR.txt
    |    |    |----Train_Metric.txt
    |    |    |----Train_ROC.jpg
    |    |    |----Train_Thresholds.txt
    |    |    |----Train_TPR.txt
    |    |----RFC\  # 利用随机森林建模(筛选)的结果。
    |    |    |----RFE_Importance.txt # 根据RFE算法获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----SavedModel_Importance.jpg
    |    |    |----SavedModel_Importance.txt # 根据模型自身特性获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----SHAP_Importance.txt # 根据SHAP算法获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----Test_FPR.txt# 模型在测试集中绘制ROC曲线所需的false positive rate
    |    |    |----Test_Metric.txt # 模型在测试集中获得的各种评价指标,包括AUC,AUC_CI_left(auc 置信区间的下区间), AUC_CI_right(auc 置信区间的上区间),AUC_pvalue, specificity,f1-score, precision,recall(Sensitivity), Optimal_threshold,以及它们在不同数据拆分中的平均值和标准差
    |    |    |----Test_ROC.jpg # 测试集中的ROC曲线图
    |    |    |----Test_Thresholds.txt # 测试集中,绘制ROC曲线时所需的threshold
    |    |    |----Test_TPR.txt # 模型在测试集中绘制ROC曲线所需的true positive rate
    |    |    |----Train_FPR.txt
    |    |    |----Train_Metric.txt
    |    |    |----Train_ROC.jpg
    |    |    |----Train_Thresholds.txt
    |    |    |----Train_TPR.txt
    |    |----RFE_cumulative_75percent_venn.png (重要图片,如果只是构建模型可以忽略该文件) # 利用RFE算法评价特征重要性。根据累计值75%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图
    |    |----RFE_cumulative_95percent_venn.png (重要图片,如果只是构建模型可以忽略该文件) # 利用RFE算法评价特征重要性。根据累计值95%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图
    |    |----RFE_summary.xlsx(重要结果) # 根据不同的累计值(75%,95%)筛选出的marker汇总,以及它们之间的交集汇总。weighted_feature_importance这个sheet用于后续客户特征挑选参考(参考筛选或建模步骤)。
    |    |----SavedModel_cumulative_75percent_venn.png (重要图片,如果只是构建模型可以忽略该文件) # 利用模型自身特性评价特征重要性。根据累计值75%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图
    |    |----SavedModel_cumulative_95percent_venn.png (重要图片,如果只是构建模型可以忽略该文件) # 利用模型自身特性评价特征重要性。根据累计值95%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图
    |    |----SavedModel_summary.xlsx(重要结果) # 根据不同的累计值(75%,95%)筛选出的marker汇总,以及它们之间的交集汇总。weighted_feature_importance这个sheet用于后续客户特征挑选参考(参考筛选或建模步骤)。
    |    |----SHAP_cumulative_75percent_venn.png (重要图片,如果只是构建模型可以忽略该文件) # 利用SHAP算法评价特征重要性。根据累计值75%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图
    |    |----SHAP_cumulative_95percent_venn.png (重要图片,如果只是构建模型可以忽略该文件) # 利用SHAP算法评价特征重要性。根据累计值95%(累计值说明参考筛选或建模方法),不同模型筛选出的biomarker的韦恩图
    |    |----SHAP_summary.xlsx(重要结果) # 根据不同的累计值(75%,95%)筛选出的marker汇总,以及它们之间的交集汇总。weighted_feature_importance这个sheet用于后续客户特征挑选参考(参考筛选或建模步骤)。
    |    |----Test_Metric.png # 评价指标
    |    |----Test_Metric.txt # 模型在测试集中获得的各种评价指标,包括AUC,AUC_CI_left(auc 置信区间的下区间), AUC_CI_right(auc 置信区间的上区间),AUC_pvalue, specificity,f1-score, precision,recall(Sensitivity), Optimal_threshold,以及它们在不同数据拆分中的平均值和标准差
    |    |----TestMetric.png # 评价指标
    |    |----Train_Metric.png # 评价指标
    |    |----Train_Metric.txt
    |    |----TrainMetric.png # 评价指标
    |    |----XGBOOST\
    |    |    |----RFE_Importance.txt # 根据RFE算法获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----SavedModel_Importance.jpg
    |    |    |----SavedModel_Importance.txt # 根据模型自身特性获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----SHAP_Importance.txt # 根据SHAP算法获得的特征重要性。每个特征在不同数据拆分中的重要性,以及它们在不同数据拆分中的均值和标准差。
    |    |    |----Test_FPR.txt# 模型在测试集中绘制ROC曲线所需的false positive rate
    |    |    |----Test_Metric.txt # 模型在测试集中获得的各种评价指标,包括AUC,AUC_CI_left(auc 置信区间的下区间), AUC_CI_right(auc 置信区间的上区间),AUC_pvalue, specificity,f1-score, precision,recall(Sensitivity), Optimal_threshold,以及它们在不同数据拆分中的平均值和标准差
    |    |    |----Test_ROC.jpg # 测试集中的ROC曲线图
    |    |    |----Test_Thresholds.txt # 测试集中,绘制ROC曲线时所需的threshold
    |    |    |----Test_TPR.txt # 模型在测试集中绘制ROC曲线所需的true positive rate
    |    |    |----Train_FPR.txt
    |    |    |----Train_Metric.txt
    |    |    |----Train_ROC.jpg
    |    |    |----Train_Thresholds.txt
    |    |    |----Train_TPR.txt
    |----readme.html
    |----readme.txt
    |----Split1\ # 某一次数据拆分的结果
    |    |----LASSO\ # 利用LASSO模型建模(筛选)的结果。
    |    |    |----Confusion\ # 混淆矩阵
    |    |    |    |----Test_Split1_LASSO_confusion_matrix.jpg # 测试集中的混淆矩阵可视化
    |    |    |    |----Test_Split1_LASSO_confusion_matrix.pdf
    |    |    |    |----Test_Split1_LASSO_confusion_matrix.txt # 测试集中的混淆矩阵
    |    |    |    |----Train_Split1_LASSO_confusion_matrix.jpg # 训练集中的混淆矩阵可视化
    |    |    |    |----Train_Split1_LASSO_confusion_matrix.pdf
    |    |    |    |----Train_Split1_LASSO_confusion_matrix.txt # 训练集中的混淆矩阵
    |    |    |----Importance\ # 特征重要性
    |    |    |    |----Split1_LASSO_RFEImportance.txt # RFE特征重要性。值越大,表明该特征越重要。
    |    |    |    |----Split1_LASSO_SHAPImportance.txt # SHAP特征重要性。值越大,表明该特征越重要。
    |    |    |    |----Split1_LASSO_variableImportance.txt # 模型的最佳超参数。
    |    |    |----Metric\ # 评价指标
    |    |    |    |----Test_Split1_LASSO_metrics.txt # 模型在测试集中的各种评价指标。
    |    |    |    |----Train_Split1_LASSO_metrics.txt # 模型在训练集中的各种评价指标。
    |    |    |----ROC\
    |    |    |    |----Test_Split1_LASSO_roc_curve.jpg # 测试集中的ROC曲线
    |    |    |    |----Test_Split1_LASSO_roc_curve.pdf
    |    |    |    |----Test_Split1_LASSO_roc_curve.txt # 测试集中用于做ROC曲线的数据。
    |    |    |    |----Test_Split1_LASSO_TrueAndPredict.txt# 在测试集中模型预测为某一类的概率,以及模型实际预测的值和真值。
    |    |    |    |----Train_Split1_LASSO_roc_curve.jpg # 训练集中的ROC曲线
    |    |    |    |----Train_Split1_LASSO_roc_curve.pdf
    |    |    |    |----Train_Split1_LASSO_roc_curve.txt # 训练集中用于做ROC曲线的数据。
    |    |    |    |----Train_Split1_LASSO_TrueAndPredict.txt # 在训练集中模型预测为某一类的概率,以及模型实际预测的值和真值。
    |    |    |----SavedModel\
    |    |    |    |----Split1_LASSO.pkl # 保存的最佳模型。
    |    |    |    |----Split1_LASSO.txt # 模型的最佳超参数。
    |    |    |    |----Split1_LASSORFE.pkl # 保存的RFE模型。
    |    |    |    |----Split1_LASSOSHAP.pkl # 保存的SHAP模型。
    |    |----LR\ # 利用logistic regression 建模(筛选)的结果。
    |    |    |----Confusion\ # 混淆矩阵
    |    |    |    |----Test_Split1_LR_confusion_matrix.jpg # 测试集中的混淆矩阵可视化
    |    |    |    |----Test_Split1_LR_confusion_matrix.pdf
    |    |    |    |----Test_Split1_LR_confusion_matrix.txt # 测试集中的混淆矩阵
    |    |    |    |----Train_Split1_LR_confusion_matrix.jpg # 训练集中的混淆矩阵可视化
    |    |    |    |----Train_Split1_LR_confusion_matrix.pdf
    |    |    |    |----Train_Split1_LR_confusion_matrix.txt # 训练集中的混淆矩阵
    |    |    |----Importance\ # 特征重要性
    |    |    |    |----Split1_LR_RFEImportance.txt # RFE特征重要性。值越大,表明该特征越重要。
    |    |    |    |----Split1_LR_SHAPImportance.txt # SHAP特征重要性。值越大,表明该特征越重要。
    |    |    |    |----Split1_LR_variableImportance.txt # 模型的最佳超参数。
    |    |    |----Metric\ # 评价指标
    |    |    |    |----Test_Split1_LR_metrics.txt # 模型在测试集中的各种评价指标。
    |    |    |    |----Train_Split1_LR_metrics.txt # 模型在训练集中的各种评价指标。
    |    |    |----ROC\
    |    |    |    |----Test_Split1_LR_roc_curve.jpg # 测试集中的ROC曲线
    |    |    |    |----Test_Split1_LR_roc_curve.pdf
    |    |    |    |----Test_Split1_LR_roc_curve.txt # 测试集中用于做ROC曲线的数据。
    |    |    |    |----Test_Split1_LR_TrueAndPredict.txt# 在测试集中模型预测为某一类的概率,以及模型实际预测的值和真值。
    |    |    |    |----Train_Split1_LR_roc_curve.jpg # 训练集中的ROC曲线
    |    |    |    |----Train_Split1_LR_roc_curve.pdf
    |    |    |    |----Train_Split1_LR_roc_curve.txt # 训练集中用于做ROC曲线的数据。
    |    |    |    |----Train_Split1_LR_TrueAndPredict.txt # 在训练集中模型预测为某一类的概率,以及模型实际预测的值和真值。
    |    |    |----SavedModel\
    |    |    |    |----Split1_LR.pkl # 保存的最佳模型。
    |    |    |    |----Split1_LR.txt # 模型的最佳超参数。
    |    |    |    |----Split1_LRRFE.pkl # 保存的RFE模型。
    |    |    |    |----Split1_LRSHAP.pkl # 保存的SHAP模型。
    |    |----RFC\  # 利用随机森林建模(筛选)的结果。
    |    |    |----Confusion\ # 混淆矩阵
    |    |    |    |----Test_Split1_RFC_confusion_matrix.jpg # 测试集中的混淆矩阵可视化
    |    |    |    |----Test_Split1_RFC_confusion_matrix.pdf
    |    |    |    |----Test_Split1_RFC_confusion_matrix.txt # 测试集中的混淆矩阵
    |    |    |    |----Train_Split1_RFC_confusion_matrix.jpg # 训练集中的混淆矩阵可视化
    |    |    |    |----Train_Split1_RFC_confusion_matrix.pdf
    |    |    |    |----Train_Split1_RFC_confusion_matrix.txt # 训练集中的混淆矩阵
    |    |    |----Importance\ # 特征重要性
    |    |    |    |----Split1_RFC_RFEImportance.txt # RFE特征重要性。值越大,表明该特征越重要。
    |    |    |    |----Split1_RFC_SHAPImportance.txt # SHAP特征重要性。值越大,表明该特征越重要。
    |    |    |    |----Split1_RFC_variableImportance.txt # 模型的最佳超参数。
    |    |    |----Metric\ # 评价指标
    |    |    |    |----Test_Split1_RFC_metrics.txt # 模型在测试集中的各种评价指标。
    |    |    |    |----Train_Split1_RFC_metrics.txt # 模型在训练集中的各种评价指标。
    |    |    |----ROC\
    |    |    |    |----Test_Split1_RFC_roc_curve.jpg # 测试集中的ROC曲线
    |    |    |    |----Test_Split1_RFC_roc_curve.pdf
    |    |    |    |----Test_Split1_RFC_roc_curve.txt # 测试集中用于做ROC曲线的数据。
    |    |    |    |----Test_Split1_RFC_TrueAndPredict.txt# 在测试集中模型预测为某一类的概率,以及模型实际预测的值和真值。
    |    |    |    |----Train_Split1_RFC_roc_curve.jpg # 训练集中的ROC曲线
    |    |    |    |----Train_Split1_RFC_roc_curve.pdf
    |    |    |    |----Train_Split1_RFC_roc_curve.txt # 训练集中用于做ROC曲线的数据。
    |    |    |    |----Train_Split1_RFC_TrueAndPredict.txt # 在训练集中模型预测为某一类的概率,以及模型实际预测的值和真值。
    |    |    |----SavedModel\
    |    |    |    |----Split1_RFC.pkl # 保存的最佳模型。
    |    |    |    |----Split1_RFC.txt # 模型的最佳超参数。
    |    |    |    |----Split1_RFCRFE.pkl # 保存的RFE模型。
    |    |    |    |----Split1_RFCSHAP.pkl # 保存的SHAP模型。
    |    |----XGBOOST\
    |    |    |----Confusion\ # 混淆矩阵
    |    |    |    |----Test_Split1_XGBOOST_confusion_matrix.jpg # 测试集中的混淆矩阵可视化
    |    |    |    |----Test_Split1_XGBOOST_confusion_matrix.pdf
    |    |    |    |----Test_Split1_XGBOOST_confusion_matrix.txt # 测试集中的混淆矩阵
    |    |    |    |----Train_Split1_XGBOOST_confusion_matrix.jpg # 训练集中的混淆矩阵可视化
    |    |    |    |----Train_Split1_XGBOOST_confusion_matrix.pdf
    |    |    |    |----Train_Split1_XGBOOST_confusion_matrix.txt # 训练集中的混淆矩阵
    |    |    |----Importance\ # 特征重要性
    |    |    |    |----Split1_XGBOOST_RFEImportance.txt # RFE特征重要性。值越大,表明该特征越重要。
    |    |    |    |----Split1_XGBOOST_SHAPImportance.txt # SHAP特征重要性。值越大,表明该特征越重要。
    |    |    |    |----Split1_XGBOOST_variableImportance.txt # 模型的最佳超参数。
    |    |    |----Metric\ # 评价指标
    |    |    |    |----Test_Split1_XGBOOST_metrics.txt # 模型在测试集中的各种评价指标。
    |    |    |    |----Train_Split1_XGBOOST_metrics.txt # 模型在训练集中的各种评价指标。
    |    |    |----ROC\
    |    |    |    |----Test_Split1_XGBOOST_roc_curve.jpg # 测试集中的ROC曲线
    |    |    |    |----Test_Split1_XGBOOST_roc_curve.pdf
    |    |    |    |----Test_Split1_XGBOOST_roc_curve.txt # 测试集中用于做ROC曲线的数据。
    |    |    |    |----Test_Split1_XGBOOST_TrueAndPredict.txt# 在测试集中模型预测为某一类的概率,以及模型实际预测的值和真值。
    |    |    |    |----Train_Split1_XGBOOST_roc_curve.jpg # 训练集中的ROC曲线
    |    |    |    |----Train_Split1_XGBOOST_roc_curve.pdf
    |    |    |    |----Train_Split1_XGBOOST_roc_curve.txt # 训练集中用于做ROC曲线的数据。
    |    |    |    |----Train_Split1_XGBOOST_TrueAndPredict.txt # 在训练集中模型预测为某一类的概率,以及模型实际预测的值和真值。
    |    |    |----SavedModel\
    |    |    |    |----Split1_XGBOOST.pkl # 保存的最佳模型。
    |    |    |    |----Split1_XGBOOST.txt # 模型的最佳超参数。
    |    |    |    |----Split1_XGBOOSTRFE.pkl # 保存的RFE模型。
    |    |    |    |----Split1_XGBOOSTSHAP.pkl # 保存的SHAP模型。
    |----Split2\ # 某一次数据拆分的结果
    |    |----LASSO\ # 利用LASSO模型建模(筛选)的结果。
    |    |    |----Confusion\ # 混淆矩阵
    |    |    |    |----Test_Split2_LASSO_confusion_matrix.jpg
    |    |    |    |----Test_Split2_LASSO_confusion_matrix.pdf
    |    |    |    |----Test_Split2_LASSO_confusion_matrix.txt
    |    |    |    |----Train_Split2_LASSO_confusion_matrix.jpg
    |    |    |    |----Train_Split2_LASSO_confusion_matrix.pdf
    |    |    |    |----Train_Split2_LASSO_confusion_matrix.txt
    |    |    |----Importance\ # 特征重要性
    |    |    |    |----Split2_LASSO_RFEImportance.txt
    |    |    |    |----Split2_LASSO_SHAPImportance.txt
    |    |    |    |----Split2_LASSO_variableImportance.txt
    |    |    |----Metric\ # 评价指标
    |    |    |    |----Test_Split2_LASSO_metrics.txt
    |    |    |    |----Train_Split2_LASSO_metrics.txt
    |    |    |----ROC\
    |    |    |    |----Test_Split2_LASSO_roc_curve.jpg
    |    |    |    |----Test_Split2_LASSO_roc_curve.pdf
    |    |    |    |----Test_Split2_LASSO_roc_curve.txt
    |    |    |    |----Test_Split2_LASSO_TrueAndPredict.txt
    |    |    |    |----Train_Split2_LASSO_roc_curve.jpg
    |    |    |    |----Train_Split2_LASSO_roc_curve.pdf
    |    |    |    |----Train_Split2_LASSO_roc_curve.txt
    |    |    |    |----Train_Split2_LASSO_TrueAndPredict.txt
    |    |    |----SavedModel\
    |    |    |    |----Split2_LASSO.pkl
    |    |    |    |----Split2_LASSO.txt
    |    |    |    |----Split2_LASSORFE.pkl
    |    |    |    |----Split2_LASSOSHAP.pkl
    |    |----LR\ # 利用logistic regression 建模(筛选)的结果。
    |    |    |----Confusion\ # 混淆矩阵
    |    |    |    |----Test_Split2_LR_confusion_matrix.jpg
    |    |    |    |----Test_Split2_LR_confusion_matrix.pdf
    |    |    |    |----Test_Split2_LR_confusion_matrix.txt
    |    |    |    |----Train_Split2_LR_confusion_matrix.jpg
    |    |    |    |----Train_Split2_LR_confusion_matrix.pdf
    |    |    |    |----Train_Split2_LR_confusion_matrix.txt
    |    |    |----Importance\ # 特征重要性
    |    |    |    |----Split2_LR_RFEImportance.txt
    |    |    |    |----Split2_LR_SHAPImportance.txt
    |    |    |    |----Split2_LR_variableImportance.txt
    |    |    |----Metric\ # 评价指标
    |    |    |    |----Test_Split2_LR_metrics.txt
    |    |    |    |----Train_Split2_LR_metrics.txt
    |    |    |----ROC\
    |    |    |    |----Test_Split2_LR_roc_curve.jpg
    |    |    |    |----Test_Split2_LR_roc_curve.pdf
    |    |    |    |----Test_Split2_LR_roc_curve.txt
    |    |    |    |----Test_Split2_LR_TrueAndPredict.txt
    |    |    |    |----Train_Split2_LR_roc_curve.jpg
    |    |    |    |----Train_Split2_LR_roc_curve.pdf
    |    |    |    |----Train_Split2_LR_roc_curve.txt
    |    |    |    |----Train_Split2_LR_TrueAndPredict.txt
    |    |    |----SavedModel\
    |    |    |    |----Split2_LR.pkl
    |    |    |    |----Split2_LR.txt
    |    |    |    |----Split2_LRRFE.pkl
    |    |    |    |----Split2_LRSHAP.pkl
    |    |----RFC\  # 利用随机森林建模(筛选)的结果。
    |    |    |----Confusion\ # 混淆矩阵
    |    |    |    |----Test_Split2_RFC_confusion_matrix.jpg
    |    |    |    |----Test_Split2_RFC_confusion_matrix.pdf
    |    |    |    |----Test_Split2_RFC_confusion_matrix.txt
    |    |    |    |----Train_Split2_RFC_confusion_matrix.jpg
    |    |    |    |----Train_Split2_RFC_confusion_matrix.pdf
    |    |    |    |----Train_Split2_RFC_confusion_matrix.txt
    |    |    |----Importance\ # 特征重要性
    |    |    |    |----Split2_RFC_RFEImportance.txt
    |    |    |    |----Split2_RFC_SHAPImportance.txt
    |    |    |    |----Split2_RFC_variableImportance.txt
    |    |    |----Metric\ # 评价指标
    |    |    |    |----Test_Split2_RFC_metrics.txt
    |    |    |    |----Train_Split2_RFC_metrics.txt
    |    |    |----ROC\
    |    |    |    |----Test_Split2_RFC_roc_curve.jpg
    |    |    |    |----Test_Split2_RFC_roc_curve.pdf
    |    |    |    |----Test_Split2_RFC_roc_curve.txt
    |    |    |    |----Test_Split2_RFC_TrueAndPredict.txt
    |    |    |    |----Train_Split2_RFC_roc_curve.jpg
    |    |    |    |----Train_Split2_RFC_roc_curve.pdf
    |    |    |    |----Train_Split2_RFC_roc_curve.txt
    |    |    |    |----Train_Split2_RFC_TrueAndPredict.txt
    |    |    |----SavedModel\
    |    |    |    |----Split2_RFC.pkl
    |    |    |    |----Split2_RFC.txt
    |    |    |    |----Split2_RFCRFE.pkl
    |    |    |    |----Split2_RFCSHAP.pkl
    |    |----XGBOOST\
    |    |    |----Confusion\ # 混淆矩阵
    |    |    |    |----Test_Split2_XGBOOST_confusion_matrix.jpg
    |    |    |    |----Test_Split2_XGBOOST_confusion_matrix.pdf
    |    |    |    |----Test_Split2_XGBOOST_confusion_matrix.txt
    |    |    |    |----Train_Split2_XGBOOST_confusion_matrix.jpg
    |    |    |    |----Train_Split2_XGBOOST_confusion_matrix.pdf
    |    |    |    |----Train_Split2_XGBOOST_confusion_matrix.txt
    |    |    |----Importance\ # 特征重要性
    |    |    |    |----Split2_XGBOOST_RFEImportance.txt
    |    |    |    |----Split2_XGBOOST_SHAPImportance.txt
    |    |    |    |----Split2_XGBOOST_variableImportance.txt
    |    |    |----Metric\ # 评价指标
    |    |    |    |----Test_Split2_XGBOOST_metrics.txt
    |    |    |    |----Train_Split2_XGBOOST_metrics.txt
    |    |    |----ROC\
    |    |    |    |----Test_Split2_XGBOOST_roc_curve.jpg
    |    |    |    |----Test_Split2_XGBOOST_roc_curve.pdf
    |    |    |    |----Test_Split2_XGBOOST_roc_curve.txt
    |    |    |    |----Test_Split2_XGBOOST_TrueAndPredict.txt
    |    |    |    |----Train_Split2_XGBOOST_roc_curve.jpg
    |    |    |    |----Train_Split2_XGBOOST_roc_curve.pdf
    |    |    |    |----Train_Split2_XGBOOST_roc_curve.txt
    |    |    |    |----Train_Split2_XGBOOST_TrueAndPredict.txt
    |    |    |----SavedModel\
    |    |    |    |----Split2_XGBOOST.pkl
    |    |    |    |----Split2_XGBOOST.txt
    |    |    |    |----Split2_XGBOOSTRFE.pkl
    |    |    |    |----Split2_XGBOOSTSHAP.pkl

