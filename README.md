# biomarker-discover
该项目利用几种常用的机器学习方法（LR, LASSO, RandomFroeset, XGBOOST,SVC_linear）来筛选标志物。它可以处理多种格式的数据，并支持多种机器学习算法和特征选择方法。

## 使用方法
* 1. 准备数据，需要准备哪些数据具体看配置文件以及配置文件解析。
* 2. 设置配置文件，如何设置配置文件具体看配置文件解析。
* 3. 运行工作流，其中 <cfg_file> 为配置文件的路径。
```
python main.py <cfg_file>
```

## 配置文件解析

**配置文件**定义了各种参数，用于运行生物标志物发现的机器学习工作流。下面列出配置文件中的主要部分及其含义：

**数据文件**

**模式1**
* `datafile`: 代谢物浓度或蛋白浓度数据文件的路径。在`模式1`下，该文件为表格型数据。该文件的第一行为样品信息和其他注释，第一列为一个代谢物或蛋白。
* `info_file`: 样品信息文件的路径。该文件必须包含名为 "Sample" 和 "Group" 的列，区分大小写。
* `marker_file`: 用于训练模型的标志物列表文件 (可选)。该文件的列名不能相同，每一列为一个panel，每一列中的元素为一个特征（代谢物或蛋白）。如果设置为 `None`，则将使用所有特征训练模型。

**模式2**
*  `datafile`: 代谢物浓度或蛋白浓度数据文件的路径。在`模式2`下，该文件为表格型数据。该文件的第一行为蛋白或代谢物编号，第一列为样本信息。最后一列为样本分组信息。
* `marker_file`: 用于训练模型的标志物列表文件 (可选)。该文件的列名不能相同，每一列为一个panel，每一列中的元素为一个特征（代谢物或蛋白）。如果设置为 `None`，则将使用所有特征训练模型。
* `test_data_file`: 测试数据集文件的路径。该文件格式与`datafile`相同。

**注意**：
* `info_file`与 `test_data_file` 只能二选一。选`info_file`表示`模式1`，选`test_data_file`表示`模式2`。

**输出目录**

* `outdir`: 结果输出目录的路径。

**比较组设置**
* `compare`: 比较组的名称设置  
每个比较组被`_vs_`分割。比如A_vs_B, 则表示两组分别为A和B。A为正常组，B为病变组。  
如果需要同时进行多组比较，比如A_vs_B_vs_C， 则表示将A、B、C三组进行比较，做3分类任务，其他任意多分类的任务以此类推。  
如果需要同时进行多个比较组的比较，比如A_vs_B;C_vs_D;...， 则表示有两个比较组，分别为A、B和C、D。A为正常组，B为病变组。C为正常组，D为病变组。每个比较组之间用`;`分割。

**预处理**

* `transform`: 代谢物数据的转换方法 (可选，默认为 "no"，不转换)。有效值包括 "log2"。
* `scale`: 代谢物数据的缩放方法 (可选，默认为 "min-max"）。有效值包括 "min-max" (最小-最大缩放)、"z-score" (标准化) 和 "no" (不缩放)。
* `category`: 分类变量的列名 (英文逗号分隔，可选)。区分大小写。
* `norminal`: 正态分布变量的列名 (英文逗号分隔，可选)。区分大小写。永远不会做 log 转换。
* `norminal_scale`: 正态分布变量的缩放方法 (可选，默认为 "none"，不缩放)。有效值同上。
* `ordinal`: 序数型变量的列名 (英文逗号分隔，可选)。

**模型**

* `method`: 机器学习算法 (英文逗号分隔)。有效值包括 "SVM" (支持向量机)、"RFC" (随机森林)、"LASSO" (LASSO 回归)、"LR" (逻辑回归)、"elasticnet" (弹性网络)、"XGBOOST" (XGBoost)。
* `k_fold`: 用于调参的 K 折交叉验证次数 (默认为 5)。
* `rfe`: 是否使用 RFE 进行特征选择 (布尔值，默认为 `True`)。
* `shap`: 是否使用 SHAP 进行特征选择 (布尔值，默认为 `True`)。
* `opti_method`: 参数寻优方法 (可选，默认为 "grid"，网格搜索)。有效值包括 "bayes" (贝叶斯优化，不推荐) 和 "grid" (网格搜索)。
* `scoring`: 模型评估指标 (字符串)。有效值包括 "roc_auc" (ROC 曲线下面积)、"accuracy" (准确率) 和 "f1" (F1 值)。
* `train_size`: 训练集所占数据的比例 (浮点数，范围为 0-1)。如果大于等于 1，则测试集和训练集相同。
* `split_times`: 划分数据集的次数 (默认为 2)，仅在模式2下生效。
* `lasso_times`: LASSO 参数寻优重复次数 (默认为 1)。
* `random_seed`: 随机种子 (用于保持可重复性，默认为 46)。
* `refit`: 是否在每次划分数据集后重新训练模型 (布尔值，默认为 `True`)。
* `n_jobs`: 并行计算使用的进程数 (默认为 20)。

**特征选择** (此部分为可选配置,暂时未完成)

* `min_features_to_select`: 最少选择的特征数量 (默认为 1)。
* `p_value`: 特征选择显著性检验的 p 值阈值 (默认为 1)。
* `adjusted_p`: 经过矫正的 p 值阈值 (默认为 1)。
* `fold_change`: 差异表达分析的 fold change 阈值 (默认为 1)。


**配置文件总结**

该配置文件定义了各种参数，涵盖了从数据加载、预处理、模型训练、评估到特征选择等整个机器学习工作流的各个步骤。用户可以根据具体数据集和需求调整这些参数。