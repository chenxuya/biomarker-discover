from src import Param, DataLoader, read_file
from src import DataGroup, DataSplit, Preprocess
from src import RFEFeatureSelector,SHAPFeatureSelector, FeatureSelectorIndividual
from src import Modeling, Testing
from src import CommonParam, add_to_dict, shape_tprfpr, shape_metric,shape_metric_dict,weight_score, interaction, batch_imp, batch_out, merge_multi_panel
from src import plot_metric, plot_roc, venn_plot
from src import annoed_readme
from os.path import join
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys

# 检查命令行参数
if len(sys.argv) != 2:
    print("Usage: python main.py <cfg_file>")
    sys.exit(1)
cfg_path = sys.argv[1]
cfgs = Param(cfg_path)

# cfgs = Param(r"/share2/users/chenxu/code/tools2/biomarker-discovery/ml.cfg")
category = None if cfgs.category==CommonParam.none else cfgs.category.split(CommonParam.semi_sep)
norminal= None if cfgs.norminal==CommonParam.none else cfgs.category.split(CommonParam.semi_sep)
rfe = cfgs.rfe
shap = cfgs.shap
gps = []
for s in cfgs.compare.split(CommonParam.semi_sep):
    gps.extend(s.split(CommonParam.vs_sep))
gps = CommonParam.vs_sep.join(gps)
if os.path.exists(cfgs.datafile) and os.path.exists(cfgs.info_file):
    dataloader = DataLoader(cfgs.datafile, cfgs.info_file,gps,
                category, norminal)
    data = dataloader.loadData()
    info = dataloader.loadInfo()
    train_sample = None
elif os.path.exists(cfgs.datafile) and os.path.exists(cfgs.test_data_file):
    train_data = read_file(cfgs.datafile)
    test_data = read_file(cfgs.test_data_file)
    data_all = pd.concat([train_data, test_data], axis=0)
    data = data_all.drop(CommonParam.group, axis=1).set_index(CommonParam.sample, drop=True).T
    info = data_all[[CommonParam.sample,CommonParam.group]].astype(str)
    train_sample = train_data[CommonParam.sample]
    cfgs.split_times = 1
elif os.path.exists(cfgs.datafile) and os.path.exists(cfgs.info_file) and os.path.exists(cfgs.test_data_file):
    raise ValueError("Please choose the specified mode. \nmode1:datafile+info_file; \nmode2:datafile+test_data_file")
else:
    raise ValueError("Please check input file")
root_out_ori = cfgs.outdir
if cfgs.marker_file != CommonParam.none:
    print('marker file is designated')
    if not os.path.exists(cfgs.marker_file):
        raise ValueError(f"no such Mark file{cfgs.marker_file}")
    else:
        marker_df = read_file(cfgs.marker_file)
else:
    marker_df = pd.DataFrame(data.index.to_list(), columns=["all_markers"])
for m in range(marker_df.shape[1]):
    marker = marker_df.iloc[:, m].dropna().to_list()
    panel_name = marker_df.columns[m]
    datagrouper = DataGroup(data.T.loc[:,marker], info)
    if marker_df.shape[1]==1:
        root_out = root_out_ori
    else:
        root_out = join(root_out_ori,panel_name)
    for compare,cur_compare_data in datagrouper.get_compares_data(cfgs.compare):
        print(f"{CommonParam.compare}:{compare}")
        dataspliter = DataSplit(cur_compare_data)
        gp = compare.split(CommonParam.vs_sep)
        labels = range(len(gp))
        amap = dict(zip(gp, labels))
        split = 0
        train_dict2 = dict()
        test_dict2 = dict()
        train_dict4 = dict()
        test_dict4 = dict()
        train_dict5 = dict()
        rfe_dict5 = dict()
        shap_dict5 = dict()
        for train, test in dataspliter.split_data(cfgs.split_times, cfgs.train_size, bygroup=True, kfold=False, train_sample=train_sample):
            split +=1
            print(f"{CommonParam.split}{split}")
            data_out = join(root_out, compare, "data")
            os.makedirs(data_out, exist_ok=True)
            train.to_csv(join(data_out, f"{CommonParam.split}{split}_{compare}_{CommonParam.train_prefix}.txt"),sep="\t", index_label=CommonParam.sample)
            test.to_csv(join(data_out, f"{CommonParam.split}{split}_{compare}_{CommonParam.test_prefix}.txt"),sep="\t", index_label=CommonParam.sample)
            preprocesser = Preprocess(train.iloc[:,:-1], test.iloc[:,:-1],
                                        train.iloc[:,-1], test.iloc[:,-1],
                                        cfgs.transform, cfgs.scale,
                                        category,norminal, cfgs.norminal_scale)
            x_train, x_test, y_train,y_test = preprocesser.process()
            y_train, y_test = y_train.map(amap), y_test.map(amap)
            for model_method in cfgs.method.split(CommonParam.semi_sep):
                cur_outdir = join(root_out, compare, f"{CommonParam.split}{split}", model_method)
                modeler = Modeling(x_train, y_train, cfgs.k_fold,
                    cfgs.scoring, cfgs.random_seed, cfgs.opti_method, cfgs.n_jobs, lasso_times=cfgs.lasso_times)
                outmodel_dir = join(cur_outdir, CommonParam.model)
                os.makedirs(outmodel_dir, exist_ok=True)
                outfile_without_suffix = join(outmodel_dir, f"{CommonParam.split}{split}_{model_method}")
                if cfgs.refit:
                    basic_model = modeler.construct(model_method)
                    modeler.save_model(basic_model, outfile_without_suffix)
                else:
                    basic_model = modeler.load_model(outfile_without_suffix + ".pkl")
                train_tester = Testing(x_train, y_train, basic_model, model_method)
                if len(y_train.unique())==2:
                    train_tester.test()
                elif len(y_train.unique())>2:
                    train_tester.multi_class_test()
                else:
                    raise ValueError("Class Number must larger than 1")
                train_res1, train_res2, train_res3, train_res4, train_res5 = train_tester.output(cur_outdir, split, CommonParam.train_prefix)
                test_tester = Testing(x_test, y_test, basic_model, model_method)
                if len(y_test.unique())==2:
                    test_tester.test(train_tester.optimal_threshold)
                elif len(y_test.unique())>2:
                    test_tester.multi_class_test()
                else:
                    raise ValueError("Class Number must larger than 1")
                test_res1, test_res2, test_res3, test_res4, test_res5 = test_tester.output(cur_outdir, split, CommonParam.test_prefix)
                if rfe:
                    RFESelector = RFEFeatureSelector(basic_model)
                    if cfgs.refit:
                        RFESelector.fit(x_train, y_train)
                        RFESelector.save_model(outfile_without_suffix+CommonParam.rfe +".pkl")
                    else:
                        RFESelector.load_model(outfile_without_suffix+CommonParam.rfe + ".pkl")
                    rfe_rank = RFESelector.rank_features()
                    add_to_dict(rfe_dict5,model_method, rfe_rank)
                    rfe_rank.to_csv(join(cur_outdir, CommonParam.importance, f"{CommonParam.split}{split}_{model_method}_{CommonParam.rfe}{CommonParam.importance}.txt"), sep="\t")
                if shap:
                    SHAPSelector = SHAPFeatureSelector(basic_model)
                    if cfgs.refit:
                        SHAPSelector.fit(x_test, y_test)
                        SHAPSelector.save_model(outfile_without_suffix+CommonParam.shap+".pkl")
                    else:
                        SHAPSelector.load_model(outfile_without_suffix+CommonParam.shap+".pkl")
                    shap_rank = SHAPSelector.rank_features()
                    add_to_dict(shap_dict5,model_method, shap_rank)
                    shap_rank.to_csv(join(cur_outdir, CommonParam.importance, f"{CommonParam.split}{split}_{model_method}_{CommonParam.shap}{CommonParam.importance}.txt"), sep="\t")

                add_to_dict(train_dict2, model_method, train_res2)
                add_to_dict(test_dict2, model_method, test_res2)
                add_to_dict(train_dict4, model_method, train_res4)
                add_to_dict(test_dict4, model_method, test_res4)
                add_to_dict(train_dict5, model_method, train_res5)
        merged_out = join(root_out, compare, CommonParam.mergedres)
        os.makedirs(merged_out, exist_ok=True)
        cur_compare_metric_train = shape_metric_dict(train_dict4)
        cur_compare_metric_test = shape_metric_dict(test_dict4)
        train_metric_fig = plot_metric(cur_compare_metric_train, None, None)
        test_metric_fig = plot_metric(cur_compare_metric_test, None, None)
        train_metric_fig.savefig(join(merged_out, f"{CommonParam.train_prefix}_{CommonParam.metric}.png"))
        test_metric_fig.savefig(join(merged_out, f"{CommonParam.test_prefix}_{CommonParam.metric}.png"))
        plt.close(train_metric_fig)
        plt.close(test_metric_fig)
        cur_compare_metric_train.to_csv(join(merged_out, f"{CommonParam.train_prefix}_{CommonParam.metric}.txt"), sep="\t", index=False)
        cur_compare_metric_test.to_csv(join(merged_out, f"{CommonParam.test_prefix}_{CommonParam.metric}.txt"), sep="\t", index=False)

        metric_ali = []
        imp_ali = []
        rfe_ali, shap_ali = [], []
        for model_method in cfgs.method.split(CommonParam.semi_sep):
            cur_model_out = join(merged_out, model_method)
            os.makedirs(cur_model_out, exist_ok=True)
            train_out = join(cur_model_out, CommonParam.train_prefix)
            test_out = join(cur_model_out, CommonParam.test_prefix)
            train_fpr, train_tpr, train_thr,train_metric, train_roc_fig = batch_out(train_out, model_method, train_dict2, train_dict4)
            test_fpr, test_tpr, test_thr,test_metric, test_roc_fig = batch_out(test_out, model_method, test_dict2, test_dict4)
            imp = batch_imp(cur_model_out, model_method, train_dict5)
            metric_ali.append(test_metric[CommonParam.mean])
            imp_ali.append(imp[CommonParam.mean])
            if rfe:
                imp_rfe = batch_imp(cur_model_out, model_method, rfe_dict5, CommonParam.rfe)
                rfe_ali.append(imp_rfe[CommonParam.mean])
            if shap:
                imp_shap = batch_imp(cur_model_out, model_method,shap_dict5, CommonParam.shap)
                shap_ali.append(imp_shap[CommonParam.mean])
        plt.close("all")
        excel_writer = pd.ExcelWriter(join(merged_out, f"{CommonParam.model}_summary.xlsx"))
        model_metric_df = pd.concat(metric_ali, axis=1)
        model_metric_df.columns = cfgs.method.split(CommonParam.semi_sep)
        model_metric_df.to_excel(excel_writer,sheet_name=CommonParam.metric)
        model_varImp_df = pd.concat(imp_ali, axis=1)
        model_varImp_df.columns = cfgs.method.split(CommonParam.semi_sep)
        model_varImp_df.to_excel(excel_writer, sheet_name=f"{CommonParam.model}_{CommonParam.importance}")
        weighted_imp = weight_score(model_metric_df, model_varImp_df, CommonParam.auc)
        weighted_imp.to_excel(excel_writer, sheet_name=f"{CommonParam.weight}_{CommonParam.model}_{CommonParam.importance}")
        feature_selector = FeatureSelectorIndividual(model_varImp_df)
        cum_ali = [0.95, 0.75]
        for cum_threshold in cum_ali:
            cumulative_percent = feature_selector.cumulative_importance_selection(cum_threshold)
            cum_intersection = interaction(cumulative_percent)
            venn_cum = venn_plot(cumulative_percent)
            venn_cum.savefig(join(merged_out, f"{CommonParam.model}_cumulative_{int(cum_threshold*100)}percent_venn.png"))
            plt.close()
            cum_intersection.to_excel(excel_writer, sheet_name=f"cumulative_{int(cum_threshold*100)}percent", index=False)
        excel_writer.close()

        if rfe:
            excel_writer = pd.ExcelWriter(join(merged_out, f"{CommonParam.rfe}_summary.xlsx"))
            model_metric_df.to_excel(excel_writer,sheet_name=CommonParam.metric)
            model_varImp_df = pd.concat(rfe_ali, axis=1)
            model_varImp_df.columns = cfgs.method.split(CommonParam.semi_sep)
            model_varImp_df.to_excel(excel_writer, sheet_name=f"{CommonParam.rfe}_{CommonParam.importance}")
            weighted_imp = weight_score(model_metric_df, model_varImp_df, CommonParam.auc)
            weighted_imp.to_excel(excel_writer, sheet_name=f"{CommonParam.weight}_{CommonParam.rfe}_{CommonParam.importance}")
            feature_selector = FeatureSelectorIndividual(model_varImp_df)
            for cum_threshold in cum_ali:
                cumulative_percent = feature_selector.cumulative_importance_selection(cum_threshold)
                cum_intersection = interaction(cumulative_percent)
                venn_cum = venn_plot(cumulative_percent)
                venn_cum.savefig(join(merged_out, f"{CommonParam.rfe}_cumulative_{int(cum_threshold*100)}percent_venn.png"))
                plt.close()
                cum_intersection.to_excel(excel_writer, sheet_name=f"cumulative_{int(cum_threshold*100)}percent", index=False)
            excel_writer.close()
            
        if shap:
            excel_writer = pd.ExcelWriter(join(merged_out, f"{CommonParam.shap}_summary.xlsx"))
            model_metric_df.to_excel(excel_writer,sheet_name=CommonParam.metric)
            model_varImp_df = pd.concat(shap_ali, axis=1)
            model_varImp_df.columns = cfgs.method.split(CommonParam.semi_sep)
            model_varImp_df.to_excel(excel_writer, sheet_name=f"{CommonParam.shap}_{CommonParam.importance}")
            weighted_imp = weight_score(model_metric_df, model_varImp_df, CommonParam.auc)
            weighted_imp.to_excel(excel_writer, sheet_name=f"{CommonParam.weight}_{CommonParam.shap}_{CommonParam.importance}")
            feature_selector = FeatureSelectorIndividual(model_varImp_df)
            for cum_threshold in cum_ali:
                cumulative_percent = feature_selector.cumulative_importance_selection(cum_threshold)
                cum_intersection = interaction(cumulative_percent)
                venn_cum = venn_plot(cumulative_percent)
                venn_cum.savefig(join(merged_out, f"{CommonParam.shap}_cumulative_{int(cum_threshold*100)}percent_venn.png"))
                plt.close()
                cum_intersection.to_excel(excel_writer, sheet_name=f"cumulative_{int(cum_threshold*100)}percent", index=False)
            excel_writer.close()

        annoed_readme(join(root_out, compare), cfgs)

# 结果汇总
metric_writer = pd.ExcelWriter(join(root_out_ori, "metric_summary.xlsx"))
if marker_df.shape[1] ==1:
    folders = glob.glob(join(root_out_ori, "**", CommonParam.mergedres))
    ali1 = [i.split("/")[-2] for i in folders]
    names = [f"{ali1[i]}" for i in range(len(ali1))]
    mer_dict = dict(zip(names, folders))
    res = merge_multi_panel(mer_dict, cfgs.split_times)
    res_mean = res.drop(columns=[CommonParam.dataset]).groupby([CommonParam.panel, CommonParam.type, CommonParam.method]).mean().reset_index()
else:
    folders = glob.glob(join(root_out_ori, "**", "**",CommonParam.mergedres))
    ali1 = [i.split("/")[-3] for i in folders]
    ali2 = [i.split("/")[-2] for i in folders]
    names = [f"{ali1[i]}__{ali2[i]}" for i in range(len(ali1))]
    mer_dict = dict(zip(names, folders))
    res = merge_multi_panel(mer_dict, cfgs.split_times)
    split_columns = res[CommonParam.panel].str.split("__", n=1, expand=True)
    # 重命名分割后的列为'Compare'和'Panel'
    split_columns.columns = [CommonParam.panel, CommonParam.compare]
    res = pd.concat([split_columns, res.drop(columns=[CommonParam.panel])], axis=1)
    res_mean = res.drop(columns=[CommonParam.dataset]).groupby([CommonParam.panel, CommonParam.compare,CommonParam.type, CommonParam.method]).mean().reset_index()
res.to_excel(metric_writer, sheet_name="metric_summary", index=False)
res_mean.to_excel(metric_writer, sheet_name="metric_summary_mean", index=False)
metric_writer.close()
