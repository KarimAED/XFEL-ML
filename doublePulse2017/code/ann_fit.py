"""
Code to fit an ann regressor to the delay data without use of feature selection.
Plots and data are saved to the results/no_feature_selection subfolder.
"""
from doublePulse2017.code.setup import get_data
from utility.pipelines.ann import ann_pipeline, ann_feature_pipeline

#%%

data = get_data()

string_data = {
    "feat_name": "Delays",
    "plot_lab": "Delay",
    "unit": "fs",
    "data_fname": "doublePulse2017/results/ex_1_ann_feat/ann_pred.npz",
    "plot_fname": "doublePulse2017/results/ex_1_ann_feat/ann_hist2d"
}

ann_pipeline(data, string_data)

#%%


data = get_data()

string_data = {
    "feat_name": "Delays",
    "plot_lab": "Delay",
    "unit": "fs",
    "data_fname": "doublePulse2017/results/ex_1_ann_feat/ann_10_feat_pred.npz",
    "plot_fname": "doublePulse2017/results/ex_1_ann_feat/ann_low_delays_hist2d"
}

ann_feature_pipeline(data, string_data)
