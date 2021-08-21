"""
Code to fit an ann regressor to the delay data without use of feature selection.
Plots and data are saved to the results/no_feature_selection subfolder.
"""
from utility.pipelines.gb import gb_pipeline, gb_feature_pipeline
from doublePulse2017.code.setup import get_data

#%%

data = get_data()

string_data = {
    "feat_name": "Delays",
    "plot_lab": "Delay",
    "unit": "fs",
    "data_fname": "doublePulse2017/results/ex_2_gb_perf/xgb_pred.npz",
    "plot_fname": "doublePulse2017/results/ex_2_gb_perf/xgb_delays_hist2d"
}

gb_pipeline(data, string_data)

#%%


data = get_data()

string_data = {
    "feat_name": "Delays",
    "plot_lab": "Delay",
    "unit": "fs",
    "data_fname": "doublePulse2017/results/ex_2_gb_perf/xgb_10_feat_pred.npz",
    "plot_fname": "doublePulse2017/results/ex_2_gb_perf/xgb_low_delays_hist2d"
}

gb_feature_pipeline(data, string_data)
