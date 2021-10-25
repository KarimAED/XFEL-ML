from doublePulse2017.code.setup import get_data
from utility.pipelines.lin import lin_feature_pipeline

data = get_data()

string_data = {
    "feat_name": "Delays",
    "plot_lab": "Delay",
    "unit": "fs",
    "data_fname": "doublePulse2017/results/ex_2_gb_perf/lin_10_feat_pred.npz",
    "plot_fname": "doublePulse2017/results/ex_2_gb_perf/lin_low_delays_hist2d"
}

lin_feature_pipeline(data, string_data, vmax=19, legend=True)
