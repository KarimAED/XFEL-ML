from pumpProbe2021.code.setup import get_pump_data, get_probe_data
from utility.pipelines.lin import lin_feature_pipeline

undulators_2_datasets = ["u2_271_37229_events.pkl", "u2_273_37026_events.pkl", "u2_275_36614_events.pkl",
                         "u2_277_38126_events.pkl", "u2_279_37854_events.pkl"]

data = get_pump_data(undulators_2_datasets[1])

string_data = {
    "feat_name": "vls_com_pump",
    "plot_lab": "Pump CoM",
    "unit": "eV",
    "data_fname": "pumpProbe2021/results/ex_1_pump_pred/lin_10_feat_pred.npz",
    "plot_fname": "pumpProbe2021/results/ex_1_pump_pred/lin_low_pump_hist2d"
}

lin_feature_pipeline(data, string_data, legend=True, pred_lims=True)
