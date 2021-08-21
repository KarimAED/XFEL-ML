from utility.pipelines.ann import ann_feature_pipeline
from pumpProbe2021.code.setup import get_pump_data, get_probe_data

undulators_2_datasets = ["u2_271_37229_events.pkl", "u2_273_37026_events.pkl", "u2_275_36614_events.pkl",
                         "u2_277_38126_events.pkl", "u2_279_37854_events.pkl"]

#%%

data = get_pump_data(undulators_2_datasets[1])

string_data = {
    "feat_name": "vls_com_pump",
    "plot_lab": "Pump CoM",
    "unit": "eV",
    "data_fname": "pumpProbe2021/results/ex_1_pump_pred/ann_10_feat_pred.npz",
    "plot_fname": "pumpProbe2021/results/ex_1_pump_pred/ann_low_pump_hist2d"
}

ann_feature_pipeline(data, string_data)

#%%

pump_data = get_pump_data(undulators_2_datasets[1], include_probe=False)

pump_str_data = {
    "feat_name": "vls_com_pump",
    "plot_lab": "Pump CoM",
    "unit": "eV",
    "data_fname": "pumpProbe2021/results/ex_2_pump_probe_corr/ann_10_feat_pump_pred.npz",
    "plot_fname": "pumpProbe2021/results/ex_2_pump_probe_corr/ann_feat_pump_hist2d_no_corr"
}

ann_feature_pipeline(pump_data, pump_str_data)

#%%

probe_data = get_probe_data(undulators_2_datasets[1], include_pump=False)

probe_str_data = {
    "feat_name": "vls_com_probe",
    "plot_lab": "Probe CoM",
    "unit": "eV",
    "data_fname": "pumpProbe2021/results/ex_2_pump_probe_corr/ann_10_feat_probe_pred.npz",
    "plot_fname": "pumpProbe2021/results/ex_2_pump_probe_corr/ann_feat_probe_hist2d_no_corr"
}

ann_feature_pipeline(probe_data, probe_str_data)

#%%

u1_data = "u1_36825_events.pkl"

u1_probe_data = get_probe_data(u1_data, include_pump=False)


u1_str_data = {
    "feat_name": "vls_com_probe",
    "plot_lab": "Probe CoM",
    "unit": "eV",
    "data_fname": "pumpProbe2021/results/ex_3_undulator_vary/ann_10_feat_u1_pred.npz",
    "plot_fname": "pumpProbe2021/results/ex_3_undulator_vary/ann_feat_u1_hist2d_no_corr"
}

ann_feature_pipeline(u1_probe_data, u1_str_data)

#%%

u3_data = "u3_36610_events.pkl"

u3_probe_data = get_probe_data(u3_data, include_pump=False)


u3_str_data = {
    "feat_name": "vls_com_probe",
    "plot_lab": "Probe CoM",
    "unit": "eV",
    "data_fname": "pumpProbe2021/results/ex_3_undulator_vary/ann_10_feat_u3_pred.npz",
    "plot_fname": "pumpProbe2021/results/ex_3_undulator_vary/ann_feat_u3_hist2d_no_corr"
}

ann_feature_pipeline(u3_probe_data, u3_str_data)
