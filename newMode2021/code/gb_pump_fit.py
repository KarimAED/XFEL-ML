from newMode2021.code.setup import get_pump_data, get_probe_data
from utility.pipelines.gb import gb_feature_pipeline

undulators_2_datasets = ["u2_271_37229_events.pkl", "u2_273_37026_events.pkl", "u2_275_36614_events.pkl",
                         "u2_277_38126_events.pkl", "u2_279_37854_events.pkl"]

#%%

event_counts = []

# checking usable event counts for each dataset
for ds in undulators_2_datasets:
    data_list = get_pump_data(ds)
    events = data_list[0].shape[0]+data_list[1].shape[0]

    # to demonstrate proper normalisation
    # print(np.std(data_list[0], axis=0).tolist())
    # print(np.mean(data_list[0], axis=0).tolist())

    event_counts.append(events)

print(event_counts)

# u2_273_37026_events.pkl has the most usable events at 20632 and index 1

#%%

data = get_pump_data(undulators_2_datasets[1])

string_data = {
    "feat_name": "vls_com_pump",
    "plot_lab": "central pump energy",
    "unit": "eV",
    "data_fname": "PaperFigures/Figure Data/Figure 3/gb_pump.npz",
    "plot_fname": "newMode2021/results/ex_1_pump_pred/xgb_low_pump_hist2d"
}

gb_feature_pipeline(data, string_data, vmax=19, legend=False)

#%%

pump_data = get_pump_data(undulators_2_datasets[1], include_probe=False)

pump_str_data = {
    "feat_name": "vls_com_pump",
    "plot_lab": "Pump CoM",
    "unit": "eV",
    "data_fname": "newMode2021/results/ex_2_pump_probe_corr/xgb_10_feat_pump_pred.npz",
    "plot_fname": "newMode2021/results/ex_2_pump_probe_corr/xgb_feat_pump_hist2d_no_corr"
}

gb_feature_pipeline(pump_data, pump_str_data)

#%%

probe_data = get_probe_data(undulators_2_datasets[1], include_pump=False)

probe_str_data = {
    "feat_name": "vls_com_probe",
    "plot_lab": "Probe CoM",
    "unit": "eV",
    "data_fname": "newMode2021/results/ex_2_pump_probe_corr/xgb_10_feat_probe_pred.npz",
    "plot_fname": "newMode2021/results/ex_2_pump_probe_corr/xgb_feat_probe_hist2d_no_corr"
}

gb_feature_pipeline(probe_data, probe_str_data)
