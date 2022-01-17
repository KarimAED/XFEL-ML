from utility.pipelines.ann import ann_feature_pipeline
from newMode2021.code.setup import get_pump_data, get_probe_data

undulators_2_datasets = ["u2_271_37229_events.pkl", "u2_273_37026_events.pkl", "u2_275_36614_events.pkl",
                         "u2_277_38126_events.pkl", "u2_279_37854_events.pkl"]

#%%

# Pump Prediction with ANN and probe in input (2 undulators)

data = get_pump_data(undulators_2_datasets[1])

string_data = {
    "feat_name": "vls_com_pump",
    "plot_lab": "central pump energy",
    "unit": "eV",
    "data_fname": "PaperFigures/Figure Data/Figure 3/ann_pump.npz",
    "plot_fname": "newMode2021/results/ex_1_pump_pred/ann_low_pump_hist2d"
}

ann_feature_pipeline(data, string_data, vmax=19, legend=False, noRefit=True)

#%%

# pump prediction with ANN and without probe in input (1 undulator)

u1_data = "u1_36825_events.pkl"

u1_probe_data = get_pump_data(u1_data)


u1_str_data = {
    "feat_name": "vls_com_pump",
    "plot_lab": "central pump energy",
    "unit": "eV",
    "data_fname": "newMode2021/results/ex_3_undulator_vary/ann_10_feat_u1_pump.npz",
    "plot_fname": "newMode2021/results/ex_3_undulator_vary/ann_feat_u1_pump_hist2d_no_corr"
}

ann_feature_pipeline(u1_probe_data, u1_str_data, vmax=19, legend=False)

#%%

# pump prediction with ANN and without probe in input (3 undulators)

u3_data = "u3_36610_events.pkl"

u3_probe_data = get_pump_data(u3_data)


u3_str_data = {
    "feat_name": "vls_com_pump",
    "plot_lab": "central pump energy",
    "unit": "eV",
    "data_fname": "newMode2021/results/ex_3_undulator_vary/ann_10_feat_u3_pump.npz",
    "plot_fname": "newMode2021/results/ex_3_undulator_vary/ann_feat_u3_pump_hist2d_no_corr"
}

ann_feature_pipeline(u3_probe_data, u3_str_data, vmax=19, legend=False)
