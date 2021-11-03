from utility.pipelines.ann import ann_feature_pipeline
from pumpProbe2021.code.setup import get_probe_data

undulators_2_datasets = ["u2_271_37229_events.pkl", "u2_273_37026_events.pkl", "u2_275_36614_events.pkl",
                         "u2_277_38126_events.pkl", "u2_279_37854_events.pkl"]

#%%

# Pump Prediction with ANN and probe in input (2 undulators)

data = get_probe_data(undulators_2_datasets[1])

string_data = {
    "feat_name": "vls_com_probe",
    "plot_lab": "central probe energy",
    "unit": "eV",
    "data_fname": "PaperFigures/Figure Data/Figure 4_5/old_u2.npz",
    "plot_fname": "pumpProbe2021/results/ex_4_probe_pred/ann_probe_wo_pump_2_no_corr"
}

ann_feature_pipeline(data, string_data, legend=False, vmax=19)

#%%

# Pump Prediction with ANN and probe in input (1 undulator)

data = get_probe_data("u1_36825_events.pkl")

string_data = {
    "feat_name": "vls_com_probe",
    "plot_lab": "central probe energy",
    "unit": "eV",
    "data_fname": "PaperFigures/Figure Data/Figure 4_5/old_u1.npz",
    "plot_fname": "pumpProbe2021/results/ex_3_undulator_vary/ann_probe_wo_pump_1_no_corr"
}

ann_feature_pipeline(data, string_data, legend=False, vmax=19)

#%%

# Pump Prediction with ANN and probe in input (3 undulators)

data = get_probe_data("u3_36610_events.pkl")

string_data = {
    "feat_name": "vls_com_probe",
    "plot_lab": "central probe energy",
    "unit": "eV",
    "data_fname": "PaperFigures/Figure Data/Figure 4_5/old_u3.npz",
    "plot_fname": "pumpProbe2021/results/ex_3_undulator_vary/ann_probe_wo_pump_3_no_corr"
}

ann_feature_pipeline(data, string_data, legend=False, vmax=19)
