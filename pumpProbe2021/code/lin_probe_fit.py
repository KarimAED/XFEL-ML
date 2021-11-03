from utility.pipelines.lin import lin_feature_pipeline
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
    "data_fname": "PaperFigures/Figure Data/Figure 3/lin_probe.npz",
    "plot_fname": "pumpProbe2021/results/ex_4_probe_pred/lin_low_probe_hist2d"
}

lin_feature_pipeline(data, string_data, pred_lims=True, legend=False, vmax=19)
