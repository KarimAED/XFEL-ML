from utility.pipelines.ann import ann_feature_pipeline
from pumpProbe2021.code.setup import get_probe_data

different_undulators_datasets = ["LW8_Run_62_40610_events_1_undulators_210827.pkl",
                                 "LW8_Run_52_33950_events_2_undulators_210827.pkl",
                                 "LW8_Run_37_50467_events_3_undulators_210827.pkl",
                                 "LW8_Run_78_30000_events_4_undulators_210827.pkl",
                                 "LW8_Run_48_33170_events_5_undulators_210827.pkl"]


#%%

for i, label in enumerate(different_undulators_datasets):
    data = get_probe_data(different_undulators_datasets[i], include_pump=False)

    string_data = {
        "feat_name": "vls_com_probe",
        "plot_lab": "central probe energy",
        "unit": "eV",
        "data_fname": "PaperFigures/Figure Data/Figure 4_5/new_u%s.npz" % str(i+1),
        "plot_fname": "pumpProbe2021/results/ex_3_undulator_vary/probe_undulator_%s_hist2d" % str(i+1)
    }

    ann_feature_pipeline(data, string_data, legend=False, vmax=19)
