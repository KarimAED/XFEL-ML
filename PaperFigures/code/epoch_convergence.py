import time as t
import matplotlib.pyplot as plt
import numpy as np
from oldMode2017.setup import get_data
from newMode2021.setup import get_data_p1

from utility.pipelines import ann

for_delay = False
filter_cols = False

#%%

if for_delay:
    selected_feats = [
        "ebeamEnergyBC2",
        "ebeamDumpCharge",
        "ebeamPkCurrBC1",
        "ebeamL3Energy",
        "ebeamXTCAVPhase",
        "ebeamLTU250",
        "ebeamLTU450",
        "ebeamPhotonEnergy",
        "AMO:R14:IOC:10:VHS5:CH3:CurrentMeasure",
        "AMO:R14:IOC:21:VHS7:CH0:VoltageMeasure",
    ]
    if not filter_cols:
        selected_feats = None
    data = get_data(filter_cols=selected_feats)
    string_data = {
        "feat_name": "Delays",
        "plot_lab": r"$T_P$",
        "unit": "fs",
        "data_fname": "tmp.npz",
        "plot_fname": "tmp",
    }
else:
    selected_feats = [
        "vls_com_probe",
        "xgmd_rmsElectronSum",
        "xgmd_energy",
        "ebeam_ebeamL3Energy",
        "gmd_energy",
        "ebeam_ebeamUndPosX",
        "vls_width_probe",
        "ebeam_ebeamUndAngY",
        "ebeam_ebeamUndPosY",
        "ebeam_ebeamLTU450",
    ]
    if not filter_cols:
        selected_feats = None
    data = get_data_p1("u2_273_37026_events.pkl", filter_cols=selected_feats)
    string_data = {
        "feat_name": "vls_com_pump",
        "plot_lab": r"$E_p$",
        "unit": "eV",
        "data_fname": "tmp.npz",
        "plot_fname": "tmp",
    }

#%%
s = t.time()
ann_est, hist = ann.ann_pipeline(data, string_data)
e = t.time()

print("Fitting duration: %.2fs" % (e - s))
0  #%%

if filter_cols:
    hist_red = hist
    mae = np.array(hist_red.history["mae"])
    val_mae = np.array(hist_red.history["val_mae"])
    mae_100 = mae * 100
else:
    hist_full = hist
    full_mae = np.array(hist_full.history["mae"])
    full_val_mae = np.array(hist_full.history["val_mae"])

#%%
plt.figure(figsize=(10, 10))

plt.subplots_adjust(
    left=0.17, bottom=0.12, right=0.99, top=0.99, wspace=0.4, hspace=0.1
)
plt.savefig("PaperFigures/ec_pulse1.pdf")
