import matplotlib.pyplot as plt
import numpy as np
import time as t
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
        selected_feats = []
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
        selected_feats = []
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

#%%
plt.figure(figsize=(6, 6))
plt.xlabel("Epochs")
plt.xlim(0, 5000)
plt.ylabel(r"$\mathcal{M}$ (normalised with variance)")
plt.plot(hist.history["mae"], label=r"Training $\mathcal{M}$")
plt.plot(hist.history["val_mae"], label=r"Validation $\mathcal{M}$")
plt.hlines(np.min(hist.history["mae"]), 0, 5000, color="k", linestyle="--")
plt.legend()
plt.show()
plt.savefig("PaperFigures/ec_pulse_1_full.pdf")
