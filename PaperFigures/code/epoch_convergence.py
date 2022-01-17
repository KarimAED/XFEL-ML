import matplotlib.pyplot as plt
import time as t
from oldMode2017.setup import get_data
from newMode2021.code.setup import get_pump_data

from utility.pipelines import ann

for_delay = True
filter_cols = True

#%%

if for_delay:
    selected_feats = ["ebeamEnergyBC2", "ebeamDumpCharge", "ebeamPkCurrBC1", "ebeamL3Energy",
                      "ebeamXTCAVPhase", "ebeamLTU250", "ebeamLTU450", "ebeamPhotonEnergy",
                      "AMO:R14:IOC:10:VHS5:CH3:CurrentMeasure", "AMO:R14:IOC:21:VHS7:CH0:VoltageMeasure"]
    if not filter_cols:
        selected_feats = []
    data = get_data(filter_cols=selected_feats)
    string_data = {
        "feat_name": "Delays",
        "plot_lab": r"$T_P$",
        "unit": "fs",
        "data_fname": "tmp.npz",
        "plot_fname": "tmp"
    }
else:
    selected_feats = ["vls_com_probe", "xgmd_rmsElectronSum", "xgmd_energy", "ebeam_ebeamL3Energy",
                      "gmd_energy", "ebeam_ebeamUndPosX", "vls_width_probe", "ebeam_ebeamUndAngY",
                      "ebeam_ebeamUndPosY", "ebeam_ebeamLTU450"]
    if not filter_cols:
        selected_feats = []
    data = get_pump_data("u2_273_37026_events.pkl", filter_cols=selected_feats)
    string_data = {
        "feat_name": "vls_com_pump",
        "plot_lab": r"$E_p$",
        "unit": "eV",
        "data_fname": "tmp.npz",
        "plot_fname": "tmp"
    }

#%%
s = t.time()
ann_est, hist = ann.ann_pipeline(data, string_data)
e = t.time()

print("Fitting duration: %.2fs" % (e-s))

#%%
import matplotlib
matplotlib.style.use("./utility/plotting/styling.mplstyle")

plt.figure(figsize=(10, 10))
plt.xlabel("Epochs")
plt.ylabel(r"$\mathcal{M}$ (normalised with variance)")
plt.plot(hist.history["mae"], label=r"Training $\mathcal{M}$")
plt.plot(hist.history["val_mae"], label=r"Validation $\mathcal{M}$")
plt.legend()
plt.show()
plt.savefig("PaperFigures/pump_allFeats.png")


