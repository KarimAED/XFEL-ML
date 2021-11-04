import numpy as np
import matplotlib.pyplot as plt
from doublePulse2017.code.setup import get_data
from pumpProbe2021.code.setup import get_pump_data
from utility.estimators import grad_boost

from utility.pipelines import ann, gb, lin

for_delay = True
n_steps = 10


#%%

if for_delay:
    selected_feats = ["ebeamEnergyBC2", "ebeamDumpCharge", "ebeamPkCurrBC1", "ebeamL3Energy",
                      "ebeamXTCAVPhase", "ebeamLTU250", "ebeamLTU450", "ebeamPhotonEnergy",
                      "AMO:R14:IOC:10:VHS5:CH3:CurrentMeasure", "AMO:R14:IOC:21:VHS7:CH0:VoltageMeasure"]
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
    data = get_pump_data("u2_273_37026_events.pkl", filter_cols=selected_feats)
    string_data = {
        "feat_name": "vls_com_pump",
        "plot_lab": r"$E_p$",
        "unit": "eV",
        "data_fname": "tmp.npz",
        "plot_fname": "tmp"
    }

x_train, x_test, y_train, y_test, input_reference, output_reference = data

step_size = x_train.shape[0] // n_steps

#%%

all_maes = []

for i in range(1, n_steps+1):
    temp_data = (x_train[:i*step_size], x_test, y_train[:i*step_size], y_test, input_reference, output_reference)

    labels = ["ann_train", "ann_test", "lin_train", "lin_test", "gb_train", "gb_test"]
    mae_i = []

    ann_est, hist = ann.ann_pipeline(temp_data, string_data)
    lin_est, feats = lin.lin_feature_pipeline(temp_data, string_data)
    gb_est = gb.gb_pipeline(temp_data, string_data)

    mae_i.append(ann_est.evaluate(x_train[:i*step_size], y_train[:i*step_size])[1])
    mae_i.append(ann_est.evaluate(x_test, y_test)[1])
    mae_i.append(grad_boost.mae(lin_est.predict(x_train[:i*step_size]), y_train[:i*step_size]))
    mae_i.append(grad_boost.mae(lin_est.predict(x_test), y_test))
    mae_i.append(grad_boost.mae(gb_est.predict(x_train[:i*step_size]), y_train[:i*step_size]))
    mae_i.append(grad_boost.mae(gb_est.predict(x_test), y_test))

    all_maes.append(mae_i)

print(all_maes)


#%%
np.savetxt("PaperFigures/Figure Data/samples_delay.csv", np.array(all_maes), delimiter=",", header=",".join(labels))

#%%

colors = ["k", "r", "b"]
plt.figure(figsize=(10,  10))
all_maes_np = np.array(all_maes).T[(0, 1, 4, 5), :]
labels_np = np.array(labels)[[0, 1, 4, 5]]
for i in range(all_maes_np.shape[0]):
    mae = all_maes_np[i]
    style = colors[i//2]
    if i % 2 == 0:
        style += "--"
    else:
        style += "-"
    plt.plot([(j+1)*step_size for j in range(len(mae))], mae, style, label=labels_np[i])

plt.legend()
plt.xlabel(r"$N_{samp}$")
plt.ylabel("MAE")
plt.show()
plt.savefig("PaperFigures/sampleConv/delay_samples.png")