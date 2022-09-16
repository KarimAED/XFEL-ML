import numpy as np
import matplotlib.pyplot as plt
from oldMode2017.setup import get_data
from newMode2021.setup import get_data_p1
from utility.helpers import mae


from utility.pipelines import ann, gb, lin

FOR_DELAY = False
N_STEPS = 10

if FOR_DELAY:
    DS_NAME = "delay"
else:
    DS_NAME = "pulse1"

SAVE_NAME = "PaperFigures/Figure Data/samples_" + DS_NAME + ".csv"
FIG_NAME = "PaperFigures/sampleConv/" + DS_NAME + "_samples.pdf"


#%%

if FOR_DELAY:
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
    data = get_data_p1("u2_273_37026_events.pkl", filter_cols=selected_feats)
    string_data = {
        "feat_name": "vls_com_pump",
        "plot_lab": r"$E_p$",
        "unit": "eV",
        "data_fname": "tmp.npz",
        "plot_fname": "tmp",
    }

x_train, x_test, y_train, y_test, input_reference, output_reference = data

step_size = x_train.shape[0] // N_STEPS

#%%

all_maes = []

for i in range(1, N_STEPS + 1):
    temp_data = (
        x_train[: i * step_size],
        x_test,
        y_train[: i * step_size],
        y_test,
        input_reference,
        output_reference,
    )

    labels = [
        "ann_train",
        "ann_test",
        "lin_train",
        "lin_test",
        "gb_train",
        "gb_test",
    ]
    mae_i = []

    ann_est, hist = ann.ann_pipeline(temp_data, string_data)
    lin_est, feats = lin.lin_feature_pipeline(temp_data, string_data)
    gb_est = gb.gb_pipeline(temp_data, string_data)

    mae_i.append(
        ann_est.evaluate(x_train[: i * step_size], y_train[: i * step_size])[1]
    )
    mae_i.append(ann_est.evaluate(x_test, y_test)[1])
    mae_i.append(
        mae(
            lin_est.predict(x_train[: i * step_size]), y_train[: i * step_size]
        )
    )
    mae_i.append(mae(lin_est.predict(x_test), y_test))
    mae_i.append(
        mae(gb_est.predict(x_train[: i * step_size]), y_train[: i * step_size])
    )
    mae_i.append(mae(gb_est.predict(x_test), y_test))

    all_maes.append(mae_i)

print(all_maes)


#%%
np.savetxt(
    SAVE_NAME, np.array(all_maes), delimiter=",", header=",".join(labels)
)
