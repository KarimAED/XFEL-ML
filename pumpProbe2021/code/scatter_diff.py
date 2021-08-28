from utility.plotting.scatter_diff import *

d1 = "pumpProbe2021/results/ex_1_pump_pred/ann_pred.npz"
d2 = "pumpProbe2021/results/ex_1_pump_pred/ann_10_feat_pred.npz"

string_d = {
    "quantity": "Pump CoM",
    "unit": "eV",
    "label_1": "full input space",
    "label_2": "reduced input space",
}

scatter_diff(d1, d2, string_d)
