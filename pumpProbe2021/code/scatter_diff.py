from utility.plotting.scatter_diff import *

d1 = "pumpProbe2021/results/ex_1_pump_pred/ann_pred.npz"
d2 = "pumpProbe2021/results/ex_1_pump_pred/ann_10_feat_pred.npz"

string_d = {
    "quantity": "central pump energy",
    "unit": "eV",
    "label_1": r"$N_{max}$",
    "label_2": r"$N_{red}$",
}

scatter_diff(d1, d2, string_d)
