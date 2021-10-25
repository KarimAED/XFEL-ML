from utility.plotting.scatter_diff import *

d1 = "doublePulse2017/results/ex_1_ann_feat/ann_pred.npz"
d2 = "doublePulse2017/results/ex_1_ann_feat/ann_10_feat_pred.npz"

string_d = {
    "quantity": "Delay",
    "unit": "fs",
    "label_1": r"$N_{max}$",
    "label_2": r"$N_{red}$",
}

scatter_diff(d1, d2, string_d)
