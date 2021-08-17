import matplotlib.pyplot as plt
import numpy as np
plt.style.use("./utility/plotting/styling.mplstyle")

all_feat_data = np.load("pumpProbe2021/results/ex_1_pump_pred/ann_pred.npz")
red_feat_data = np.load("pumpProbe2021/results/ex_1_pump_pred/ann_10_feat_pred.npz")

af_y = all_feat_data["test_out"]
af_p = all_feat_data["test_pred"]

rf_y = red_feat_data["test_out"]
rf_p = red_feat_data["test_pred"]


all_y = np.append(rf_y, af_y)
all_p = np.append(rf_p, af_p)

x = [np.min(all_y), np.max(all_y)]

binary_label = np.append(np.zeros(rf_y.size), np.ones(af_y.size))

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(x)
ax.set_ylim(x)
ax.set_xlabel("Expected Pump CoM")
ax.set_ylabel("Predicted Pump CoM")
scatter = ax.scatter(all_y, all_p, c=binary_label,
                     cmap="bwr", s=2, alpha=0.2)
ax.plot(x, x, "k--", label="x=y")
labels = {"0": "reduced input space", "1": "full input space"}
leg = ax.legend(scatter.legend_elements(alpha=1, prop="colors")[0], ["reduced input space", "full input space"])
ax.add_artist(leg)
