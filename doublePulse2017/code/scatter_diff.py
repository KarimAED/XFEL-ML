import matplotlib.pyplot as plt
import numpy as np
plt.style.use("./utility/plotting/styling.mplstyle")

all_feat_data = np.load("doublePulse2017/results/ex_1_ann_feat/ann_pred.npz")
red_feat_data = np.load("doublePulse2017/results/ex_1_ann_feat/ann_10_feat_pred.npz")

af_y = all_feat_data["test_out"]
af_p = all_feat_data["test_pred"]

rf_y = red_feat_data["test_out"]
rf_p = red_feat_data["test_pred"]

plt.figure(figsize=(7, 7))

plt.scatter(rf_y, rf_p, s=1, alpha=0.2, label="reduced input space")
plt.scatter(af_y, af_p, s=1, alpha=0.2, label="full input space")
plt.plot([-20, 25], [-20, 25], "k--", label="x=y")
plt.legend()
plt.xlim(-20, 25)
plt.ylim(-20, 25)

plt.xlabel("measured delay in fs")
plt.ylabel("predicted delay in fs")
