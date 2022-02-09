import numpy as np
import matplotlib.pyplot as plt

plt.style.use("utility/plotting/styling.mplstyle")

old_names = ["old_u1.npz", "old_u2.npz", "old_u3.npz"]
new_names = [
    "new_u1.npz",
    "new_u2.npz",
    "new_u3.npz",
    "new_u4.npz",
    "new_u5.npz",
]

normalised = True
#%%


def get_mae(fname):
    data = np.load("PaperFigures/Figure Data/Figure 4_5/%s" % fname)
    x = data["test_out"]
    y = data["test_pred"]
    mae = np.mean(np.abs(x - y))
    if normalised:
        mae = mae / np.std(x)
    return mae


#%%


fig = plt.figure(figsize=(7, 3.5))

old_mae = [get_mae(i) for i in old_names]
new_mae = [get_mae(i) for i in new_names]
x_axis = [i + 1 for i in range(len(new_names))]

plt.plot(x_axis[:3], old_mae, c="b", ls="-", marker="o")
plt.plot(x_axis, new_mae, c="g", ls="--", marker="d")

plt.xlabel("u")

if normalised:
    unit = r"\sigma"
else:
    unit = "eV"
plt.ylabel(r"$\mathcal{M}$ in $%s$" % unit)

plt.show()

str_name = "figure4"

plt.savefig("PaperFigures/joined/%s.png" % str_name)

#%%
