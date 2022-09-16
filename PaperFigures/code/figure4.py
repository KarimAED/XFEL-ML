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
new_lin_names = [
    "new_lin_u1.npz",
    "new_lin_u2.npz",
    "new_lin_u3.npz",
    "new_lin_u4.npz",
    "new_lin_u5.npz"
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


fig = plt.figure(figsize=(14, 7))

old_mae = [get_mae(i) for i in old_names]
new_mae = [get_mae(i) for i in new_names]
new_lin_mae = [get_mae(i) for i in new_lin_names]
x_axis = [i + 1 for i in range(len(new_names))]

plt.plot(
    x_axis[:3],
    old_mae,
    c="b",
    ls="-",
    marker="o",
    label="ANN - pulse 1 and 2",
)
plt.plot(
    x_axis, new_mae, c="g", ls="--", marker="d", label="ANN - pulse 2 only"
)
plt.plot(x_axis, new_lin_mae, c="m", ls="-.", marker="$\\bigotimes$",
         markersize=13, label="LIN - pulse 2 only")

plt.xlabel("number of undulators between pulses")
plt.xticks(list(range(1, 6)), labels=list(range(1, 6)))

if normalised:
    unit = r"$\sigma$"
else:
    unit = "eV"
plt.ylabel(r"$\mathcal{M}$" + f" ({unit})")

plt.gca().legend()
plt.show()

str_name = "figure4"

plt.tight_layout()
plt.subplots_adjust(
    left=0.13, bottom=0.15, right=0.99, top=0.87, wspace=0.4, hspace=0.1
)
plt.savefig("PaperFigures/joined/%s.pdf" % str_name)

#%%
