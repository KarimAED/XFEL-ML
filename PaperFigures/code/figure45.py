import numpy as np
import matplotlib.pyplot as plt
plt.style.use("PaperFigures/code/final.mplstyle")

file_names = ["old_u1.npz", "old_u2.npz", "old_u3.npz",
              "new_u1.npz", "new_u2.npz", "new_u3.npz"]

title = "ANN"
normalised = True


fig = plt.figure(figsize=(21, 14))


for i, fname in enumerate(file_names):
    data = np.load("PaperFigures/Figure Data/Figure 4_5/%s" % fname)
    x = data["test_out"]
    y = data["test_pred"]
    ax = plt.subplot(2, 3, i+1)
    plt.tick_params(axis="x", which="both", top=False)
    mae = np.mean(np.abs(x-y))
    if normalised:
        unit = r"$\sigma$"
        mae = mae / np.std(x)
    else:
        unit = "eV"
    edges = (np.min(x), np.max(x))
    x = np.append(x, edges)
    y = np.append(y, edges)
    pos = edges
    plt.text(
        .1, .95,
        "%s; MAE=%.2f%s" % (title, mae, unit),
        transform=ax.transAxes, va="top"
    )
    plt.hist2d(x, y, cmap="Purples", bins=100, vmax=19)
    plt.plot(edges, edges, "k--")

if normalised:
    str_name = "figure5"
else:
    str_name = "figure4"

plt.savefig("PaperFigures/joined/%s.png" % str_name)

#%%