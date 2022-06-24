import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# create colormap
# ---------------

# create a colormap that consists of
# - 1/5 : custom colormap, ranging from white to the first color of the colormap
# - 4/5 : existing colormap

# set upper part: 4 * 256/4 entries
upper = mpl.cm.Blues(np.arange(256)[256 // 3 :])

# set lower part: 1 * 256/4 entries
# - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
lower = np.ones((int(256 / 4), 4))
# - modify the first three columns (RGB):
#   range linearly between white (1,1,1) and the first color of the upper colormap
for i in range(3):
    lower[:, i] = np.linspace(1, upper[0, i], lower.shape[0])

# combine parts of colormap
cmap = np.vstack((lower, upper))

# convert to matplotlib colormap
cmap = mpl.colors.ListedColormap(cmap, name="myBlues", N=cmap.shape[0])

#%%

LEGEND = False

file_names = [
    "lin_pump.npz",
    "lin_probe.npz",
    "gb_pump.npz",
    "gb_probe.npz",
    "ann_pump.npz",
    "ann_probe.npz",
]

titles = ["LIN", "GB", "ANN"]
units = [r"\sigma", r"\sigma"]


edges = []

for i, fname in enumerate(file_names):
    data = np.load("PaperFigures/Figure Data/Figure 3/%s" % fname)
    x = (data["test_out"] - np.mean(data["train_out"])) / np.std(data["train_out"])
    if len(edges) <= (i % 2):
        edges.append([np.min(x), np.max(x)])
    elif np.min(x) < edges[i % 2][0]:
        edges[i % 2][0] = np.min(x)
    elif np.max(x) > edges[i % 2][1]:
        edges[i % 2][1] = np.max(x)

fig = plt.figure(figsize=(14, 14))


for i, fname in enumerate(file_names):
    data = np.load("PaperFigures/Figure Data/Figure 3/%s" % fname)
    x = data["test_out"]
    y = data["test_pred"]
    x_mean = np.mean(data["train_out"])
    x_std = np.std(data["train_out"])
    x = (x - x_mean) / x_std
    y = (y - x_mean) / x_std
    ax = plt.subplot(3, 2, i + 1)
    if i // 2 == 0:
        plt.tick_params(axis="x", which="both", top=False)
    else:
        plt.tick_params(axis="x", which="both", top=True)
    if not i // 2 == 2:
        ax.set_xticklabels([])
    else:
        if i % 2 == 0:
            ax.set_xlabel(r"Measured $E_1(\sigma)$")
        else:
            ax.set_xlabel(r"Measured $E_2(\sigma)$")
    # ax.yaxis.set_major_formatter(lambda x, pos: str(int(x)))
    unit = units[i % 2]
    mae = np.mean(np.abs(x - y))
    e = edges[i % 2]
    if mae > 1000:
        pred_edges = (np.min(y), np.max(y))
        ax.ticklabel_format(
            style="scientific", axis="y", scilimits=(0, 0), useMathText=True
        )
        # mae = mae / 1000
    else:
        pred_edges = ()
        x = np.append(x, e)
        y = np.append(y, e)
    if i % 2 == 0:
        ax.set_ylabel(r"Predicted $E_1(%s)$" % unit)
    else:
        ax.set_ylabel(r"Predicted $E_2(%s)$" % unit)
    plt.text(
        0.1,
        0.95,
        r"%s; $\mathcal{M}=%.2f%s$" % (titles[i // 2], mae, unit),
        transform=ax.transAxes,
        va="top",
    )
    plt.locator_params(axis="x", nbins=4)
    hist = plt.hist2d(x, y, cmap=cmap, bins=50, vmax=45)
    if i == 8 and LEGEND:
        plt.colorbar(ax=ax)
    plt.plot((np.min(x), np.max(x)), (np.min(x), np.max(x)), "k--")

plt.tight_layout()
plt.subplots_adjust(
    left=0.15, bottom=0.1, right=0.95, top=0.95, wspace=0.4, hspace=0.1
)
plt.savefig("PaperFigures/joined/figure3.pdf")

#%%
