import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

# create colormap
# ---------------

# create a colormap that consists of
# - 1/5 : custom colormap, ranging from white to the first color of the colormap
# - 4/5 : existing colormap

# set upper part: 4 * 256/4 entries
upper = mpl.cm.Blues(np.arange(256)[256//3:])

# set lower part: 1 * 256/4 entries
# - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
lower = np.ones((int(256/4), 4))
# - modify the first three columns (RGB):
#   range linearly between white (1,1,1) and the first color of the upper colormap
for i in range(3):
  lower[:, i] = np.linspace(1, upper[0, i], lower.shape[0])

# combine parts of colormap
cmap = np.vstack((lower, upper))

# convert to matplotlib colormap
cmap = mpl.colors.ListedColormap(cmap, name='myBlues', N=cmap.shape[0])

#%%
plt.style.use("PaperFigures/code/final.mplstyle")

legend = False

file_names = ["lin_delay.npz", "lin_pump.npz", "lin_probe.npz",
              "gb_delay.npz", "gb_pump.npz", "gb_probe.npz",
              "ann_delay.npz", "ann_pump.npz", "ann_probe.npz"]

titles = ["LIN", "GB", "ANN"]
units = ["fs", "eV", "eV"]


edges = []

for i, fname in enumerate(file_names):
    data = np.load("PaperFigures/Figure Data/Figure 3/%s" % fname)
    x = data["test_out"]
    if len(edges) <= (i % 3):
        edges.append([np.min(x), np.max(x)])
    elif np.min(x) < edges[i % 3][0]:
        edges[i % 3][0] = np.min(x)
    elif np.max(x) > edges[i % 3][1]:
        edges[i % 3][1] = np.max(x)

fig = plt.figure(figsize=(20, 20))


for i, fname in enumerate(file_names):
    data = np.load("PaperFigures/Figure Data/Figure 3/%s" % fname)
    x = data["test_out"]
    y = data["test_pred"]
    ax = plt.subplot(3, 3, i+1)
    if i//3 == 0:
        plt.tick_params(axis="x", which="both", top=False)
    else:
        plt.tick_params(axis="x", which="both", top=True)
    if not (i//3 == 2):
        ax.set_xticklabels([])
    unit = units[i % 3]
    mae = np.mean(np.abs(x-y))
    e = edges[i % 3]
    if mae > 1000:
        pred_edges = (np.min(y), np.max(y))
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0), useMathText=True)
        mae = mae/1000
        unit = "keV"
    else:
        pred_edges = ()
        x = np.append(x, e)
        y = np.append(y, e)
    plt.text(
        .1, .95,
        r"%s; $\mathcal{M}=%.2f%s$" % (titles[i//3], mae, unit),
        transform=ax.transAxes, va="top"
    )

    hist = plt.hist2d(x, y, cmap=cmap, bins=50, vmax=45)
    if i == 8 and legend:
        plt.colorbar(ax=ax)
    plt.plot(edges, edges, "k--")

plt.savefig("PaperFigures/joined/figure3.png")

#%%

