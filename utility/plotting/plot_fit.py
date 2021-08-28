import numpy as np
import matplotlib.pyplot as plt
plt.style.use("./utility/plotting/styling.mplstyle")
from matplotlib.ticker import MaxNLocator


def plot_pvm(y, pred, label, x_label, y_label, save_str="", legend=True, vmax=None):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    x = np.array([np.min(y), np.max(y)])
    y = np.append(y, x)
    pred = np.append(pred, x)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.plot(x, x, "w--", label="x=y")
    if vmax is not None:
        h = ax.hist2d(y, pred, bins=np.linspace(x[0], x[1], 100), vmax=vmax)
    else:
        h = ax.hist2d(y, pred, bins=np.linspace(x[0], x[1], 100))
    print(np.max(h[0]))
    ax.legend()
    if legend:
        cbar = fig.colorbar(h[3])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(label)
    plt.show()
    if save_str:
        plt.savefig(save_str+".png")


def plot_scatter(y, pred, label, x_label, y_label):
    plt.figure()
    x = np.array([np.min(y), np.max(y)])
    plt.plot(x, x, "k--", label="x=y")

    plt.scatter(y, pred, s=0.1, alpha=0.3, label=label)
    plt.xlim(x[0], x[1])
    plt.ylim(x[0], x[1])
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_hist(hist):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(hist.history["loss"], label="Training loss")
    plt.plot(hist.history["val_loss"], label="Validation loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.plot(hist.history["mae"], label="Training MAE")
    plt.plot(hist.history["val_mae"], label="Validation MAE")
    plt.legend()
    plt.show()
