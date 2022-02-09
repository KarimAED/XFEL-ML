from deprecated import deprecated

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("./utility/plotting/styling.mplstyle")
from matplotlib.ticker import MaxNLocator


def plot_pvm(
    y,
    pred,
    label,
    x_label,
    y_label,
    save_str="",
    legend=True,
    vmax=None,
    pred_lims=False,
):
    """
    Function to plot 2d-histogram of measurements (labels, x-axis) and predictions (y-axis)

    :param y: 1d-array of float, labels
    :param pred: 1d-array of float, predictions
    :param label: str, legend title to use
    :param x_label: str, x-axis label
    :param y_label: str, y-axis label
    :param save_str: str, filename for saving the figure
    :param legend: bool, if a cbar legend should be shown
    :param vmax: int, if a maximum value for the colormap is desired (for uniform figure look)
    :param pred_lims: bool, if limits of prediction instead of labels should be used on y-axis
    :return: None
    """
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    edge_x = np.array([np.min(y), np.max(y)])
    edge_y = np.array([np.min(pred), np.max(pred)])
    y = np.append(y, edge_x)
    if pred_lims:
        ax.ticklabel_format(
            style="scientific", axis="y", scilimits=(0, 0), useMathText=True
        )
        p_lim = edge_y
    else:
        p_lim = edge_x
    pred = np.append(pred, p_lim)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.plot(edge_x, edge_x, "w--")
    bins = [
        np.linspace(edge_x[0], edge_x[1], 100),
        np.linspace(p_lim[0], p_lim[1], 100),
    ]
    if vmax is not None:
        h = ax.hist2d(y, pred, bins=bins, vmax=vmax)
    else:
        h = ax.hist2d(y, pred, bins=bins)
    print(np.max(h[0]))
    if legend:
        cbar = fig.colorbar(h[3])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(label)
    plt.show()
    if save_str:
        plt.savefig(save_str + ".png")


@deprecated("Use 2d-histogram instead")
def plot_scatter(y, pred, label, x_label, y_label):
    """
    More primitive scatter plot to illustrate predictions vs measurements, deprecated.

    :param y: 1d-array of float, labels
    :param pred: 1d-array of float, predictions
    :param label: str, legend title to use
    :param x_label: str, x-axis label
    :param y_label: str, y-axis label
    :return: None
    """
    plt.figure()
    x = np.array([np.min(y), np.max(y)])
    plt.plot(x, x, "k--")

    plt.scatter(y, pred, s=0.1, alpha=0.3, label=label)
    plt.xlim(x[0], x[1])
    plt.ylim(x[0], x[1])
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_hist(hist):
    """
    Plots the history of the fitting across epochs(for ann)

    :param hist: tf.keras.History object
    :return: None
    """
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(hist.history["loss"], label="Training loss")  # plot loss
    plt.plot(
        hist.history["val_loss"], label="Validation loss"
    )  # plot loss on validation
    plt.legend()
    plt.show()

    # we plot another figure for mae
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.plot(hist.history["mae"], label="Training MAE")  # plot mae
    plt.plot(
        hist.history["val_mae"], label="Validation MAE"
    )  # plot validation mae
    plt.legend()
    plt.show()
