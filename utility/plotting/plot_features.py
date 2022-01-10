import numpy as np
import matplotlib.pyplot as plt
plt.style.use("./utility/plotting/styling.mplstyle")  # use styling file


def plot_feat_hist(feats, labels):
    """
    Function to plot the bar diagram of the top features

    :param feats: 1d-array of float, values to plot for each feature
    :param labels: 1d-array of str, label for each feature
    :return: None
    """
    plt.figure(figsize=(7, 3.5))
    ax = plt.subplot(111)

    # normalize the bar diagram
    ax.bar([i for i in range(len(labels))], feats/np.min(feats) - 1, ls="-", lw=3, alpha=0.5, color="tab:pink")
    ax.set_yscale("log")
    ax.set_ylabel(r"$I_j$")
    ax.set_xlabel("j")

    plt.show()


def plot_feat_cumulative(vals):
    """
    Function to plot a line plot of the mae when using the top x features (cumulatively) to get predictions

    :param vals: 1d-array of float, mae to plot for the top x features included
    :return: None
    """
    plt.figure(figsize=(7, 3.5))
    ax = plt.subplot(111)
    
    ax.plot(vals)
    ax.set_ylabel(r"$\mathcal{M}(M)")
    ax.set_xlabel("M")
    
    plt.show()


def plot_both(feats, labels, vals):
    """
    Function to plot both cumulative mae and individual feature importance on one plot

    :param feats: 1d-array of float, mae when scrambling one feature, used to calculate importance
    :param labels: 1d-array of str, label of features
    :param vals: 1d-array of float, mae to plot for the top x features included
    :return: None
    """

    plt.figure(figsize=(7, 3.5))

    ax1 = plt.subplot(111)

    # plot feature importance
    ax1.bar([i for i in range(len(labels))], feats/np.min(feats) - 1,
            align='center', edgecolor='black', color='grey', alpha=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$I_j$")
    ax1.set_xlabel("j")

    ax = ax1.twinx()  # on the same x-axis

    ax.plot(vals, color="b")  # plot mae of cumulative features included
    ax.tick_params(axis='y', colors='b')
    ax.yaxis.label.set_color('b')
    ax.set_ylabel(r"$\mathcal{M}$(j)")
    ax.spines['right'].set_color('b')

    plt.show()
    plt.tight_layout()
    plt.savefig("feat_sel.png")
