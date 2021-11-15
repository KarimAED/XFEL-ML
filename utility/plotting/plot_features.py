import numpy as np
import matplotlib.pyplot as plt
plt.style.use("./utility/plotting/styling.mplstyle")


def plot_feat_hist(feats, labels):
    plt.figure(figsize=(7, 3.5))
    ax = plt.subplot(111)

    ax.bar([i for i in range(len(labels))], feats/np.min(feats) - 1, ls="-", lw=3, alpha=0.5, color="tab:pink")
    ax.set_yscale("log")
    ax.set_ylabel(r"$I_j$")
    ax.set_xlabel("j")

    plt.show()


def plot_feat_cumulative(vals):
    plt.figure(figsize=(7, 3.5))
    ax = plt.subplot(111)
    
    ax.plot(vals)
    ax.set_ylabel(r"$\mathcal{M}(M)")
    ax.set_xlabel("M")
    
    plt.show()


def plot_both(feats, labels, vals):

    plt.figure(figsize=(7, 3.5))

    ax1 = plt.subplot(111)

    ax1.bar([i for i in range(len(labels))], feats/np.min(feats) - 1,
            align='center', edgecolor='black', color='grey', alpha=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$I_j$")
    ax1.set_xlabel("j")

    ax = ax1.twinx()

    ax.plot(vals, color="b")
    ax.tick_params(axis='y', colors='b')
    ax.yaxis.label.set_color('b')
    ax.set_ylabel(r"$\mathcal{M}$(j)")
    ax.spines['right'].set_color('b')

    plt.show()
    plt.tight_layout()
    plt.savefig("feat_sel.png")
