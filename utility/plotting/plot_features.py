import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.style.use("./utility/plotting/styling.mplstyle")


def plot_feat_hist(feats, labels):
    plt.figure(figsize=(15, 7))
    ax = plt.subplot(111)
    ax.set_xticklabels(labels, rotation=90)

    ax.bar(labels, feats)
    ax.set_ylabel("MAE with given column shuffled")
    ax.set_xlabel("Shuffled Column")

    plt.show()


def plot_feat_cumulative(vals):
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111)
    
    ax.plot(vals)
    ax.set_ylabel("MAE with top x Features")
    ax.set_xlabel("x")
    
    plt.show()
