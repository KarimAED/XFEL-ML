import numpy as np
import matplotlib.pyplot as plt
plt.style.use("./utility/plotting/styling.mplstyle")


def plot_feat_hist(feats, labels):
    plt.figure(figsize=(15, 7))
    ax = plt.subplot(111)

    ax.bar([i+1 for i in range(len(labels))], np.min(feats))
    ax.set_ylabel("M")
    ax.set_xlabel("i")

    plt.show()


def plot_feat_cumulative(vals):
    plt.figure(figsize=(7, 3.5))
    ax = plt.subplot(111)
    
    ax.plot(vals)
    ax.set_ylabel("M(N)")
    ax.set_xlabel("N")
    
    plt.show()
