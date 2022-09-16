import matplotlib.pyplot as plt
import numpy as np

def plot_sample_convergence(labels, all_maes, x_max, n_steps=10):
    step_size = x_max // n_steps
    colors = ["k", "r", "b"]
    all_maes_np = np.array(all_maes).T[(0, 1, 4, 5), :]
    labels_np = np.array(labels)[[0, 1, 4, 5]]
    for i in range(all_maes_np.shape[0]):
        mae = all_maes_np[i]
        style = colors[i // 2]
        if i % 2 == 0:
            style += "--"
        else:
            style += "-"
        plt.plot(
            [(j + 1) * step_size for j in range(len(mae))],
            mae,
            style,
            label=labels_np[i],
        )

    plt.xlim(0, x_max)
    plt.legend()
    plt.xlabel(r"S")
    plt.ylabel(r"$\mathcal{M}$")


def plot_epoch_convergence():
    plt.xlabel("Epochs")
    plt.xscale("log")
    plt.ylabel(r"log($\mathcal{M}$)")
    plt.plot(
        np.arange(len(mae)),
        np.log(full_mae),
        "b",
        label=r"M=101, train",
    )
    plt.plot(
        np.arange(len(mae)),
        np.log(full_val_mae),
        "b",
        label=r"M=101, val",
        ls="--",
        alpha=0.5,
    )
    plt.plot(np.arange(len(mae)), np.log(mae), "r", label=r"M=10, train")
    plt.plot(
        np.arange(len(mae)),
        np.log(val_mae),
        "r",
        label=r"M=10, val",
        ls="--",
        alpha=0.5,
    )

    plt.legend(loc="lower left")