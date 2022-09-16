import numpy as np
import matplotlib.pyplot as plt
from utility.plotting.plot_convergence import plot_sample_convergence

SAVE_NAME_DELAY = "PaperFigures/s3_sample_convergence/samples_delay.csv"
SAVE_NAME_PULSE1 = "PaperFigures/s3_sample_convergence/samples_pulse1.csv"


def load_convergence(save_name):
    with open(save_name, "r") as inp:
        line = inp.readline()

    labels = line.split(" ")[1].split(",")

    all_maes = np.loadtxt(save_name, skiprows=1, delimiter=",")
    return labels, all_maes

DELAY_DATA = load_convergence(SAVE_NAME_DELAY)
PULSE1_DATA = load_convergence(SAVE_NAME_PULSE1)


plt.style.use("utility/plotting/styling.mplstyle")
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,8))

plt.sca(ax1)
plt.text(17_000, 0.625, "(a)", fontdict={"weight": 15, "size": 35})
plot_sample_convergence(*PULSE1_DATA, 20_000, n_steps=11)
plt.sca(ax2)
plt.text(26_000, 0.235, "(b)", fontdict={"weight": 15, "size": 35})
plot_sample_convergence(*DELAY_DATA, 30_000, n_steps=11)
plt.tight_layout()
plt.savefig("PaperFigures/s3_sample_convergence/sample_convergence.pdf")
plt.show()

