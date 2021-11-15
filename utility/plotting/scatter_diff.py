import matplotlib.pyplot as plt
import numpy as np
plt.style.use("./utility/plotting/styling.mplstyle")


def scatter_diff(data_1, data_2, string_args):
    print(f"Plotting scatter difference for {data_1} and {data_2}")
    all_feat_data = np.load(data_1)
    red_feat_data = np.load(data_2)

    af_y = all_feat_data["test_out"]
    af_p = all_feat_data["test_pred"]

    rf_y = red_feat_data["test_out"]
    rf_p = red_feat_data["test_pred"]


    all_y = np.append(rf_y, af_y)
    all_p = np.append(rf_p, af_p)

    x = [np.min(all_y), np.max(all_y)]

    binary_label = np.append(np.zeros(rf_y.size), np.ones(af_y.size))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(x)
    ax.set_ylim(x)
    # ax.set_xlabel(f"Measured {string_args['quantity']} ({string_args['unit']})")
    # ax.set_ylabel(f"Predicted {string_args['quantity']} ({string_args['unit']})")
    scatter = ax.scatter(all_y, all_p, c=binary_label,
                         cmap="bwr", s=2, alpha=0.2)
    ax.plot(x, x, "k--")
    leg = ax.legend(scatter.legend_elements(alpha=1, prop="colors")[0],
                    [string_args['label_1'], string_args['label_2']], loc="lower right")
    ax.add_artist(leg)
    plt.show()
