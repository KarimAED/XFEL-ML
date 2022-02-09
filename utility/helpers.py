import logging
import numpy as np

logger = logging.getLogger("general_util")


# Helper function to evaluate mean absolute error (mae of two numpy arrays)
def mae(x, y):
    logger.info("MAE Here")
    return np.mean(np.abs(x - y))


def permute(x_test, i):
    x_te_masked = []
    for j in range(x_test.shape[1]):
        if j != i:
            x_te_masked.append(x_test[:, j])
        elif j == i:  # filter feature to be checked (set to 0)
            rng = np.random.default_rng(1)
            x_te_masked.append(rng.permutation(x_test[:, j]))
    x_te_masked = np.stack(x_te_masked).T
    return x_te_masked


def top_x_data(data, ranking, x):
    x_train, x_test, y_train, y_test, input_reference, output_reference = data
    feats = ranking[: x + 1]
    x_tr = x_train[:, feats]
    x_te = x_test[:, feats]
    data_temp = list(data)
    data_temp[0] = x_tr
    data_temp[1] = x_te
    data_temp[4] = input_reference.iloc[:, feats]

    return data_temp


def rescale_output(
    feat_name, output_reference, y_train, y_test, pred_train, pred_test
):
    # get the relevant column from the output reference
    out_ref = output_reference[feat_name]

    # return targets & predictions to original units
    test_out = y_test * out_ref.loc["test_std"] + out_ref.loc["test_mean"]
    test_pred = pred_test * out_ref.loc["test_std"] + out_ref.loc["test_mean"]

    train_out = y_train * out_ref.loc["train_std"] + out_ref.loc["train_mean"]
    train_pred = (
        pred_train * out_ref.loc["train_std"] + out_ref.loc["train_mean"]
    )

    return out_ref, train_out, test_out, train_pred, test_pred
