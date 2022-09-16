import numpy as np


# Helper function to evaluate mean absolute error (mae of two numpy arrays)
def mae(x_arr, y_arr):
    """Shorthand to evaluate mean absolute error between two numpy arrays.
    Must have the same shape.

    :param x_arr: np.ndarray, first array for mae.
    :param y_arr: np.ndarray, second array for mae.
    :return: float, mean absolute error between arrays.
    """
    return np.mean(np.abs(x_arr - y_arr))


def permute(x_test, i):
    """Perform permutation of the ith column of the x_test array.

    :param x_test: np.ndarray, array to take a column to permute from.
    :param i: int, index of the column to permute.
    :return: np.ndarray, array with column permuted.
    """
    x_te_masked = []
    for j in range(x_test.shape[1]):
        if j != i:
            x_te_masked.append(x_test[:, j])
        elif j == i:  # filter feature to be checked (set to 0)
            rng = np.random.default_rng(1)
            x_te_masked.append(rng.permutation(x_test[:, j]))
    x_te_masked = np.stack(x_te_masked).T
    return x_te_masked


def top_x_data(data, ranking, x_int):
    """Modify data to only contain the top x features.

    :param data:
    :param ranking:
    :param x_int:
    :return:
    """
    print(ranking)
    x_train, x_test, _, _, input_reference, _ = data
    feats = ranking[: x_int + 1]
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
