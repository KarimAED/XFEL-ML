import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def norm(data, ref, label):
    """
    Normalises the data along axis 0, and stores the mean and standard deviation to the ref df,
    under <label>_mean and <label>_std.

    :param data: np.array, 2 dims, the data to be normalised
    :param ref: pandas.DataFrame, the dataframe to which to save the mean and standard deviation
    :param label: str, the label to use for the index of the ref df
    :return: np.array, normalised version of data
    """
    assert type(data) == np.ndarray, "norm: data must be np.ndarray"
    assert len(data.shape) in (1, 2), "norm: data dimensionality not 1 or 2"
    assert np.issubdtype(data.dtype, np.number), "norm: non-numeric data"
    assert type(ref) == pd.DataFrame, "norm: ref must be pd.DataFrame"
    assert type(label) == str, "norm: label must be string"
    if len(ref.columns) > 1:
        assert data.shape[1] == len(
            ref.columns
        ), "norm: number of columns in ref and data must match"
    else:
        assert (
            len(data.shape) == 1
        ), "norm: number of columns in ref and data must match"

    data_mean = pd.Series(
        data=np.mean(data, axis=0),
        index=ref.columns,
        name=label + "_mean",
        dtype=np.double,
    )
    data_std = pd.Series(
        data=np.std(data, axis=0),
        index=ref.columns,
        name=label + "_std",
        dtype=np.double,
    )

    assert np.all(data_std.values != 0), "norm: data has 0 std"

    ref = ref.append(data_mean)
    ref = ref.append(data_std)

    return (data - data_mean.values) / data_std.values, ref


def train_test_norm(inp_df, out_df, split, normalize=True):
    """
    Takes a set of input and output (label) data, splits each into training and test set based on the ratio split,
    and optionally normalizes them, storing the mean and standard deviation in dictionaries to also be returned.

    :param inp_df: pd.DataFrame, 2 dims, input features with events along axis 0 and features along axis 1
    :param out_df: pd.DataFrame, 1 or 2 dims, output labels, with events along axis 0 and if present, features along axis 1
    :param split: float, percentage of data to use for test set
    :param normalize: bool, default=True, if the data should be normalised
    :return: tuple, (X_train, X_test, y_train, y_test, inp_ref, out_ref),
        where the refs refer to the dfs containing normalisation data
    """

    # get labels from input and output data
    inp_labels = list(inp_df.columns)
    if len(out_df.shape) == 1:
        out_labels = [out_df.name]
    else:
        out_labels = list(out_df.columns)

    # get the underlying np.arrays to split and normalise
    X = inp_df.values.astype(np.double)
    y = out_df.values.astype(np.double)

    # initialise dfs to store normalisation info
    inp_ref = pd.DataFrame(columns=inp_labels)
    out_ref = pd.DataFrame(columns=out_labels)

    # randomly split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

    # normalise train and test set separately to avoid data leak
    if normalize:
        X_train, inp_ref = norm(X_train, inp_ref, "train")
        X_test, inp_ref = norm(X_test, inp_ref, "test")

        y_train, out_ref = norm(y_train, out_ref, "train")
        y_test, out_ref = norm(y_test, out_ref, "test")

    return X_train, X_test, y_train, y_test, inp_ref, out_ref
