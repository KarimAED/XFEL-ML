import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def norm(data, ref, label):
    data_mean = pd.Series(data=np.mean(data, axis=0), index=ref.columns, name=label + "_mean")
    data_std = pd.Series(data=np.std(data, axis=0), index=ref.columns, name=label + "_std")
    ref = ref.append(data_mean)
    ref = ref.append(data_std)

    return (data - data_mean.values) / data_std.values, ref


def train_test_norm(inp_df, out_df, split, normalize=True):

    inp_labels = list(inp_df.columns)
    if len(out_df.shape) == 1:
        out_labels = [out_df.name]
    else:
        out_labels = list(out_df.columns)

    X = inp_df.values
    y = out_df.values

    inp_ref = pd.DataFrame(columns=inp_labels)
    out_ref = pd.DataFrame(columns=out_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

    if normalize:
        X_train, inp_ref = norm(X_train, inp_ref, "train")
        X_test, inp_ref = norm(X_test, inp_ref, "test")

        y_train, out_ref = norm(y_train, out_ref, "train")
        y_test, out_ref = norm(y_test, out_ref, "test")

    return X_train, X_test, y_train, y_test, inp_ref, out_ref