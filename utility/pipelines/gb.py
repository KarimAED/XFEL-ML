import numpy as np
import pandas as pd

from utility.estimators import grad_boost
from utility.plotting import plot_fit, plot_features
from utility.pipelines import permute as perm


def gb_pipeline(data, string_data, save=True, plot=True, vmax=None, legend=True):
    """
    Pipeline to extract data, fit a gb regressor and save + plot the results.

    :param data: array-like, contains x_train, x_test, y_train, y_test, input_reference, output_reference,
        as returned by data scripts[]
        for j in range(len(i_ref.columns)):
            if j != i:
                x_te_masked.append(x_test[:, j])
            elif j == i:
                x_te_masked.append(np.zeros(x_test.shape[0]))
        x_te_masked = np.stack(x_te_masked).T
    :param string_data: dict-like, contains strings as labels, filenames, etc.
    :param save: bool, if the results are to be saved
    :param plot: bool, if the results are to be plotted in a 2d-hist
    :param vmax: int, vmax to use for the colorbar of the 2d-hist
    :param legend: bool, if a colorbar legend should be displayed
    :return: Estimator object
    """
    x_train, x_test, y_train, y_test, input_reference, output_reference = data

    print(input_reference)
    print(output_reference)

    xgb = grad_boost.fit_xgboost(x_train, y_train)

    print(f"Training MAE: {grad_boost.mae(xgb.predict(x_train), y_train)}")
    print(f"Testing MAE: {grad_boost.mae(xgb.predict(x_test), y_test)}")

    predictions = xgb.predict(x_test)

    out_ref = output_reference[string_data["feat_name"]]

    test_out = y_test * out_ref.loc["test_std"] + out_ref.loc["test_mean"]
    test_pred = predictions * out_ref.loc["test_std"] + out_ref.loc["test_mean"]

    train_out = y_train * out_ref.loc["train_std"] + out_ref.loc["train_mean"]
    train_pred = xgb.predict(x_train) * out_ref.loc["train_std"] + out_ref.loc["train_mean"]

    if save:
        np.savez(string_data["data_fname"],
                 train_out=train_out, train_pred=train_pred, test_out=test_out, test_pred=test_pred)

    if plot:
        test_std = out_ref.loc['test_std']

        plot_lab = string_data["plot_lab"]
        unit = string_data["unit"]

        label = "GB; MAE: {}{}".format(round(grad_boost.mae(xgb.predict(x_test), y_test) * test_std, 2), unit)
        kwargs = {"legend": legend}
        if vmax is not None:
            kwargs["vmax"] = vmax
        plot_fit.plot_pvm(test_out, test_pred,
                          label,
                          f"Measured {plot_lab} ({unit})", f"Predicted {plot_lab} ({unit})",
                          string_data["plot_fname"], **kwargs)

    return xgb


def gb_feature_pipeline(data, string_data, vmax=None, legend=False, noRefit=False):
    """
    Pipeline used to fit a gb regressor and then perform feature selection, ranking features
    and only using the top 10 features for refit.

    :param data: array-like, contains x_train, x_test, y_train, y_test, input_reference, output_reference,
        as returned by data scripts
    :param string_data: dict-like, contains strings as labels, filenames, etc.
    :param vmax: int, vmax to use for the colorbar of the 2d-hist
    :param legend: bool, if a colorbar legend should be displayed
    :return: returns refit estimator + top 10 features used in the refit
    """
    x_train, x_test, y_train, y_test, input_reference, output_reference = data
    xgb = gb_pipeline(data, string_data, plot=False, save=False, vmax=vmax, legend=legend)

    i_ref = input_reference

    scores = []
    excluded_features = []
    excluded_index = []
    for i in range(len(i_ref.columns)):
        score = 0
        for k in range(5):
            x_te_masked = perm.permute(x_test, i, i_ref)
            score += grad_boost.mae(xgb.predict(x_te_masked), y_test) / 5
        excluded_features.append(i_ref.columns[i])
        excluded_index.append(i)
        scores.append(score)

    feature_rank = pd.DataFrame({"features": excluded_features, "mae_score": scores, "feat_ind": excluded_index})
    feature_rank.sort_values("mae_score", inplace=True, ascending=False)

    test_std = output_reference.loc['test_std', string_data["feat_name"]]

    ranking = feature_rank["feat_ind"].values

    if noRefit:
        return i_ref.columns[ranking]

    scores = []

    for l in range(0, len(ranking), 5):
        feats = ranking[:l + 1]
        x_tr = x_train[feats]
        x_te = x_test[feats]
        data_temp = list(data)
        data_temp[0] = x_tr
        data_temp[1] = x_te
        data_temp[4] = input_reference[feats]
        temp_gb = gb_pipeline(data_temp, string_data, save=False, plot=False)
        # collect scores with top x features included
        scores.append(grad_boost.mae(temp_gb.predict(x_te_masked), y_test)
                      * output_reference.loc['test_std', string_data["feat_name"]])

    plot_features.plot_both(feature_rank["mae_score"].values, feature_rank["features"].values, scores)

    key_features = i_ref.columns[ranking][:10]
    key_feat_ind = ranking[:10]

    print(key_features)

    x_tr_filt = x_train[:, key_feat_ind]
    x_te_filt = x_test[:, key_feat_ind]

    data_filtered = list(data)
    data_filtered[0] = x_tr_filt
    data_filtered[1] = x_te_filt
    data_filtered[4] = input_reference.iloc[:, key_feat_ind]

    new_xgb = gb_pipeline(data_filtered, string_data, vmax=vmax, legend=legend)

    return new_xgb, key_features

