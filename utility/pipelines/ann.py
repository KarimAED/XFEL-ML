import numpy as np
import pandas as pd

from utility.estimators import neural_network
from utility.plotting import plot_fit, plot_features


def ann_pipeline(data, string_data, save=True, plot=True, vmax=None, legend=True):
    """
    Pipeline to extract data, fit an ann and save + plot the results.

    :param data: array-like, contains x_train, x_test, y_train, y_test, input_reference, output_reference,
        as returned by data scripts
    :param string_data: dict-like, contains strings as labels, filenames, etc.
    :param save: bool, if the results are to be saved
    :param plot: bool, if the results are to be plotted in a 2d-hist
    :param vmax: int, vmax to use for the colorbar of the 2d-hist
    :param legend: bool, if a colorbar legend should be displayed
    :return: tuple of Estimator and History objects
    """

    # unpack data
    x_train, x_test, y_train, y_test, input_reference, output_reference = data

    # print the reference values (std and mean used for normalization)
    print(input_reference)
    print(output_reference)

    # generate layer list
    layers = neural_network.get_layers([20, 20], "relu", "l2", 0, False)

    # fit ann with data and params, save history also
    ann, hist = neural_network.fit_ann(x_train, y_train, layers, epochs=5_000, rate=0.001)

    # print training and testing mean absolute error (KPMs), both are printed to inform about potential overfitting
    print(f"Training MAE: {ann.evaluate(x_train, y_train)[1]}")
    print(f"Testing MAE: {ann.evaluate(x_test, y_test)[1]}")

    # make predictions on the test set
    predictions = ann.predict(x_test).T[0]

    # get the relevant column from the output reference
    out_ref = output_reference[string_data["feat_name"]]

    # return targets & predictions to original units
    test_out = y_test * out_ref.loc["test_std"] + out_ref.loc["test_mean"]
    test_pred = predictions * out_ref.loc["test_std"] + out_ref.loc["test_mean"]

    train_out = y_train * out_ref.loc["train_std"] + out_ref.loc["train_mean"]
    train_pred = ann.predict(x_train).T[0] * out_ref.loc["train_std"] + out_ref.loc["train_mean"]

    # file is not saved for e.g. the first step in the feature selection pipeline
    if save:

        # save data to filename specified in the string_data dict
        np.savez(string_data["data_fname"],
                 train_out=train_out, train_pred=train_pred, test_out=test_out, test_pred=test_pred)

    # plotting not always desirable
    if plot:

        # get labels from string data
        unit = string_data["unit"]
        xy_label = string_data["plot_lab"]

        # get std to use for un-normalized mae calculation
        std = out_ref.loc['test_std']

        label = "ANN; MAE: {}{}".format(round(ann.evaluate(x_test, y_test)[1] * std, 2), unit)
        kwargs = {"legend": legend}
        if vmax is not None:  # if vmax is provided, use uniform colormap range
            kwargs["vmax"] = vmax
        plot_fit.plot_pvm(test_out, test_pred,
                          label,
                          f"Measured {xy_label} ({unit})", f"Predicted {xy_label} ({unit})",
                          string_data["plot_fname"], **kwargs)

    return ann, hist


def ann_feature_pipeline(data, string_data, vmax=None, legend=True, noRefit=False):
    """
    Pipeline used to fit an ann and then perform feature selection, ranking features
    and only using the top 10 features for refit, if desired.

    :param data: array-like, contains x_train, x_test, y_train, y_test, input_reference, output_reference,
        as returned by data scripts
    :param string_data: dict-like, contains strings as labels, filenames, etc.
    :param vmax: int, vmax to use for the colorbar of the 2d-hist
    :param legend: bool, if a colorbar legend should be displayed
    :param noRefit: bool, if true, the function terminates after having found the key features, otherwise ann is fit again
    :return: if noRefit: returns features ranked by feature importance,
        else: returns refit estimator + top 10 features used in the refit
    """
    # unpack the data
    x_train, x_test, y_train, y_test, input_reference, output_reference = data
    ann, hist = ann_pipeline(data, string_data, save=False, plot=False)  # perform initial fitting

    # set up everything to rank features
    i_ref = input_reference
    scores = []
    excluded_features = []
    excluded_index = []

    # loop over all features one by oneto exclude them
    for i in range(len(i_ref.columns)):
        x_te_masked = []
        for j in range(len(i_ref.columns)):
            if j != i:
                x_te_masked.append(x_test[:, j])
            elif j == i:  # filter feature to be checked (set to 0)
                x_te_masked.append(np.zeros(x_test.shape[0]))
        x_te_masked = np.stack(x_te_masked).T
        excluded_features.append(i_ref.columns[i])
        excluded_index.append(i)
        # evaluate estimator performance with one feature scrambled (on test set)
        scores.append(ann.evaluate(x_te_masked, y_test)[1])

    # get data frame ranking all features
    feature_rank = pd.DataFrame({"features": excluded_features, "mae_score": scores, "feat_ind": excluded_index})
    feature_rank.sort_values("mae_score", inplace=True, ascending=False)  # sort data frame by feature importance

    ranking = feature_rank["feat_ind"].values

    scores = []

    # loop over all features again, including only the top x features but using the old estimator
    for l in range(len(ranking)):
        feats = ranking[:l + 1]
        x_te_masked = []
        for j in range(len(i_ref.columns)):
            if j in feats:
                x_te_masked.append(x_test[:, j])
            else:
                rng = np.random.default_rng(1)
                x_te_masked.append(rng.permutation(x_test[:, j]))
        x_te_masked = np.stack(x_te_masked).T
        # collect scores with top x features included
        scores.append(ann.evaluate(x_te_masked, y_test)[1] * output_reference.loc['test_std', string_data["feat_name"]])

    # plot feature importance and mae score with features up to feature j on one plot
    plot_features.plot_both(feature_rank["mae_score"].values, feature_rank["features"].values, scores)

    # select top 10 features as key features
    key_features = i_ref.columns[ranking][:10]
    key_feat_ind = ranking[:10]

    print(i_ref.columns[ranking].tolist())

    # return all features ranked if not to refit
    if noRefit:
        return i_ref.columns[ranking]

    # filter out only the top features
    x_tr_filt = x_train[:, key_feat_ind]
    x_te_filt = x_test[:, key_feat_ind]

    data_filtered = data.copy()
    data_filtered[0] = x_tr_filt
    data_filtered[1] = x_te_filt
    data_filtered[4] = input_reference.iloc[:, key_feat_ind]

    # refit estimator with top features
    new_ann, hist = ann_pipeline(data_filtered, string_data, vmax=vmax, legend=legend)

    return new_ann, key_features
