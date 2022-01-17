import logging
import numpy as np
import pandas as pd

from utility.estimators import neural_network
from utility.plotting import plot_fit, plot_features
from utility import helpers

logger = logging.getLogger("pipelines")


def ann_pipeline(data, string_data, save=True, plot=True, vmax=None, legend=True, verbose=2):
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
    print("Starting ann training...")
    logger.info("Fitting ANN on data...")

    # unpack data
    x_train, x_test, y_train, y_test, input_reference, output_reference = data

    # print the reference values (std and mean used for normalization)
    logger.info(input_reference)
    logger.info(output_reference)

    # generate layer list
    layers = neural_network.get_layers([20, 20], "relu", "l2", 0, False)

    # fit ann with data and params, save history also
    ann, hist = neural_network.fit_ann(x_train, y_train, layers, epochs=5_000, rate=0.001, verbose=verbose)

    # print training and testing mean absolute error (KPMs), both are printed to inform about potential overfitting
    print(f"Training MAE: {ann.evaluate(x_train, y_train)[1]}")
    print(f"Testing MAE: {ann.evaluate(x_test, y_test)[1]}")

    # make predictions on the test set
    predictions = ann.predict(x_test).T[0]
    pred_train = ann.predict(x_train).T[0]

    out_ref, train_out, test_out, train_pred, test_pred = helpers.rescale_output(string_data["feat_name"],
                                                                                 output_reference,
                                                                                 y_train,
                                                                                 y_test,
                                                                                 pred_train,
                                                                                 predictions)

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
    permuted_features = []
    permuted_index = []

    # loop over all features one by oneto exclude them
    for i in range(len(i_ref.columns)):
        score = 0
        for k in range(5):  # average over 5 permutations
            print("feature %i / %i; permutation %i / 5" % (i + 1, len(i_ref.columns), k), end="\r")
            x_te_masked = helpers.permute(x_test, i)
            score += ann.evaluate(x_te_masked, y_test, verbose=0)[1] / 5
        permuted_features.append(i_ref.columns[i])
        permuted_index.append(i)
        # evaluate estimator performance with one feature scrambled (on test set)
        scores.append(score)

    # get data frame ranking all features
    feature_rank = pd.DataFrame({"features": permuted_features, "mae_score": scores, "feat_ind": permuted_index})
    feature_rank.sort_values("mae_score", inplace=True, ascending=False)  # sort data frame by feature importance

    ranking = feature_rank["feat_ind"].values

    logger.info(i_ref.columns[ranking].tolist())

    # return all features ranked if not to refit
    if noRefit:
        return i_ref.columns[ranking]

    scores = []

    # loop over all features again, including only the top x features and refitting the estimator for each of them
    for l in range(0, len(ranking), 5):
        print("Refitting with the top %i / %i feats" % (l+1, len(ranking)), end="\r")
        data_temp = helpers.top_x_data(data, ranking, l)
        temp_ann, temp_hist = ann_pipeline(data_temp, string_data, save=False, plot=False, verbose=0)
        # collect scores with top x features included
        scores.append(temp_ann.evaluate(data_temp[1], y_test)[1]
                      * output_reference.loc['test_std', string_data["feat_name"]])

    # plot feature importance and mae score with features up to feature j on one plot
    plot_features.plot_both(feature_rank["mae_score"].values, feature_rank["features"].values, scores)

    # select top 10 features as key features
    key_features = i_ref.columns[ranking][:10]

    data_filtered = helpers.top_x_data(data, ranking, 10)

    # refit estimator with top features
    new_ann, hist = ann_pipeline(data_filtered, string_data, vmax=vmax, legend=legend)

    return new_ann, key_features
