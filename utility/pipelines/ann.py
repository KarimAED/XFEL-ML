"""
File defining the main pipelines used to fit an artificial neural network
onto our data.

Methods:
    ann_pipeline: fits an ANN to data without feature selection
    ann_feature_pipeline: fits an ANN to data with feature selection
"""
import logging
from types import SimpleNamespace
import numpy as np
import pandas as pd

from utility.estimators import neural_network
from utility.plotting import plot_fit, plot_features
from utility import helpers

logger = logging.getLogger("pipelines")


def ann_pipeline(data, string_data, **kwargs):
    """Pipeline to extract data, fit an ann and save + plot the results.

    Pipeline to extract data, fit an ann and save + plot the results.

    Args:
        data (array[object]): contains x_train, x_test, y_train, y_test,
            input_reference, output_reference, as returned by data scripts
        string_data (dict): contains strings as labels, filenames, etc.
        key save (bool): if the results are to be saved
        key plot (bool): if the results are to be plotted in a 2d-hist
        key vmax (int): vmax to use for the colorbar of the 2d-hist
        key legend (bool): if a colorbar legend should be displayed

    Returns:
        array[tf.keras.Sequential, tf.history]: tuple of Estimator
            and training History objects
    """
    kw_dict = {
        "save": True,
        "plot": True,
        "vmax": None,
        "legend": True,
        "verbose": 2,
    }

    for key, val in kwargs.items():
        kw_dict[key] = val

    args = SimpleNamespace(**kw_dict)

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
    ann, hist = neural_network.fit_ann(
        x_train,
        y_train,
        layers,
        epochs=5_000,
        rate=0.001,
        verbose=args.verbose,
    )

    # print training and testing mean absolute error (KPMs)
    # both are printed to inform about potential overfitting
    print(f"Training MAE: {ann.evaluate(x_train, y_train)[1]}")
    print(f"Testing MAE: {ann.evaluate(x_test, y_test)[1]}")

    # make predictions on the test set
    predictions = ann.predict(x_test).T[0]
    pred_train = ann.predict(x_train).T[0]

    (
        out_ref,
        train_out,
        test_out,
        train_pred,
        test_pred,
    ) = helpers.rescale_output(
        string_data["feat_name"],
        output_reference,
        y_train,
        y_test,
        pred_train,
        predictions,
    )

    # file is not saved for e.g. the first step
    # in the feature selection pipeline
    if args.save:
        # save data to filename specified in the string_data dict
        np.savez(
            string_data["data_fname"],
            train_out=train_out,
            train_pred=train_pred,
            test_out=test_out,
            test_pred=test_pred,
        )

    # plotting not always desirable
    if args.plot:

        # get labels from string data
        unit = string_data["unit"]
        xy_label = string_data["plot_lab"]

        # get std to use for un-normalized mae calculation
        std = out_ref.loc["test_std"]

        mae = round(ann.evaluate(x_test, y_test)[1] * std, 2)
        label = f"ANN; MAE: {mae}{unit}"
        kwargs = {"legend": args.legend}
        if args.vmax is not None:
            # if vmax is provided, use consistent colormap range
            kwargs["vmax"] = args.vmax
        plot_fit.plot_pvm(
            test_out,
            test_pred,
            label,
            f"Measured {xy_label} ({unit})",
            f"Predicted {xy_label} ({unit})",
            string_data["plot_fname"],
            **kwargs,
        )

    return ann, hist


def ann_feature_pipeline(
    data,
    string_data,
    vmax=None,
    legend=True,
    no_refit=False,
):
    """Pipeline used to fit an ann with feature selection

    Pipeline used to fit an ann and then perform feature selection,
    ranking features and only using the top 10 features for refit,
    if desired.

    Args:
        data (array[object]): contains x_train, x_test, y_train, y_test,
            input_reference, output_reference, as returned by data scripts
        string_data (dict): contains strings as labels, filenames, etc.
        vmax (int): vmax to use for the colorbar of the 2d-hist
        legend (bool): if a colorbar legend should be displayed
        no_refit (bool): if true, the function terminates after having
            found the key features, otherwise ann is fit again

    Returns:
        object: if no_refit returns features ranked by feature importance,
            else returns refit estimator, top 10 features used in the refit
    """
    # unpack the data
    _, x_test, _, y_test, input_reference, output_reference = data
    ann, _ = ann_pipeline(
        data, string_data, save=False, plot=False
    )  # perform initial fitting

    # set up everything to rank features
    i_ref = input_reference
    scores = []
    permuted_features = []
    permuted_index = []

    # loop over all features one by oneto exclude them
    for i, col in enumerate(i_ref.columns):
        score = 0
        for k in range(5):  # average over 5 permutations
            max_col = len(i_ref.columns)
            print(
                f"feature {i + 1} / {max_col}; permutation {k} / 5",
                end="\r",
            )
            x_te_masked = helpers.permute(x_test, i)
            score += ann.evaluate(x_te_masked, y_test, verbose=0)[1] / 5
        permuted_features.append(col)
        permuted_index.append(i)
        # evaluate estimator performance
        # with one feature scrambled (on test set)
        scores.append(score)

    # get data frame ranking all features
    feature_rank = pd.DataFrame(
        {
            "features": permuted_features,
            "mae_score": scores,
            "feat_ind": permuted_index,
        }
    )
    feature_rank.sort_values(
        "mae_score", inplace=True, ascending=False
    )  # sort data frame by feature importance

    ranking = feature_rank["feat_ind"].values

    logger.info(i_ref.columns[ranking].tolist())

    # return all features ranked if not to refit
    if no_refit:
        return i_ref.columns[ranking]

    scores = []

    # loop over all features again
    # including only the top x features
    # and refitting the estimator for each of them
    for feat_set in range(0, len(ranking), 5):
        print(
            f"Refitting with the top {feat_set + 1} / {len(ranking)} feats",
            end="\r",
        )
        data_temp = helpers.top_x_data(data, ranking, feat_set)
        temp_ann, _ = ann_pipeline(
            data_temp, string_data, save=False, plot=False, verbose=0
        )
        # collect scores with top x features included
        scores.append(
            temp_ann.evaluate(data_temp[1], y_test)[1]
            * output_reference.loc["test_std", string_data["feat_name"]]
        )

    # plot feature importance and mae score
    # with features up to feature j on one plot
    plot_features.plot_both(
        feature_rank["mae_score"].values,
        feature_rank["features"].values,
        scores,
    )

    # select top 10 features as key features
    key_features = i_ref.columns[ranking][:10]

    data_filtered = helpers.top_x_data(data, ranking, 10)

    # refit estimator with top features
    new_ann, _ = ann_pipeline(
        data_filtered, string_data, vmax=vmax, legend=legend
    )

    return new_ann, key_features
