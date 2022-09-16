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

from utility.estimators import neural_network
from utility.plotting import plot_fit
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

    if args.verbose != 0:
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
    if args.verbose != 0:
        print(f"Training MAE: {ann.evaluate(x_train, y_train, verbose=args.verbose)[1]}")
        print(f"Testing MAE: {ann.evaluate(x_test, y_test, verbose=args.verbose)[1]}")

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