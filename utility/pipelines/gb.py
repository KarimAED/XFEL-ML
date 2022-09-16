import numpy as np
import pandas as pd

from utility.estimators import grad_boost
from utility.plotting import plot_fit, plot_features
from utility import helpers


def gb_pipeline(
    data, string_data, save=True, plot=True, vmax=None, legend=True
):
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

    print(f"Training MAE: {helpers.mae(xgb.predict(x_train), y_train)}")
    print(f"Testing MAE: {helpers.mae(xgb.predict(x_test), y_test)}")

    predictions = xgb.predict(x_test)
    pred_train = xgb.predict(x_train)

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

    if save:
        np.savez(
            string_data["data_fname"],
            train_out=train_out,
            train_pred=train_pred,
            test_out=test_out,
            test_pred=test_pred,
        )

    if plot:
        test_std = out_ref.loc["test_std"]

        plot_lab = string_data["plot_lab"]
        unit = string_data["unit"]

        label = "GB; MAE: {}{}".format(
            round(helpers.mae(xgb.predict(x_test), y_test) * test_std, 2), unit
        )
        kwargs = {"legend": legend}
        if vmax is not None:
            kwargs["vmax"] = vmax
        plot_fit.plot_pvm(
            test_out,
            test_pred,
            label,
            f"Measured {plot_lab} ({unit})",
            f"Predicted {plot_lab} ({unit})",
            string_data["plot_fname"],
            **kwargs,
        )

    return xgb, None
