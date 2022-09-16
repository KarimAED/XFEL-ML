import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utility.plotting import plot_features, plot_fit
from utility import helpers


def lin_pipeline(
    data,
    string_data,
    plot=True,
    save=True,
    pred_lims=False,
    legend=True,
    vmax=None,
):
    x_train, x_test, y_train, y_test, inp_df, out_df = data

    print(inp_df)
    print(out_df)

    lin_model = LinearRegression()

    lin_model.fit(x_train, y_train)

    print(f"Training MAE: {helpers.mae(lin_model.predict(x_train), y_train)}")
    print(f"Testing MAE: {helpers.mae(lin_model.predict(x_test), y_test)}")

    predictions = lin_model.predict(x_test)
    pred_train = lin_model.predict(x_train)

    (
        out_ref,
        train_out,
        test_out,
        train_pred,
        test_pred,
    ) = helpers.rescale_output(
        string_data["feat_name"],
        out_df,
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
        plot_lab = string_data["plot_lab"]
        unit = string_data["unit"]
        label = "LIN; MAE: {}{}".format(
            round(
                helpers.mae(lin_model.predict(x_test), y_test)
                * out_ref["test_std"],
                2,
            ),
            unit,
        )

        kwargs = {"pred_lims": pred_lims, "legend": legend}
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
    return lin_model, None