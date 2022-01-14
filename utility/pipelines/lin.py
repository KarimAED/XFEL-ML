import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utility.plotting import plot_features, plot_fit
from utility.estimators import grad_boost
from utility.pipelines import helpers


def lin_pipeline(data, string_data, plot=True, save=True, pred_lims=False, legend=True, vmax=None):
    x_train, x_test, y_train, y_test, inp_df, out_df = data

    print(inp_df)
    print(out_df)

    lin_model = LinearRegression()

    lin_model.fit(x_train, y_train)

    print(f"Training MAE: {helpers.mae(lin_model.predict(x_train), y_train)}")
    print(f"Testing MAE: {helpers.mae(lin_model.predict(x_test), y_test)}")

    predictions = lin_model.predict(x_test)
    pred_train = lin_model.predict(x_train)

    out_ref, train_out, test_out, train_pred, test_pred = helpers.rescale_output(string_data["feat_name"],
                                                                                 out_df,
                                                                                 y_train,
                                                                                 y_test,
                                                                                 pred_train,
                                                                                 predictions)

    if save:
        np.savez(string_data["data_fname"],
                 train_out=train_out, train_pred=train_pred, test_out=test_out, test_pred=test_pred)

    if plot:
        plot_lab = string_data["plot_lab"]
        unit = string_data["unit"]
        label = "LIN; MAE: {}{}".format(
            round(helpers.mae(lin_model.predict(x_test), y_test) * out_ref["test_std"], 2), unit)

        kwargs = {"pred_lims": pred_lims, "legend": legend}
        if vmax is not None:
            kwargs["vmax"] = vmax
        plot_fit.plot_pvm(test_out, test_pred,
                          label,
                          f"Measured {plot_lab} ({unit})", f"Predicted {plot_lab} ({unit})",
                          string_data["plot_fname"], **kwargs)
    return lin_model


def lin_feature_pipeline(data, string_data, pred_lims=False, legend=True, vmax=None, noRefit=False):
    x_train, x_test, y_train, y_test, input_reference, output_reference = data

    lin_model = lin_pipeline(data, string_data, plot=False, save=False)  # if we don't plot, no need to pass other kwarg

    i_ref = input_reference

    scores = []
    permuted_features = []
    permuted_index = []

    # loop over all features one by oneto exclude them
    for i in range(len(i_ref.columns)):
        score = 0
        for k in range(5):  # average over 5 permutations
            x_te_masked = helpers.permute(x_test, i)
            score += helpers.mae(lin_model.predict(x_te_masked), y_test) / 5
        permuted_features.append(i_ref.columns[i])
        permuted_index.append(i)
        # evaluate estimator performance with one feature scrambled (on test set)
        scores.append(score)

    feature_rank = pd.DataFrame({"features": permuted_features, "mae_score": scores, "feat_ind": permuted_index})
    feature_rank.sort_values("mae_score", inplace=True, ascending=False)

    ranking = feature_rank["feat_ind"].values

    if noRefit:
        return i_ref.columns[ranking]

    scores = []

    for l in range(0, len(ranking), 5):
        data_temp = helpers.top_x_data(data, ranking, l)
        temp_lin = lin_pipeline(data_temp, string_data, save=False, plot=False)
        # collect scores with top x features included
        scores.append(helpers.mae(temp_lin.predict(data_temp[1]), y_test)
                      * output_reference.loc['test_std', string_data["feat_name"]])

    plot_features.plot_both(feature_rank["mae_score"].values, feature_rank["features"].values, scores)

    key_features = i_ref.columns[ranking][:10]

    data_filtered = helpers.top_x_data(data, ranking, 10)

    new_lin = lin_pipeline(data_filtered, string_data, pred_lims=pred_lims, legend=legend, vmax=vmax)

    return new_lin, key_features
