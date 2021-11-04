import numpy as np
import pandas as pd

from utility.estimators import neural_network
from utility.plotting import plot_fit, plot_features


def ann_pipeline(data, string_data):
    x_train, x_test, y_train, y_test, input_reference, output_reference = data
    print(input_reference)
    print(output_reference)

    layers = neural_network.get_layers([20, 20], "relu", "l2", 0, False)

    ann, hist = neural_network.fit_ann(x_train, y_train, layers, epochs=5_000, rate=0.001)

    print(f"Training MAE: {ann.evaluate(x_train, y_train)[1]}")
    print(f"Testing MAE: {ann.evaluate(x_test, y_test)[1]}")

    predictions = ann.predict(x_test).T[0]

    out_ref = output_reference[string_data["feat_name"]]

    test_out = y_test * out_ref.loc["test_std"] + out_ref.loc["test_mean"]
    test_pred = predictions * out_ref.loc["test_std"] + out_ref.loc["test_mean"]

    train_out = y_train * out_ref.loc["train_std"] + out_ref.loc["train_mean"]
    train_pred = ann.predict(x_train).T[0] * out_ref.loc["train_std"] + out_ref.loc["train_mean"]

    np.savez(string_data["data_fname"],
             train_out=train_out, train_pred=train_pred, test_out=test_out, test_pred=test_pred)

    eval_mae = ann.evaluate(x_test, y_test)[1]

    unit = string_data["unit"]
    label = string_data["plot_lab"]

    plot_fit.plot_pvm(test_out, test_pred,
                      f"ANN; MAE: {round(eval_mae, 2)}{unit}",
                      f"Expected {label} ({unit})", f"Predicted {label} ({unit})",
                      string_data["plot_fname"])

    return ann, hist


def ann_feature_pipeline(data, string_data, vmax=None, legend=True, noRefit=False):
    x_train, x_test, y_train, y_test, input_reference, output_reference = data
    print(input_reference)
    print(output_reference)

    layers = neural_network.get_layers([20, 20], "relu", "l2", 0, False)

    ann, hist = neural_network.fit_ann(x_train, y_train, layers, epochs=2_500, rate=0.001)

    print(f"Training MAE: {ann.evaluate(x_train, y_train)[1]}")
    print(f"Testing MAE: {ann.evaluate(x_test, y_test)[1]}")

    i_ref = input_reference
    scores = []
    excluded_features = []
    excluded_index = []
    for i in range(len(i_ref.columns)):
        x_te_masked = []
        for j in range(len(i_ref.columns)):
            if j != i:
                x_te_masked.append(x_test[:, j])
            elif j == i:
                x_te_masked.append(np.zeros(x_test.shape[0]))
        x_te_masked = np.stack(x_te_masked).T
        excluded_features.append(i_ref.columns[i])
        excluded_index.append(i)
        scores.append(ann.evaluate(x_te_masked, y_test)[1])

    feature_rank = pd.DataFrame({"features": excluded_features, "mae_score": scores, "feat_ind": excluded_index})
    feature_rank.sort_values("mae_score", inplace=True, ascending=False)

    plot_features.plot_feat_hist(feature_rank["mae_score"].values, feature_rank["features"].values)

    ranking = feature_rank["feat_ind"].values

    scores = []

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
        scores.append(ann.evaluate(x_te_masked, y_test)[1] * output_reference.loc['test_std', string_data["feat_name"]])

    plot_features.plot_feat_cumulative(scores)

    key_features = i_ref.columns[ranking][:10]
    key_feat_ind = ranking[:10]

    print(i_ref.columns[ranking].tolist())

    if noRefit:
        return i_ref.columns[ranking]

    x_tr_filt = x_train[:, key_feat_ind]
    x_te_filt = x_test[:, key_feat_ind]

    layers = neural_network.get_layers([20, 20], "relu", "l2", 0, False)
    new_ann, hew_hist = neural_network.fit_ann(x_tr_filt, y_train, layers, epochs=5_000, rate=0.001)

    print(f"Training MAE: {new_ann.evaluate(x_tr_filt, y_train)[1]}")
    print(f"Testing MAE: {new_ann.evaluate(x_te_filt, y_test)[1]}")

    predictions = new_ann.predict(x_te_filt).T[0]

    out_ref = output_reference[string_data["feat_name"]]

    test_out = y_test * out_ref.loc["test_std"] + out_ref.loc["test_mean"]
    test_pred = predictions * out_ref.loc["test_std"] + out_ref.loc["test_mean"]

    train_out = y_train * out_ref.loc["train_std"] + out_ref.loc["train_mean"]
    train_pred = new_ann.predict(x_tr_filt).T[0] * out_ref.loc["train_std"] + out_ref.loc["train_mean"]

    np.savez(string_data["data_fname"],
             train_out=train_out, train_pred=train_pred, test_out=test_out, test_pred=test_pred)

    unit = string_data["unit"]
    xy_label = string_data["plot_lab"]

    std = out_ref.loc['test_std']

    label = "ANN; MAE: {}{}".format(round(new_ann.evaluate(x_te_filt, y_test)[1]*std, 2), unit)

    if vmax is not None:
        plot_fit.plot_pvm(test_out, test_pred,
                          label,
                          f"Measured {xy_label} ({unit})", f"Predicted {xy_label} ({unit})",
                          string_data["plot_fname"], vmax=vmax, legend=legend)
    else:
        plot_fit.plot_pvm(test_out, test_pred,
                          label,
                          f"Measured {xy_label} ({unit})", f"Predicted {xy_label} ({unit})",
                          string_data["plot_fname"], legend=legend)
    label = "ANN; MAE: {}{}".format(round(new_ann.evaluate(x_te_filt, y_test)[1], 2), r"$\sigma$")

    if vmax is not None:
        plot_fit.plot_pvm(test_out, test_pred,
                          label,
                          f"Measured {xy_label} ({unit})", f"Predicted {xy_label} ({unit})",
                          string_data["plot_fname"]+"_normed", vmax=vmax, legend=legend)
    else:
        plot_fit.plot_pvm(test_out, test_pred,
                          label,
                          f"Measured {xy_label} ({unit})", f"Predicted {xy_label} ({unit})",
                          string_data["plot_fname"]+"_normed", legend=legend)

    return new_ann, key_features
