import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utility.plotting import plot_features, plot_fit
from pumpProbe2021.code.setup import get_pump_data
from utility.estimators import grad_boost

undulators_2_datasets = ["u2_271_37229_events.pkl", "u2_273_37026_events.pkl", "u2_275_36614_events.pkl",
                         "u2_277_38126_events.pkl", "u2_279_37854_events.pkl"]

#%%

x_train, x_test, y_train, y_test, inp_df, out_df = get_pump_data(undulators_2_datasets[1])

print(inp_df)
print(out_df)

#%%

input_reference = inp_df
output_reference = out_df

lin_model = LinearRegression()

lin_model.fit(x_train, y_train)

print(f"Training MAE: {grad_boost.mae(lin_model.predict(x_train), y_train)}")
print(f"Testing MAE: {grad_boost.mae(lin_model.predict(x_test), y_test)}")


#%%
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
    scores.append(grad_boost.mae(lin_model.predict(x_te_masked), y_test))

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
    scores.append(grad_boost.mae(lin_model.predict(x_te_masked), y_test)*output_reference.loc['test_std', 'vls_com_pump'])

plot_features.plot_feat_cumulative(scores)

#%%

key_features = i_ref.columns[ranking][:10]
key_feat_ind = ranking[:10]

print(key_features)

x_tr_filt = x_train[:, key_feat_ind]
x_te_filt = x_test[:, key_feat_ind]

new_lin = LinearRegression()
new_lin.fit(x_tr_filt, y_train)

#%%
predictions = new_lin.predict(x_te_filt)

out_ref = output_reference["vls_com_pump"]

test_out = y_test*out_ref.loc["test_std"]+out_ref.loc["test_mean"]
test_pred = predictions*out_ref.loc["test_std"]+out_ref.loc["test_mean"]

train_out = y_train*out_ref.loc["train_std"]+out_ref.loc["train_mean"]
train_pred = new_lin.predict(x_tr_filt)*out_ref.loc["train_std"]+out_ref.loc["train_mean"]


#%%
mae = grad_boost.mae(lin_model.predict(x_test), y_test)*output_reference.loc['test_std', 'vls_com_pump']
label = f"LIN; MAE: {round(mae, 2)}eV"

plot_fit.plot_pvm(test_out, lin_model.predict(x_test)*out_ref.loc["test_std"],
                  label,
                  "Expected Pump CoM (eV)", "Predicted Pump CoM (eV)",
                  "pumpProbe2021/results/ex_1_pump_pred/lin_low_pump_hist2d")
