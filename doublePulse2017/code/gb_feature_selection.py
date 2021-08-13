import numpy as np
import pandas as pd

from utility.estimators import grad_boost
from utility.plotting import plot_features, plot_fit
from doublePulse2017.code.setup import get_data


#%%

x_train, x_test, y_train, y_test, input_reference, output_reference = get_data()

print(input_reference)
print(output_reference)

#%%
xgb = grad_boost.fit_xgboost(x_train, y_train)

print(f"Training MAE: {grad_boost.mae(xgb.predict(x_train), y_train)}")
print(f"Testing MAE: {grad_boost.mae(xgb.predict(x_test), y_test)}")


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
    scores.append(grad_boost.mae(xgb.predict(x_te_masked), y_test))

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
    scores.append(grad_boost.mae(xgb.predict(x_te_masked), y_test))

plot_features.plot_feat_cumulative(scores)


#%%

key_features = i_ref.columns[ranking][:10]
key_feat_ind = ranking[:10]

print(key_features)

x_tr_filt = x_train[:, key_feat_ind]
x_te_filt = x_test[:, key_feat_ind]

new_xgb = grad_boost.fit_xgboost(x_tr_filt, y_train)

print(f"Training MAE: {grad_boost.mae(new_xgb.predict(x_tr_filt), y_train)}")
print(f"Testing MAE: {grad_boost.mae(new_xgb.predict(x_te_filt), y_test)}")

ticks = [-15, -10, -5, 0, 5, 10, 15, 20, 25]

test_std = output_reference.loc['test_std', 'Delays']

label = f"ANN; MAE: {round(grad_boost.mae(new_xgb.predict(x_te_filt), y_test)*test_std, 2)}fs"

plot_fit.plot_pvm(y_test, new_xgb.predict(x_te_filt),
                  label,
                  "Expected Delay in fs", "Predicted Delay in fs",
                  ticks, ticks, "doublePulse2017/results/ex_2_gb_perf/xgb_feat_low_hist2d")

#%%
predictions = new_xgb.predict(x_te_filt)

out_ref = output_reference["Delays"]

test_out = y_test*out_ref.loc["test_std"]+out_ref.loc["test_mean"]
test_pred = predictions*out_ref.loc["test_std"]+out_ref.loc["test_mean"]

train_out = y_train*out_ref.loc["train_std"]+out_ref.loc["train_mean"]
train_pred = new_xgb.predict(x_tr_filt)*out_ref.loc["train_std"]+out_ref.loc["train_mean"]

np.savez("doublePulse2017/results/ex_2_gb_perf/xgb_10_feat_pred.npz",
         train_out=train_out, train_pred=train_pred, test_out=test_out, test_pred=test_pred)