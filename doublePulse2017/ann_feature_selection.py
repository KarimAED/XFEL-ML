import numpy as np
import pandas as pd

from utility.estimators import neural_network
from utility.plotting import plot_features
from doublePulse2017.setup import get_data


#%%

x_train, x_test, y_train, y_test, input_reference, output_reference = get_data()

print(input_reference)
print(output_reference)

#%%
layers = neural_network.get_layers([50, 20, 20], "relu", "l2", 0, False)

ann, hist = neural_network.fit_ann(x_train, y_train, layers, epochs=2_000)

print(f"Training MAE: {ann.evaluate(x_train, y_train)[1]}")
print(f"Testing MAE: {ann.evaluate(x_test, y_test)[1]}")


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
            x_te_masked.append(np.zeros(x_test.shape[0]))
    x_te_masked = np.stack(x_te_masked).T
    scores.append(ann.evaluate(x_te_masked, y_test)[1])

plot_features.plot_feat_cumulative(scores)


#%%

key_features = i_ref.columns[ranking][:25]
key_feat_ind = ranking[:25]

print(key_features)

x_tr_filt = x_train[:, key_feat_ind]
x_te_filt = x_test[:, key_feat_ind]

layers = neural_network.get_layers([50, 20, 20], "relu", "l2", 0, False)
new_ann, hew_hist = neural_network.fit_ann(x_tr_filt, y_train, layers)

print(f"Training MAE: {ann.evaluate(x_train, y_train)[1]}")
print(f"Testing MAE: {ann.evaluate(x_test, y_test)[1]}")
