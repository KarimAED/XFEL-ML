"""
Code to fit an ann regressor to the delay data without use of feature selection.
Plots and data are saved to the results/no_feature_selection subfolder.
"""
import numpy as np

from utility.estimators import grad_boost, neural_network
from utility.plotting import plot_fit
from doublePulse2017.setup import get_data


#%%

x_train, x_test, y_train, y_test, input_reference, output_reference = get_data()

print(input_reference)
print(output_reference)

#%%
layers = neural_network.get_layers([50, 50, 20], "relu", "l2", 0, False)

ann, hist = neural_network.fit_ann(x_train, y_train, layers, epochs=5_000, rate=0.001)
#%%

print(f"Training MAE: {ann.evaluate(x_train, y_train)[1]}")
print(f"Testing MAE: {ann.evaluate(x_test, y_test)[1]}")

#%%
predictions = ann.predict(x_test).T[0]

out_ref = output_reference["Delays"]

test_out = y_test*out_ref.loc["test_std"]+out_ref.loc["test_mean"]
test_pred = predictions*out_ref.loc["test_std"]+out_ref.loc["test_mean"]

train_out = y_train*out_ref.loc["train_std"]+out_ref.loc["train_mean"]
train_pred = ann.predict(x_train).T[0]*out_ref.loc["train_std"]+out_ref.loc["train_mean"]

np.savez("doublePulse2017/results/ex_1_ann_feat/ann_pred.npz",
         train_out=train_out, train_pred=train_pred, test_out=test_out, test_pred=test_pred)

#%%

eval_mae = ann.evaluate(x_test, y_test)[1]*out_ref.loc["test_std"]

ticks = [-15, -10, -5, 0, 5, 10, 15, 20, 25]

plot_fit.plot_pvm(test_out, test_pred,
                  f"ANN; MAE: {round(eval_mae, 2)}fs",
                  "Expected Delay in fs", "Predicted Delay in fs",
                  ticks, ticks, "doublePulse2017/results/ex_1_ann_feat/ann_hist2d")

