"""
Code to fit an ann regressor to the delay data without use of feature selection.
Plots and data are saved to the results/no_feature_selection subfolder.
"""
import numpy as np

from utility.estimators import grad_boost
from utility.plotting import plot_fit
from doublePulse2017.setup import get_data


#%%
x_train, x_test, y_train, y_test, input_reference, output_reference = get_data()

print(input_reference)
print(output_reference)

#%%
xgb = grad_boost.fit_xgboost(x_train, y_train)

print(f"Training MAE: {grad_boost.mae(xgb.predict(x_train), y_train)}")
print(f"Testing MAE: {grad_boost.mae(xgb.predict(x_test), y_test)}")

#%%
predictions = xgb.predict(x_test)

out_ref = output_reference["Delays"]

test_out = y_test*out_ref.loc["test_std"]+out_ref.loc["test_mean"]
test_pred = predictions*out_ref.loc["test_std"]+out_ref.loc["test_mean"]

train_out = y_train*out_ref.loc["train_std"]+out_ref.loc["train_mean"]
train_pred = xgb.predict(x_train).T[0]*out_ref.loc["train_std"]+out_ref.loc["train_mean"]

np.savez("doublePulse2017/results/ex_2_gb_perf/xgb_pred.npz",
         train_out=train_out, train_pred=train_pred, test_out=test_out, test_pred=test_pred)

#%%

eval_mae = grad_boost.mae(xgb.predict(x_test), y_test)*out_ref.loc["test_std"]

ticks = [-15, -10, -5, 0, 5, 10, 15, 20, 25]

plot_fit.plot_pvm(test_out, test_pred,
                  f"ANN; MAE: {round(eval_mae, 2)}fs",
                  "Expected Delay in fs", "Predicted Delay in fs",
                  ticks, ticks, "doublePulse2017/results/ex_2_gb_perf/gb_hist2d")
