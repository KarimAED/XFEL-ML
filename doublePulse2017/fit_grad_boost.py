import numpy as np

from utility.estimators import grad_boost
from utility.plotting import plot_fit
from doublePulse2017.setup import get_data


#%%

x_train, x_test, y_train, y_test, input_reference, output_reference = get_data()

print(input_reference)
print(output_reference)

#%%

gradient_booster = grad_boost.fit_xgboost(x_train, y_train)

predictions = gradient_booster.predict(x_test)

print(f"Training MAE: {np.mean(np.abs(gradient_booster.predict(x_train)-y_train))}")
print(f"Testing MAE: {np.mean(np.abs(predictions-y_test))}")

#%%

plot_fit.plot_pvm(y_test, predictions, "XGBoost PvM", "Expected", "Predicted")
