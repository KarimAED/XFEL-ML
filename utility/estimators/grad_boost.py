from deprecated import deprecated

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb
import numpy as np


mae = lambda x, y: np.mean(np.abs(x-y))


@deprecated("The histogram gradient boosting regressor is experimental and should not be used.")
def fit_grad_boost(x_tr, y_tr, reg=1000, lr=0.1, iter=1000, samples=20, leaves=31):
    reg = HistGradientBoostingRegressor(l2_regularization=reg, verbose=1, learning_rate=lr,
                                        early_stopping=False, random_state=1, max_iter=iter,
                                        min_samples_leaf=samples, max_leaf_nodes=leaves)
    reg.fit(x_tr, y_tr)
    return reg


def fit_xgboost(x_tr, y_tr, n_est=20, n_jobs=4, gamma=0.1, l2=0.0):
    reg = xgb.XGBRegressor(random_state=1, n_estimators=n_est, verbosity=2,
                           n_jobs=n_jobs, gamma=gamma, reg_lambda=l2, tree_method="exact")

    reg.fit(x_tr, y_tr)
    return reg
