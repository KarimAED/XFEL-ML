import logging
from deprecated import deprecated

# deprecated imports for HistGradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

import xgboost as xgb
import numpy as np

logger = logging.getLogger(__name__)


@deprecated(
    "The histogram gradient boosting regressor is experimental and should not be used."
)
def fit_grad_boost(
    x_tr, y_tr, reg=1, lr=0.1, iterations=1000, samples=20, leaves=31
):
    """
    Deprecated function to fit a gradient boosting regressor using experimental HistGradientBoostingRegressor from sklearn

    :param x_tr: 2d-array, training input data, events along 1st axis, features along 2nd axis
    :param y_tr: 1d-array, training target data, events along 1st axis
    :param reg: float, coefficient of l2-regularization to be applied
    :param lr: float, learning rate
    :param iterations: int, number of regressors
    :param samples: int, number of samples per decision tree node
    :param leaves: int, number of leaves per decision tree
    :return: HistGradientBoostingRegressor, fitted regressor with the given data and parameters
    """

    logger.warning(
        "Called deprecated function fit_grad_boost which relies on experimental Regressor."
    )
    # initialize regressor
    reg = HistGradientBoostingRegressor(
        l2_regularization=reg,
        verbose=1,
        learning_rate=lr,
        early_stopping=False,
        random_state=1,
        max_iter=iterations,
        min_samples_leaf=samples,
        max_leaf_nodes=leaves,
    )
    reg.fit(x_tr, y_tr)  # fit regressor
    return reg


def fit_xgboost(x_tr, y_tr, n_est=20, n_jobs=4, gamma=0.1, l2=0.0):
    """
    Function to fit a gradient boosting regressor using XGBoost
    :param x_tr: 2d-array, training input data, events along 1st axis, features along 2nd axis
    :param y_tr: 1d-array, training target data, events along 1st axis
    :param n_est: int, number of decision trees to use
    :param n_jobs: int, number of jobs for parallelisation
    :param gamma: float, learning rate
    :param l2: float, coefficient of l2-regularization to be applied
    :return: xgb.XGBRegressor, fitted regressor with given parameters
    """
    logger.info("Called fit_xgboost")
    # initialize regressor
    reg = xgb.XGBRegressor(
        random_state=1,
        n_estimators=n_est,
        verbosity=2,
        n_jobs=n_jobs,
        gamma=gamma,
        reg_lambda=l2,
        tree_method="exact",
    )

    reg.fit(x_tr, y_tr)  # fit regressor
    return reg
