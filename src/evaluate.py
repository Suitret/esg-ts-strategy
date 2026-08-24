"""
evaluate.py
===========
Step 11 (predictive half): metrics for "is the forecast any good", kept
separate from strategy P&L (backtest.py) on purpose -- Step 11 explicitly
asks for BOTH, because they can disagree. A model can have poor RMSE but
good directional accuracy / IC and still make money; a model can have
decent RMSE and still lose money after realistic costs and 50/50 direction
calls. Reporting only one of these is a classic way this kind of analysis
misleads people.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def predictive_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    ic = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))  # rank IC, standard in finance
    naive_rmse = float(np.sqrt(np.mean(y_true ** 2)))  # RMSE of always predicting 0
    return {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": dir_acc,
        "rank_ic": ic,
        "rmse_vs_naive_zero": rmse / naive_rmse if naive_rmse else np.nan,
        "n": int(len(y_true)),
    }
