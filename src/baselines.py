"""
baselines.py
============
Step 4/5: naive & statistical baselines. If a fancy model can't beat these,
the fancy model isn't earning its complexity. This is the bar everything
else has to clear.

- NaiveZero: predicts 0 return (i.e. "price won't move"). Sounds dumb, but
  in efficient, noisy markets it is a brutally hard baseline to beat on
  pure RMSE, because most of a stock's forward return really is unpredictable
  noise. Beating it on RMSE is rare and suspicious; beating it on
  directional accuracy / strategy P&L is the more honest test.
- NaiveDrift (persistence): predicts tomorrow's return = today's realised
  return (momentum baseline).
- Ridge regression on the engineered features: a genuine statistical
  baseline that any of the ML/DL models need to beat to justify their
  complexity.
"""

import numpy as np
from sklearn.linear_model import Ridge


class NaiveZero:
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))


class NaiveDrift:
    """Predicts using the most recent realised 1-day return as next-day forecast."""
    def fit(self, X, y): return self
    def predict(self, X):
        col = "lag_ret_1d" if "lag_ret_1d" in X.columns else X.columns[0]
        return X[col].values


class RidgeBaseline:
    def __init__(self, alpha=5.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
