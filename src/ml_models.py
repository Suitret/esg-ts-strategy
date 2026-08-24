"""
ml_models.py
============
Step 5: a real ML model. Gradient-boosted trees (sklearn's
GradientBoostingRegressor / HistGradientBoostingRegressor) are the standard
strong tabular baseline in industry -- they handle nonlinearity and feature
interactions that Ridge can't, without needing sequence structure the way an
LSTM does. This is deliberately the "ML" tier between the statistical
baseline and the deep-learning model, matching the project's 5-model ladder:
naive -> statistical -> ML -> deep learning.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


class GBMModel:
    def __init__(self, max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0):
        self.model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            l2_regularization=l2_regularization,
            random_state=0,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self, feature_names):
        # HistGBR doesn't expose feature_importances_ directly; use a
        # RandomForest fit as a cheap proxy purely for interpretability plots.
        rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0, n_jobs=-1)
        return rf, feature_names
