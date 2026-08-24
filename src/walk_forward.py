"""
walk_forward.py
================
Step 7: walk-forward validation. This is the single most important piece of
methodology in the whole project, so it gets its own module and extra
comments.

WHY NOT A RANDOM K-FOLD SPLIT?
A random shuffled split lets the model train on data from AFTER the point
it's being tested on. In a real deployment you never have that luxury --
you only ever have the past. A model validated with random k-fold can look
great and then fail completely in production. This is the single most common
mistake in applied financial ML.

WHAT WE DO INSTEAD: expanding-window walk-forward with a PURGE GAP.
    [-------- train --------][ purge ][-- test --]
    [-------------- train --------------][ purge ][-- test --]
    [------------------ train -----------------------][ purge ][-- test --]

- The training window EXPANDS (or optionally rolls) forward in time.
- Test windows are contiguous, non-overlapping, and always strictly AFTER
  the training window that produced the model being tested.
- The PURGE GAP drops `horizon` rows between train and test. Why: our target
  at row t uses close price at t+horizon. If horizon=5 and we test starting
  the day right after the last training day, the last few training labels
  actually depend on prices that fall inside (or after) the test window --
  a subtle leak. Purging `horizon` rows removes any label that overlaps the
  test period.
- We additionally support an embargo (extra buffer) for safety since
  autocorrelated features (rolling stats) can smear information across the
  gap too.

Every fold's model is retrained from scratch, and every fold's test
predictions are OUT OF SAMPLE relative to that fold's model, and IN
CHRONOLOGICAL ORDER -- exactly like a live deployment where you retrain
periodically and only ever trade forward.
"""

import numpy as np
import pandas as pd


def make_walk_forward_folds(dates: pd.DatetimeIndex, n_splits=5, test_size=None,
                             purge=1, embargo=0, min_train=250):
    """
    dates : sorted unique trading dates in the panel
    Returns list of dicts: {train: DatetimeIndex, test: DatetimeIndex, fold: i}
    """
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    n = len(dates)
    if test_size is None:
        test_size = max(1, (n - min_train) // n_splits)

    folds = []
    train_end = min_train
    fold_i = 0
    while train_end + purge + embargo + test_size <= n and fold_i < n_splits:
        train_dates = dates[:train_end]
        test_start = train_end + purge + embargo
        test_end = test_start + test_size
        test_dates = dates[test_start:test_end]
        folds.append({"fold": fold_i, "train": train_dates, "test": test_dates})
        train_end = test_end  # expanding window: next fold's train includes this fold's test period
        fold_i += 1
    return folds


def split_panel(feat: pd.DataFrame, train_dates, test_dates, target_col, feature_cols):
    tr = feat.loc[(train_dates, slice(None)), :].dropna(subset=feature_cols + [target_col])
    te = feat.loc[(test_dates, slice(None)), :].dropna(subset=feature_cols + [target_col])
    X_tr, y_tr = tr[feature_cols], tr[target_col]
    X_te, y_te = te[feature_cols], te[target_col]
    return X_tr, y_tr, X_te, y_te, tr, te
