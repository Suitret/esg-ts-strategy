"""
features.py
============
Step 3: engineer lag/rolling features. Step 8 (no look-ahead) is enforced
HERE, not bolted on later -- it's a property of how every feature is built.

THE LOOK-AHEAD RULE we follow everywhere in this file:
    A feature stored on row t may only use information available AT OR
    BEFORE the close of day t. The label/target stored on row t is the
    return realised AFTER day t (i.e. buying at day t's close and selling
    at day t+h's close). So every (feature_t, label_t) pair is causally
    valid: you could have computed feature_t in real time on day t and only
    found out label_t on day t+h.

Concretely:
- rolling/lag features use pandas .shift(1) BEFORE rolling, or windows that
  end at t (never centered, never using .shift(-k)).
- the target column is explicitly offset forward and is the ONLY forward-
  looking column in the frame -- and it's clearly named `target_fwd_ret_{h}d`.
- cross-sectional features (e.g. z-score vs the universe on day t) use only
  day-t information from other tickers, which is fine (no time travel), but
  we compute them per-day using only tickers present that day.
"""

import numpy as np
import pandas as pd


def _per_ticker(panel: pd.DataFrame, func):
    """Apply a per-ticker function to a (date,ticker)-indexed panel."""
    out = []
    for t, df in panel.groupby(level="ticker"):
        d = df.droplevel("ticker").sort_index()
        d = func(d)
        d["ticker"] = t
        out.append(d)
    res = pd.concat(out).reset_index().set_index(["date", "ticker"]).sort_index()
    return res


def add_features(panel: pd.DataFrame, horizon=1) -> pd.DataFrame:
    def build(df):
        df = df.copy()
        ret1 = df["close"].pct_change()

        # --- lag features: past returns, several lookbacks ---
        for lag in [1, 2, 3, 5, 10]:
            df[f"lag_ret_{lag}d"] = ret1.shift(lag - 1)  # shift(1)=yesterday's realised return

        # --- rolling / momentum features (windows END at t, use only past closes) ---
        for w in [5, 10, 20, 60]:
            df[f"mom_{w}d"] = df["close"].pct_change(w)                 # w-day momentum
            df[f"vol_{w}d"] = ret1.rolling(w).std()                     # realised volatility
            df[f"ma_gap_{w}d"] = df["close"] / df["close"].rolling(w).mean() - 1  # dist. from MA

        # --- volume features ---
        df["vol_chg_5d"] = df["volume"].pct_change(5)
        df["vol_z_20d"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()

        # --- technical: RSI(14), computed causally with an expanding/rolling EMA ---
        delta = df["close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - 100 / (1 + rs)

        # --- realised range (Parkinson-style vol proxy) ---
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]

        # --- target: FORWARD return, the only forward-looking column ---
        df[f"target_fwd_ret_{horizon}d"] = df["close"].shift(-horizon) / df["close"] - 1

        return df

    feat = _per_ticker(panel, build)

    # cross-sectional feature: today's momentum rank vs the rest of the
    # universe ON THE SAME DAY -- causally fine, no future info, but adds a
    # "relative strength" signal that pure single-asset features miss.
    feat["mom_20d_xrank"] = feat.groupby(level="date")["mom_20d"].rank(pct=True)

    return feat


def train_test_row_report(feat: pd.DataFrame, horizon=1):
    target_col = f"target_fwd_ret_{horizon}d"
    n_total = len(feat)
    n_valid = feat[target_col].notna().sum() if target_col in feat.columns else 0
    print(f"[features] built {feat.shape[1]} columns, {n_total} rows "
          f"({n_valid} rows have a usable non-NaN target)")
    return feat
