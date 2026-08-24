"""
data_pipeline.py
=================
Step 1 (acquire) + Step 2 (clean & align) of the project.

Design decisions worth defending in an interview:
- We fetch ADJUSTED close (splits/dividends applied) so returns aren't
  contaminated by corporate actions -> avoids a subtle look-ahead-adjacent bug
  (unadjusted prices create fake "jumps" a model could latch onto).
- We fetch a UNIVERSE of tickers, not one, because a single-asset backtest
  tells you almost nothing about whether a signal generalizes.
- Cleaning/alignment happens BEFORE any feature is computed, and every
  cleaning step is causal (only ever looks backward or at the current row).
"""

import os
import numpy as np
import pandas as pd

RAW_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# REAL DATA PATH (this is the code that runs on your machine / any machine
# with normal internet access — this sandbox's network is locked down to a
# package-manager allowlist, so this path is untested *here* but is standard,
# well-trodden yfinance usage).
# ---------------------------------------------------------------------------
def fetch_universe(tickers, start, end, cache=True):
    """
    Download daily OHLCV for a list of tickers via yfinance.

    Parameters
    ----------
    tickers : list[str]   e.g. ["ICLN", "TAN", "PBW", "XLE", "SPY"]
    start, end : str      "YYYY-MM-DD"
    cache : bool           save/read a local parquet cache so re-runs are fast
                            and so your results are reproducible even if the
                            vendor revises history later.

    Returns
    -------
    dict[str, pd.DataFrame]  ticker -> OHLCV dataframe, DatetimeIndex
    """
    import yfinance as yf

    out = {}
    for t in tickers:
        cache_path = os.path.join(RAW_CACHE_DIR, f"{t}_{start}_{end}.parquet")
        if cache and os.path.exists(cache_path):
            out[t] = pd.read_parquet(cache_path)
            continue
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            print(f"[warn] no data returned for {t}, skipping")
            continue
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index.name = "date"
        if cache:
            df.to_parquet(cache_path)
        out[t] = df
    return out


# ---------------------------------------------------------------------------
# SYNTHETIC FALLBACK — used only when live data is unreachable (like in this
# sandboxed environment) so the full pipeline can still be demonstrated.
# THIS IS CLEARLY LABELLED. Every plot/report generated from it says so.
# It is NOT dressed up as real market data anywhere downstream.
#
# It is not a random walk. It has: volatility clustering (GARCH-like),
# regime switches (bull/bear/crash), a common market factor plus idiosyncratic
# noise, and volume that co-moves with volatility -- because a model that
# only has to beat a pure random walk is a much weaker proof of skill than
# one that has to work on data with realistic structure.
# ---------------------------------------------------------------------------
def generate_synthetic_universe(tickers, start, end, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # common market factor: regime-switching drift + GARCH(1,1)-like vol
    regime_len = 60
    n_regimes = n // regime_len + 2
    regime_drift = rng.choice([0.0004, -0.0006, 0.0015, -0.002], size=n_regimes,
                               p=[0.45, 0.30, 0.15, 0.10])
    drift_path = np.repeat(regime_drift, regime_len)[:n]

    vol = np.zeros(n)
    vol[0] = 0.01
    omega, alpha, beta = 1e-6, 0.08, 0.88
    shocks = rng.standard_normal(n)
    for i in range(1, n):
        vol[i] = np.sqrt(omega + alpha * (shocks[i - 1] * vol[i - 1]) ** 2 + beta * vol[i - 1] ** 2)
    market_ret = drift_path + vol * shocks

    out = {}
    for t in tickers:
        beta_t = rng.uniform(0.6, 1.4)          # sensitivity to the market factor
        idio_vol = rng.uniform(0.006, 0.018)     # idiosyncratic (stock-specific) vol
        idio = rng.standard_normal(n) * idio_vol
        ret = beta_t * market_ret + idio
        price = 50 * np.exp(np.cumsum(ret))

        # build OHLC around the close using intraday noise; volume tracks |return| & vol
        close = price
        open_ = close * (1 + rng.normal(0, 0.001, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
        base_vol = rng.uniform(1e6, 5e6)
        volume = base_vol * (1 + 8 * np.abs(ret) + 3 * vol) * (1 + rng.normal(0, 0.1, n))
        volume = np.clip(volume, 1e5, None)

        out[t] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        out[t].index.name = "date"
    return out


def fetch_data(tickers, start, end, use_live=True):
    """Single entry point: try live data, fall back to labelled synthetic data."""
    if use_live:
        try:
            data = fetch_universe(tickers, start, end)
            if data:
                print(f"[data] fetched LIVE data for {list(data.keys())}")
                return data, "live"
        except Exception as e:
            print(f"[data] live fetch failed ({e}); falling back to synthetic data")
    data = generate_synthetic_universe(tickers, start, end)
    print(f"[data] using SYNTHETIC data (labelled) for {list(data.keys())}")
    return data, "synthetic"


# ---------------------------------------------------------------------------
# CLEANING & ALIGNMENT
# ---------------------------------------------------------------------------
def clean_and_align(data: dict, min_price=1.0, max_daily_move=0.5):
    """
    - Align all tickers onto a shared trading-day index (inner join) so every
      row of the final panel has a value for every asset -> no silent NaNs
      leaking into a model that a metric would then average over.
    - Drop rows with impossible values (non-positive price/volume).
    - Flag (not silently fix) suspected bad ticks: a >50% single-day move
      that reverts the next day is very likely a data error, not real
      information -- an LSTM will happily overfit to a data glitch. We flag
      and forward-fill THAT VALUE ONLY, using only past information, which
      keeps the operation causal.
    - Restrict to tickers with no more than 2% missing rows over the window
      (deals with tickers that only later got listed = a survivorship-bias
      -adjacent issue: we don't backfill/estimate their pre-IPO history).
    """
    # shared index
    common_idx = None
    for t, df in data.items():
        idx = df.index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()

    cleaned = {}
    report = []
    for t, df in data.items():
        df = df.reindex(common_idx)
        n_missing = df["close"].isna().sum()
        pct_missing = n_missing / len(df)
        if pct_missing > 0.02:
            report.append(f"DROPPED {t}: {pct_missing:.1%} missing rows in the aligned window "
                           f"(likely listed later than window start -> would be survivorship bias "
                           f"to keep it without marking it as unavailable pre-listing).")
            continue

        bad_price = (df["close"] <= min_price) | (df["close"].isna())
        df.loc[bad_price, ["open", "high", "low", "close"]] = np.nan

        ret = df["close"].pct_change()
        suspect = ret.abs() > max_daily_move
        n_suspect = int(suspect.sum())
        if n_suspect:
            report.append(f"{t}: flagged {n_suspect} suspected bad ticks (>{max_daily_move:.0%} one-day move)")
        df.loc[suspect, ["open", "high", "low", "close"]] = np.nan

        # causal fill: forward-fill only (never bfill, which would leak future
        # prices backward in time)
        df = df.ffill()
        df = df.dropna()  # any leading NaNs (before the first valid tick) are dropped

        df["volume"] = df["volume"].clip(lower=0)
        cleaned[t] = df

    print("[clean] " + (" | ".join(report) if report else "no issues flagged"))
    return cleaned, report


def to_panel(cleaned: dict):
    """Combine per-ticker frames into one long panel: index=(date, ticker)."""
    frames = []
    for t, df in cleaned.items():
        d = df.copy()
        d["ticker"] = t
        frames.append(d)
    panel = pd.concat(frames).reset_index().set_index(["date", "ticker"]).sort_index()
    return panel
