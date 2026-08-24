"""
backtest.py
===========
Step 9: transaction costs & slippage. Step 11 (strategy performance).

A model with great RMSE can still LOSE MONEY as a strategy once you account
for the fact that trading isn't free. This module turns predictions into
positions into a realistic P&L.

MECHANICS
- Signal -> position: cross-sectional ranking. Each day, rank all assets by
  predicted forward return; go long the top-K, short the bottom-K (or
  long-only top-K if `long_only=True`), equal-weighted within each side.
  This is a standard, honest way to trade a return-forecasting signal --
  it's dollar-neutral (roughly market-neutral) so it isolates whether the
  MODEL has skill, rather than whether the market went up that period.
- Costs: every time a position CHANGES, you pay
      cost = turnover * (commission_bps + slippage_bps) / 10000
  Turnover = |new_weight - old_weight| per asset, summed. This penalizes
  a model that flips its mind every day (high turnover) much more than one
  with stable convictions -- which is realistic, and is exactly the kind of
  thing that makes naive "paper P&L" numbers from a Kaggle-style notebook
  wildly optimistic.
- We apply the position decided using day-t's close-of-day signal to the
  return realised from day t to day t+1 (i.e. you trade at t's close,
  you're exposed to the t->t+1 return) -- consistent with the causal target
  definition in features.py.
"""

import numpy as np
import pandas as pd


def predictions_to_positions(pred_df: pd.DataFrame, top_k=3, long_only=False):
    """
    pred_df: index=(date,ticker), column 'pred' = predicted forward return.
    Returns weights_df: index=(date,ticker), column 'weight' in [-1,1]-ish,
    such that weights sum to 0 (long/short) or to 1 (long-only) each day.
    """
    def rank_day(g):
        g = g.sort_values("pred", ascending=False)
        n = len(g)
        k = min(top_k, n // 2 if not long_only else n)
        w = pd.Series(0.0, index=g.index)
        if k == 0:
            return w
        longs = g.index[:k]
        if long_only:
            w.loc[longs] = 1.0 / k
        else:
            shorts = g.index[-k:]
            w.loc[longs] = 0.5 / k
            w.loc[shorts] = -0.5 / k
        return w

    weights = pred_df.groupby(level="date", group_keys=False).apply(rank_day)
    weights.name = "weight"
    return weights.to_frame()


def run_backtest(pred_df: pd.DataFrame, fwd_ret_col_actual: pd.Series,
                  top_k=3, long_only=False, commission_bps=5, slippage_bps=10,
                  rebalance_every=1):
    """
    pred_df: index=(date,ticker), col 'pred'
    fwd_ret_col_actual: same index, ACTUAL realised forward return (ground truth),
                         used only to compute P&L, never fed back into the model.
    rebalance_every: trade only every N trading days; positions are HELD
        (forward-filled) between rebalances. This directly controls turnover,
        which is often the dominant cost driver for a daily long/short signal
        on a small universe -- see reports/findings.md for why this matters.
    Returns a DataFrame of daily portfolio results and a dict of summary metrics.
    """
    weights = predictions_to_positions(pred_df, top_k=top_k, long_only=long_only)["weight"]
    df = pd.concat([weights.rename("weight"), fwd_ret_col_actual.rename("fwd_ret")], axis=1).dropna()

    wide_w = df["weight"].unstack("ticker").fillna(0.0).sort_index()
    wide_r = df["fwd_ret"].unstack("ticker").fillna(0.0).sort_index()

    if rebalance_every > 1:
        # only the signal computed on rebalance dates is acted on; on other
        # days we hold whatever we already own (forward-fill), which is what
        # a real desk would do to control turnover/cost drag.
        rebal_mask = np.arange(len(wide_w)) % rebalance_every == 0
        held = wide_w.copy()
        held.iloc[~rebal_mask] = np.nan
        wide_w = held.ffill().fillna(0.0)

    gross_ret = (wide_w * wide_r).sum(axis=1)

    turnover = wide_w.diff().abs().sum(axis=1)
    turnover.iloc[0] = wide_w.iloc[0].abs().sum()  # cost of putting on the very first positions
    cost_rate = (commission_bps + slippage_bps) / 10000.0
    costs = turnover * cost_rate

    net_ret = gross_ret - costs

    equity = (1 + net_ret).cumprod()
    equity_gross = (1 + gross_ret).cumprod()

    results = pd.DataFrame({
        "gross_ret": gross_ret, "net_ret": net_ret, "turnover": turnover,
        "costs": costs, "equity": equity, "equity_gross": equity_gross,
    })

    metrics = summarize_strategy(net_ret, turnover)
    metrics["gross_sharpe"] = annualized_sharpe(gross_ret)
    return results, metrics


def annualized_sharpe(daily_ret, periods=252):
    if daily_ret.std() == 0 or len(daily_ret) < 2:
        return 0.0
    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(periods))


def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    return float(dd.min())


def summarize_strategy(net_ret: pd.Series, turnover: pd.Series, periods=252):
    equity = (1 + net_ret).cumprod()
    n_years = len(net_ret) / periods
    cagr = float(equity.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 and equity.iloc[-1] > 0 else float("nan")
    downside = net_ret[net_ret < 0]
    sortino = float(net_ret.mean() / downside.std() * np.sqrt(periods)) if len(downside) > 1 and downside.std() > 0 else 0.0
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": cagr,
        "ann_vol": float(net_ret.std() * np.sqrt(periods)),
        "sharpe": annualized_sharpe(net_ret, periods),
        "sortino": sortino,
        "max_drawdown": max_drawdown(equity),
        "avg_daily_turnover": float(turnover.mean()),
        "hit_rate": float((net_ret > 0).mean()),
        "n_days": len(net_ret),
    }
