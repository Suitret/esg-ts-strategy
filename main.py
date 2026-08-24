"""
main.py
=======
Orchestrates the whole project, steps 1-11. Run with:
    python3 main.py

Produces:
  outputs/predictive_metrics.csv   - RMSE/MAE/dir.acc/rank-IC per model per fold
  outputs/strategy_metrics.csv     - Sharpe/CAGR/MaxDD/etc per model
  outputs/equity_curves.png        - cumulative return of each strategy vs benchmark
  outputs/predictions_sample.csv   - raw out-of-sample predictions (for audit)
  outputs/fold_diagnostics.png     - per-fold Sharpe/IC to show WHERE models fail
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_pipeline import fetch_data, clean_and_align, to_panel
from features import add_features, train_test_row_report
from walk_forward import make_walk_forward_folds, split_panel
from baselines import NaiveZero, NaiveDrift, RidgeBaseline
from ml_models import GBMModel
from dl_model import LSTMModel, build_sequences
from backtest import run_backtest
from evaluate import predictive_metrics

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# ESG / clean-energy themed universe + one conventional-energy and one
# broad-market ticker for contrast, plus SPY purely as a reporting benchmark.
TICKERS = ["ICLN", "TAN", "PBW", "QCLN", "FAN", "LIT", "XLE"]
BENCHMARK = "SPY"
START, END = "2018-01-01", "2025-12-31"
HORIZON = 1          # predict next-day return
LOOKBACK = 20         # LSTM sequence length
N_SPLITS = 6
TOP_K = 2
COMMISSION_BPS, SLIPPAGE_BPS = 5, 10

FEATURE_COLS = None  # filled in after feature engineering


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    # ---------- Steps 1-2: acquire, clean & align ----------
    log("Step 1-2: fetching + cleaning data")
    all_tickers = TICKERS + [BENCHMARK]
    raw, source = fetch_data(all_tickers, START, END, use_live=True)
    cleaned, clean_report = clean_and_align(raw)
    bench_df = cleaned.pop(BENCHMARK) if BENCHMARK in cleaned else None
    panel = to_panel(cleaned)
    log(f"data source = {source} | panel shape = {panel.shape}")

    # ---------- Step 3: features (causal) ----------
    log("Step 3: engineering causal lag/rolling features")
    feat = add_features(panel, horizon=HORIZON)
    train_test_row_report(feat, horizon=HORIZON)

    target_col = f"target_fwd_ret_{HORIZON}d"
    global FEATURE_COLS
    FEATURE_COLS = [c for c in feat.columns if c not in
                    ["open", "high", "low", "close", "volume", target_col]]
    log(f"{len(FEATURE_COLS)} features: {FEATURE_COLS}")

    # ---------- Step 7 setup: walk-forward folds ----------
    all_dates = feat.index.get_level_values("date").unique()
    folds = make_walk_forward_folds(all_dates, n_splits=N_SPLITS, purge=HORIZON, embargo=2, min_train=400)
    log(f"{len(folds)} walk-forward folds")
    for f in folds:
        log(f"  fold {f['fold']}: train {f['train'][0].date()}..{f['train'][-1].date()} "
            f"({len(f['train'])}d) -> test {f['test'][0].date()}..{f['test'][-1].date()} ({len(f['test'])}d)")

    models_tabular = {
        "naive_zero": NaiveZero(),
        "naive_drift": NaiveDrift(),
        "ridge": RidgeBaseline(alpha=5.0),
        "gbm": GBMModel(),
    }

    pred_records = []       # long-format predictions across all models/folds
    pred_metric_rows = []

    for fold in folds:
        fi = fold["fold"]
        X_tr, y_tr, X_te, y_te, tr_full, te_full = split_panel(
            feat, fold["train"], fold["test"], target_col, FEATURE_COLS)
        if len(X_tr) < 50 or len(X_te) < 5:
            log(f"  fold {fi}: skipping (too few rows after dropna)")
            continue

        for name, model in models_tabular.items():
            model.fit(X_tr.values, y_tr.values)
            preds = model.predict(X_te)
            m = predictive_metrics(y_te.values, preds)
            m.update({"model": name, "fold": fi})
            pred_metric_rows.append(m)
            rec = te_full[[target_col]].copy()
            rec["pred"] = preds
            rec["model"] = name
            rec["fold"] = fi
            pred_records.append(rec.reset_index())

        # ---------- Step 6: LSTM (sequence model), same fold, causal windowing ----------
        Xtr_seq, ytr_seq, meta_tr = build_sequences(tr_full, FEATURE_COLS, target_col, lookback=LOOKBACK)
        Xte_seq, yte_seq, meta_te = build_sequences(te_full, FEATURE_COLS, target_col, lookback=LOOKBACK)
        # small held-out validation slice from the END of train (still causal:
        # it's chronologically before test, just used for early stopping)
        if len(Xtr_seq) > 100:
            val_cut = int(len(Xtr_seq) * 0.85)
            Xtr_fit, ytr_fit = Xtr_seq[:val_cut], ytr_seq[:val_cut]
            Xval, yval = Xtr_seq[val_cut:], ytr_seq[val_cut:]
        else:
            Xtr_fit, ytr_fit, Xval, yval = Xtr_seq, ytr_seq, None, None

        if len(Xtr_fit) >= 50 and len(Xte_seq) >= 5:
            lstm = LSTMModel(n_features=len(FEATURE_COLS), lookback=LOOKBACK, hidden_size=32,
                              epochs=20, lr=1e-3, batch_size=128, seed=fi)
            lstm.fit(Xtr_fit, ytr_fit, Xval, yval, verbose=False)
            preds_lstm = lstm.predict(Xte_seq)
            m = predictive_metrics(yte_seq, preds_lstm)
            m.update({"model": "lstm", "fold": fi})
            pred_metric_rows.append(m)
            rec = pd.DataFrame(meta_te, columns=["date", "ticker"])
            rec[target_col] = yte_seq
            rec["pred"] = preds_lstm
            rec["model"] = "lstm"
            rec["fold"] = fi
            pred_records.append(rec)
            log(f"  fold {fi}: trained all models incl. LSTM "
                f"({len(Xtr_fit)} train seqs, {len(Xte_seq)} test seqs)")
        else:
            log(f"  fold {fi}: not enough sequence rows for LSTM, skipping DL model this fold")

    pred_df_all = pd.concat(pred_records, ignore_index=True)
    pred_df_all = pred_df_all.rename(columns={target_col: "actual"})
    pred_metrics_df = pd.DataFrame(pred_metric_rows)

    pred_df_all.to_csv(os.path.join(OUT, "predictions_sample.csv"), index=False)
    pred_metrics_df.to_csv(os.path.join(OUT, "predictive_metrics_by_fold.csv"), index=False)

    # ---------- aggregate predictive metrics across folds ----------
    agg = pred_metrics_df.groupby("model").agg(
        rmse=("rmse", "mean"), mae=("mae", "mean"),
        directional_accuracy=("directional_accuracy", "mean"),
        rank_ic=("rank_ic", "mean"), n=("n", "sum")
    ).round(5).sort_values("rank_ic", ascending=False)
    agg.to_csv(os.path.join(OUT, "predictive_metrics.csv"))
    log("\n=== OUT-OF-SAMPLE PREDICTIVE METRICS (avg across folds) ===\n" + agg.to_string())

    # ---------- Steps 9-11: backtest each model's strategy ----------
    log("Step 9-11: running backtests (with transaction costs + slippage)")
    strategy_rows = []
    equity_curves = {}
    for name in pred_df_all["model"].unique():
        sub = pred_df_all[pred_df_all["model"] == name].set_index(["date", "ticker"]).sort_index()
        pred_series = sub[["pred"]]
        actual_series = sub["actual"]
        results, metrics = run_backtest(pred_series, actual_series, top_k=TOP_K,
                                         long_only=False, commission_bps=COMMISSION_BPS,
                                         slippage_bps=SLIPPAGE_BPS)
        metrics["model"] = name
        strategy_rows.append(metrics)
        equity_curves[name] = results["equity"]

    strat_df = pd.DataFrame(strategy_rows).set_index("model").sort_values("sharpe", ascending=False)
    strat_df.to_csv(os.path.join(OUT, "strategy_metrics.csv"))
    log("\n=== STRATEGY (NET OF COSTS), DAILY REBALANCE ===\n" + strat_df.round(4).to_string())

    # ---------- turnover-mitigation comparison: rebalance weekly instead of daily ----------
    log("Running turnover-mitigation comparison (weekly rebalance)")
    weekly_rows = []
    for name in pred_df_all["model"].unique():
        sub = pred_df_all[pred_df_all["model"] == name].set_index(["date", "ticker"]).sort_index()
        results_w, metrics_w = run_backtest(sub[["pred"]], sub["actual"], top_k=TOP_K,
                                             long_only=False, commission_bps=COMMISSION_BPS,
                                             slippage_bps=SLIPPAGE_BPS, rebalance_every=5)
        metrics_w["model"] = name
        weekly_rows.append(metrics_w)
    weekly_df = pd.DataFrame(weekly_rows).set_index("model").sort_values("sharpe", ascending=False)
    weekly_df.to_csv(os.path.join(OUT, "strategy_metrics_weekly_rebalance.csv"))
    log("\n=== STRATEGY (NET OF COSTS), WEEKLY REBALANCE (turnover mitigation) ===\n" + weekly_df.round(4).to_string())

    # benchmark buy-and-hold equity curve (equal-weight ESG universe AND SPY if we have it)
    bench_curves = {}
    ew_ret = panel[target_col if target_col in panel.columns else "close"]
    close_wide = panel["close"].unstack("ticker")
    ew_daily_ret = close_wide.pct_change().mean(axis=1)
    bench_curves["equal_weight_buy_hold"] = (1 + ew_daily_ret.reindex(
        pred_df_all.set_index(["date", "ticker"]).index.get_level_values("date").unique()
    ).fillna(0)).cumprod()
    if bench_df is not None:
        spy_ret = bench_df["close"].pct_change()
        common_dates = equity_curves[list(equity_curves.keys())[0]].index
        bench_curves["SPY_buy_hold"] = (1 + spy_ret.reindex(common_dates).fillna(0)).cumprod()

    # ---------- plots ----------
    log("plotting equity curves and fold diagnostics")
    plt.figure(figsize=(11, 6))
    for name, eq in equity_curves.items():
        plt.plot(eq.index, eq.values, label=f"{name} (strategy)")
    for name, eq in bench_curves.items():
        plt.plot(eq.index, eq.values, "--", label=name, alpha=0.7)
    plt.title(f"Out-of-sample equity curves (walk-forward, net of {COMMISSION_BPS}+{SLIPPAGE_BPS}bps costs)\n"
               f"data source: {source.upper()}" + (" -- SYNTHETIC DEMO DATA, not real market data" if source == "synthetic" else ""))
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "equity_curves.png"), dpi=140)
    plt.close()

    fold_ic = pred_metrics_df.pivot(index="fold", columns="model", values="rank_ic")
    plt.figure(figsize=(10, 5))
    fold_ic.plot(marker="o", ax=plt.gca())
    plt.axhline(0, color="black", lw=0.8)
    plt.title("Rank IC (predictive skill) per walk-forward fold, per model\n(shows WHERE each model works or breaks)")
    plt.ylabel("Spearman rank IC")
    plt.xlabel("fold (chronological)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fold_diagnostics.png"), dpi=140)
    plt.close()

    log(f"\nData source used: {source}")
    log("DONE. See outputs/ for CSVs and PNGs.")
    return agg, strat_df, pred_metrics_df


if __name__ == "__main__":
    main()
