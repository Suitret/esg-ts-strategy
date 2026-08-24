# ESG Financial Time-Series Forecasting & Strategy Backtesting

A from-scratch project demonstrating the full, honest workflow of a
quantitative research task: forecast returns on a small universe of
ESG/clean-energy-themed equities, and rigorously test whether that forecast
would have made a tradeable strategy — not just a good-looking accuracy
number.

It was built to close a specific gap: showing hands-on understanding of
**time-series forecasting (statistical → ML → deep learning)** and
**realistic backtesting methodology (walk-forward validation, no
look-ahead bias, transaction costs/slippage, out-of-sample testing)** —
the exact things a quant-research-adjacent data science role cares about
more than raw model sophistication.

## TL;DR results

- Built a 5-rung model ladder: naive baselines → linear (Ridge) → gradient
  boosted trees (GBM) → LSTM (deep learning, sequence model).
- Evaluated all 5 with **walk-forward validation** (5 chronological folds,
  purge gap to prevent leakage), not a random train/test split.
- None of the models showed a reliable, cost-surviving edge on daily-
  frequency returns — and I show *why*, quantitatively: turnover-driven
  transaction costs turn even a roughly break-even gross signal into a
  strongly negative net strategy, and reducing rebalance frequency
  materially recovers performance. Full writeup: [`reports/findings.md`](reports/findings.md).
- That "it didn't work, and here's the rigorous proof of exactly why and
  what I'd try next" is the actual deliverable — it's what real research
  looks like, and it directly demonstrates the methodology the target role
  cares about (see "why a negative result is the point" below).

## Project structure

```
esg-ts-strategy/
├── main.py                    # orchestrates the entire pipeline end-to-end
├── requirements.txt
├── src/
│   ├── data_pipeline.py       # Steps 1-2: acquire, clean, align
│   ├── features.py            # Step 3: causal lag/rolling feature engineering
│   ├── walk_forward.py        # Step 7: walk-forward CV with purge/embargo
│   ├── baselines.py           # Step 4: naive + linear baselines
│   ├── ml_models.py           # Step 5: gradient boosted trees
│   ├── dl_model.py            # Step 6: LSTM deep-learning model (PyTorch)
│   ├── backtest.py            # Step 9: strategy P&L with costs/slippage
│   └── evaluate.py            # Step 11: predictive metrics (RMSE/IC/etc.)
├── outputs/                   # generated CSVs + PNGs from the last run
└── reports/
    └── findings.md            # Step 12: documented failure cases & analysis
```

Every file has a docstring at the top explaining **why** it's built the way
it is, not just what it does — that's deliberate, so you (or an interviewer
reading the code) can see the reasoning, not just the mechanics.

## How to run it

```bash
pip install -r requirements.txt
python3 main.py
```

This regenerates everything in `outputs/`: predictive metrics, strategy
metrics (daily and weekly rebalance), and two plots (equity curves, and
per-fold rank-IC diagnostics).

### Running on real data

`src/data_pipeline.py` has a genuine `yfinance`-based `fetch_universe()`
function. The **build/demo environment used to develop this project has its
network locked down to package registries only** (pypi, npm, github — no
`finance.yahoo.com`), so `main.py` automatically falls back to a clearly
labelled synthetic data generator when the live fetch fails. On a normal
machine with internet access, `main.py`'s call `fetch_data(..., use_live=True)`
will transparently pull real adjusted OHLCV data instead — **no code changes
needed**, and every downstream step (features, walk-forward, models,
backtest) is written against the same panel format regardless of data
source. This is exactly the situation you should design for in real
research: point-in-time data providers, cached vendors, or local databases
get swapped in and out, so the pipeline is deliberately decoupled from any
one data source.

The synthetic generator itself is not a toy random walk — it has volatility
clustering (GARCH(1,1)), regime switches, and a shared market factor with
per-asset betas — specifically so that the walk-forward / cost / leakage
machinery gets exercised against something with realistic structure, not
something trivially easy or trivially impossible to predict.

## Why each step exists, and what would go wrong without it

**1-2. Acquire, clean, align.** Adjusted prices (splits/dividends handled)
so returns aren't corrupted by corporate actions. Tickers with too much
missing history in the analysis window are **dropped, not backfilled** —
silently estimating a stock's pre-IPO history is a subtle form of
survivorship bias. Suspected bad ticks (implausible one-day moves) are
flagged and forward-filled using only past data — see the sentence-level
comments in `clean_and_align()`.

**3. Feature engineering, and the look-ahead rule.** Every feature at row
`t` uses only information available at or before day `t`'s close, enforced
by using `.shift(k)` with `k >= 1` (or windows ending at `t`) everywhere,
never `.shift(-k)`. The target column is the **only** forward-looking
column in the dataset, and it's explicitly named `target_fwd_ret_{h}d` so
it can't be mistaken for a feature by accident. This single discipline is
the difference between a backtest that means something and one that's
quietly cheating.

**4-5. Naive & statistical baselines, then ML.** `naive_zero` (predict no
move) is a genuinely hard baseline to beat on RMSE at daily frequency —
this is a feature of the project, not a flaw: it's exactly why RMSE alone
is a misleading headline metric for this problem (see findings.md, failure
case #1), and why rank IC / directional accuracy / strategy P&L are all
reported alongside it rather than RMSE in isolation. Ridge regression and
gradient-boosted trees (`HistGradientBoostingRegressor`) sit above that as
genuine statistical/ML baselines any fancier model has to beat to justify
its complexity.

**6. One deep-learning model: LSTM.** Deliberately not a from-scratch TFT
or N-BEATS: those are complex enough that a first attempt mostly teaches
you the library's edge cases rather than time-series methodology. An LSTM
uses the same core idea (recurrently encode a lookback window → forecast)
and is the right complexity to get end-to-end correct — causal sequence
windowing, a validation split used only for early stopping (never touching
test data), and feature/target standardization fit on **train data only**
(fitting a scaler on the full dataset including test data is a very common,
very subtle form of look-ahead leakage — see `LSTMModel.fit()`). Once this
is solid, reading the TFT/N-BEATS/DeepAR papers is a much easier next step,
because you already understand the problem they're solving.

**7. Walk-forward validation with a purge gap.** A random/shuffled
train-test split lets a model train on data from after the point it's being
tested on — completely unrealistic, and the single most common mistake in
applied financial ML. `src/walk_forward.py` implements an **expanding
window, chronological, purged** split instead: train windows grow forward
in time, test windows always come strictly after, and a `purge` gap of
`horizon` rows is dropped between train and test because a label at row `t`
already depends on the price at `t+horizon` — without the purge, some
training labels would overlap with the test period.

**8. No look-ahead bias.** Not a separate step in the code — it's a
property enforced throughout `features.py`, `walk_forward.py`, and
`dl_model.py` (see above). It's called out explicitly here because it's the
thing most tutorials get wrong.

**9. Transaction costs & slippage.** `src/backtest.py` charges
`turnover × (commission_bps + slippage_bps)` every time a position changes,
which directly penalizes a model that flips its predicted ranking often.
This turned out to be the single biggest driver of strategy performance in
this project (see findings.md, failure case #4) — a much more realistic
and more interesting result than a "paper P&L" that ignores costs entirely.

**10-11. Out-of-sample testing & dual evaluation.** All reported numbers
are computed only on each fold's held-out test window, never on training
data. Predictive quality (`evaluate.py`: RMSE, MAE, directional accuracy,
rank IC) and strategy quality (`backtest.py`: Sharpe, Sortino, CAGR, max
drawdown, turnover, hit rate) are reported **separately and side by side**,
because they can and did disagree here — a model can have reasonable RMSE
and still lose money once turnover and costs are accounted for.

**12. Documented failure cases.** [`reports/findings.md`](reports/findings.md)
walks through four specific, quantified failure modes from the actual run
(RMSE being misleading, ML/DL not beating linear, fold-to-fold instability,
and costs destroying gross edge) plus a concrete mitigation (lower
rebalance frequency) and a list of next steps.

## Why a negative/muted result is the point, not a problem

It would have been easy to keep tweaking the synthetic data or the model
until some backtest curve went up and to the right, take a screenshot, and
call it a portfolio project. That's precisely the failure mode this
project's own methodology (walk-forward validation, purge gaps, held-out
test folds, cost accounting) exists to catch. The real skill being
demonstrated — and the thing worth talking about in an interview — is:
*I know how to test a financial forecasting idea rigorously enough to trust
a "no" as much as a "yes."* Most trading signals genuinely don't survive
costs; being able to prove that quantitatively, on a rigorous pipeline, is
the more valuable and more honest signal to send a quant-adjacent
employer than an unrealistically clean equity curve would be.

## Honest limitations

- Universe is small (7 assets) — a cross-sectional top/bottom-2 rank
  strategy on 7 names is inherently noisy; this is called out as a next
  step in findings.md.
- No point-in-time fundamental/ESG-score data is used (price/volume only,
  as scoped) — a natural extension would be adding point-in-time ESG scores
  or sector/fundamental data, with the same no-look-ahead discipline
  applied to *those* features (e.g. ESG scores are often restated/revised —
  a real point-in-time feed would need timestamp-of-availability, not
  timestamp-of-effective-date).
- Costs are modelled as flat bps on turnover, not a market-impact model
  that scales with trade size vs. average daily volume — reasonable for a
  small book, would need revisiting at larger size.
- The demo run here uses synthetic data because of this environment's
  network restrictions; conclusions about *methodology* (what the pipeline
  correctly measures and why) hold regardless of data source, but
  conclusions about *this specific universe's tradeability* only hold once
  you rerun on real data.
