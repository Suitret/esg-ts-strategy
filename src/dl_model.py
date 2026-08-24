"""
dl_model.py
===========
Step 6: one deep-learning time-series model.

We use a small LSTM. The application mentions TFT / N-BEATS / DeepAR / GNNs
etc. -- those are absolutely worth reading about, but building one from
scratch as your first serious DL time-series project is a bad trade-off:
they're complex enough that debugging them teaches you more about the
library than about time-series methodology. An LSTM is the right complexity
level to get right end-to-end (causal windowing, proper train/val/test,
no leakage) and it uses the SAME core idea as TFT/DeepAR (recurrent encoding
of a lookback window -> forecast). Once this is solid, reading the TFT/
N-BEATS papers is much easier because you already know what problem they're
solving.

KEY DIFFERENCE FROM THE ML MODEL: the LSTM consumes a SEQUENCE of the last
`lookback` days of features per asset, not a single flat row. Sequence
construction must remain causal: the sequence ending at day t only contains
rows up to and including day t, and the label is still the forward return
computed in features.py (shift(-horizon), already causal).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SeqDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(feat_df, feature_cols, target_col, lookback=20):
    """
    Build (n_samples, lookback, n_features) tensors PER TICKER, then
    concatenate. A sequence ending at date t uses feature rows
    [t-lookback+1, ..., t] (all <= t, causal) and is labelled with
    target_col at row t (which is already the FORWARD return, i.e. it
    depends on prices AFTER t -- exactly the same causal contract as the
    tabular models).
    """
    X_list, y_list, meta = [], [], []
    for ticker, df in feat_df.groupby(level="ticker"):
        d = df.droplevel("ticker").sort_index()
        vals = d[feature_cols].values
        target = d[target_col].values
        dates = d.index
        for i in range(lookback - 1, len(d)):
            if np.isnan(target[i]) or np.isnan(vals[i - lookback + 1:i + 1]).any():
                continue
            X_list.append(vals[i - lookback + 1:i + 1])
            y_list.append(target[i])
            meta.append((dates[i], ticker))
    if not X_list:
        return np.empty((0, lookback, len(feature_cols))), np.empty((0,)), []
    return np.stack(X_list), np.array(y_list), meta


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last = h_n[-1]              # final hidden state = summary of the whole lookback window
        return self.head(last).squeeze(-1)


class LSTMModel:
    """Wraps training/inference; standardises features using TRAIN-set stats only
    (fit on train, applied to test) -- fitting the scaler on the full dataset
    including test data is itself a (very common) form of look-ahead leakage."""

    def __init__(self, n_features, lookback=20, hidden_size=32, epochs=25, lr=1e-3,
                 batch_size=256, device=None, seed=0):
        torch.manual_seed(seed)
        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMForecaster(n_features, hidden_size).to(self.device)
        self.mu, self.sigma = None, None

    def _scale(self, X):
        return (X - self.mu) / self.sigma

    def fit(self, X_seq_train, y_train, X_seq_val=None, y_val=None, verbose=False):
        # scaler fit on TRAIN ONLY
        self.mu = X_seq_train.reshape(-1, X_seq_train.shape[-1]).mean(axis=0)
        self.sigma = X_seq_train.reshape(-1, X_seq_train.shape[-1]).std(axis=0) + 1e-8
        Xtr = self._scale(X_seq_train)
        # target scaling: standardize target too (helps LSTM training stability)
        self.y_mu, self.y_sigma = y_train.mean(), y_train.std() + 1e-8
        ytr = (y_train - self.y_mu) / self.y_sigma

        train_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        best_val, best_state, patience, bad_epochs = np.inf, None, 5, 0
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item() * len(yb)
            train_loss = total_loss / len(train_loader.dataset)

            val_loss = train_loss
            if X_seq_val is not None and len(y_val) > 0:
                val_loss = self._eval_loss(X_seq_val, y_val)
                if val_loss < best_val - 1e-6:
                    best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
                else:
                    bad_epochs += 1
            if verbose:
                print(f"  epoch {epoch+1}/{self.epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
            if X_seq_val is not None and bad_epochs >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _eval_loss(self, X_seq, y):
        self.model.eval()
        with torch.no_grad():
            Xs = self._scale(X_seq)
            ys = (y - self.y_mu) / self.y_sigma
            xb = torch.tensor(Xs, dtype=torch.float32).to(self.device)
            pred = self.model(xb).cpu().numpy()
            return float(np.mean((pred - ys) ** 2))

    def predict(self, X_seq):
        self.model.eval()
        with torch.no_grad():
            Xs = self._scale(X_seq)
            xb = torch.tensor(Xs, dtype=torch.float32).to(self.device)
            pred = self.model(xb).cpu().numpy()
        return pred * self.y_sigma + self.y_mu   # back to real return units
