"""Grid-headroom forecasting: LSTM (ARTO-EV) versus persistence, seasonal-naive, ridge AR and MLP baselines."""
import numpy as np, torch, torch.nn as nn, os, time
from .config import SimConfig, STEPS_PER_DAY
from .gen import gen_exo

SPD = STEPS_PER_DAY
W = 32           # look-back window (8 h)
H = 4            # horizon (1 h)
NF = 6


def features(exo, cfg):
    R = exo["R"]
    pvp = cfg.pv_per_station * cfg.N
    hr = exo["hour"]
    f = np.stack([exo["head"] / R, np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
                  exo["temp"] / 30.0, exo["pv"] / pvp, exo["weekend"].astype(float)], -1)    # [E,T,6]
    return f


def make_windows(exo, cfg, t_idx=None):
    """Return X [E,T',W,NF], Y_aux (yesterday same-slot headroom for the next H, [E,T',H]), head_now [E,T'],
    target [E,T',H] (or None) for every valid t. t index list returned."""
    f = features(exo, cfg)
    E, T, _ = f.shape
    R = exo["R"]
    head = exo["head"] / R
    pad = np.concatenate([np.repeat(f[:, :1], W - 1, 1), f], 1)
    ts = np.arange(T) if t_idx is None else t_idx
    X = np.stack([pad[:, t:t + W] for t in ts], 1)                       # [E,T',W,NF]
    yidx = np.clip(ts[:, None] + np.arange(1, H + 1)[None, :] - SPD, 0, T - 1)
    yest = head[:, yidx]                                                  # [E,T',H]
    tidx = np.clip(ts[:, None] + np.arange(1, H + 1)[None, :], 0, T - 1)
    tgt = head[:, tidx]
    now = head[:, ts]
    return X, yest, now, tgt


class LSTMForecaster(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(NF, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden + H + 1, 64), nn.ReLU(), nn.Linear(64, H))

    def forward(self, x, yest, now):
        out, _ = self.lstm(x)
        h = out[:, -1]
        z = torch.cat([h, yest - now[:, None], now[:, None]], 1)
        return now[:, None] + self.fc(z)      # predicts headroom / R at t+1..t+H


class MLPForecaster(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(W * NF + H + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, H))

    def forward(self, x, yest, now):
        z = torch.cat([x.flatten(1), yest - now[:, None], now[:, None]], 1)
        return now[:, None] + self.net(z)


def build_dataset(cfg, n_series, days, seed):
    rng = np.random.default_rng(seed)
    exo = gen_exo(rng, n_series, cfg, days=days)
    # windows only for t in [SPD, T-H) so that yesterday's values exist
    T = days * SPD
    ts = np.arange(SPD, T - H)
    X, yest, now, tgt = make_windows(exo, cfg, ts)
    fl = lambda a: a.reshape(-1, *a.shape[2:])
    return dict(X=torch.tensor(fl(X), dtype=torch.float32), yest=torch.tensor(fl(yest), dtype=torch.float32),
                now=torch.tensor(fl(now), dtype=torch.float32), y=torch.tensor(fl(tgt), dtype=torch.float32),
                hour=np.tile(exo["hour"][:, ts], 1).reshape(-1), exo=exo)


def train_nn(model, tr, va, seed, epochs=14, lr=2e-3, bs=512):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(tr["y"])
    best, best_sd = 1e9, None
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            p = model(tr["X"][j], tr["yest"][j], tr["now"][j])
            loss = nn.functional.smooth_l1_loss(p, tr["y"][j], beta=0.02)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.l1_loss(model(va["X"], va["yest"], va["now"]), va["y"]).item()
        if v < best:
            best, best_sd = v, {k: x.clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_sd)
    model.eval()
    return model


def metrics(pred, y, R):
    """pred,y [n,H] in units of R. Returns MAE/RMSE in kW and MAPE %, for horizon 1 and 4 and all-horizon mean."""
    err = (pred - y)
    out = {}
    for name, sl in [("h1", slice(0, 1)), ("h4", slice(3, 4)), ("all", slice(0, H))]:
        e = err[:, sl]
        out[f"mae_{name}"] = float(np.abs(e).mean() * R)
        out[f"rmse_{name}"] = float(np.sqrt((e ** 2).mean()) * R)
        out[f"mape_{name}"] = float((np.abs(e) / np.abs(y[:, sl])).mean() * 100)
    return out


def ridge_fit(tr, alpha=1.0):
    def feat(d):
        return np.concatenate([d["X"].reshape(len(d["X"]), -1).numpy(), (d["yest"] - d["now"][:, None]).numpy(),
                               d["now"][:, None].numpy(), np.ones((len(d["now"]), 1))], 1)
    A = feat(tr)
    Y = (tr["y"] - tr["now"][:, None]).numpy()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ Y)
    return lambda d: d["now"][:, None].numpy() + feat(d) @ w


class EnvForecast:
    """Callable used by EVNetEnv: exo -> forecasts [E,T,H] of headroom in kW, made at each slot using data up to that slot."""
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg

    def __call__(self, exo):
        E, T = exo["head"].shape
        R = exo["R"]
        ts = np.arange(T)
        X, yest, now, _ = make_windows(exo, self.cfg, ts)
        X = torch.tensor(X.reshape(-1, W, NF), dtype=torch.float32)
        yest = torch.tensor(yest.reshape(-1, H), dtype=torch.float32)
        now = torch.tensor(now.reshape(-1), dtype=torch.float32)
        with torch.no_grad():
            p = self.model(X, yest, now).numpy().reshape(E, T, H)
        return p * R


def load_forecaster(seed, path, cfg):
    m = LSTMForecaster()
    m.load_state_dict(torch.load(f"{path}/lstm_s{seed}.pt"))
    m.eval()
    return EnvForecast(m, cfg)
