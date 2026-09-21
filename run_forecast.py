"""Forecasting testbed: 10 seeds, LSTM vs baselines on held-out synthetic series."""
import numpy as np, torch, json, time, os, sys
torch.set_num_threads(2)
from evsim.config import SimConfig
from evsim import forecast as F
cfg = SimConfig()
R = cfg.R_per_station * cfg.N
os.makedirs("results/models", exist_ok=True)
res = {"lstm": [], "mlp": [], "persistence": [], "seasonal_naive": [], "ridge": []}
tr_all = None
for seed in range(10):
    t0 = time.time()
    tr = F.build_dataset(cfg, 160, 4, 1000 + seed)
    va = F.build_dataset(cfg, 40, 4, 2000 + seed)
    te = F.build_dataset(cfg, 60, 4, 3000 + seed)       # held-out
    y = te["y"].numpy()
    res["persistence"].append(F.metrics(np.repeat(te["now"].numpy()[:, None], F.H, 1), y, R))
    res["seasonal_naive"].append(F.metrics(te["yest"].numpy(), y, R))
    res["ridge"].append(F.metrics(F.ridge_fit(tr)(te), y, R))
    for name, mk in [("lstm", F.LSTMForecaster), ("mlp", F.MLPForecaster)]:
        torch.manual_seed(seed)
        m = F.train_nn(mk(), tr, va, seed)
        with torch.no_grad():
            p = m(te["X"], te["yest"], te["now"]).numpy()
        res[name].append(F.metrics(p, y, R))
        if name == "lstm":
            torch.save(m.state_dict(), f"results/models/lstm_s{seed}.pt")
    print(seed, round(time.time() - t0, 1), {k: round(v[-1]["mae_all"], 2) for k, v in res.items()}, flush=True)
json.dump(res, open("results/forecast.json", "w"))
