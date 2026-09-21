"""Federated anomaly detection: local-only vs FedAvg vs centralised autoencoders on non-IID stations (10 seeds)."""
import sys, json, time, os, copy, numpy as np, torch
torch.set_num_threads(1)
from evsim.security import *
cfg = SimConfig()
ROUNDS, LOC_EP = 20, 2


def train_local(model, X, seed, epochs, lr=2e-3, bs=256):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.as_tensor(X); n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            x = X[perm[i:i + bs]]
            loss = ((model(x) - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return model


def score(model, W):
    X = torch.as_tensor(W.reshape(len(W), -1))
    with torch.no_grad():
        return ((model(X) - X) ** 2).mean(1).numpy()


def evaluate_models(models, S):
    """models: list of per-station models (may be the same object). Per-station threshold at 99 % of own benign val scores."""
    al, sc, lb = [], [], []
    for n in range(cfg.N):
        va = S["va"][:, n].reshape(-1, WIN * NCH); te = S["te"][:, n].reshape(-1, WIN * NCH)
        thr = np.quantile(score(models[n], va), 0.99)
        s = score(models[n], te)
        al.append(s > thr); sc.append((s - thr) / max(thr, 1e-9)); lb.append(S["lab"][:, n].reshape(-1))
    return det_metrics(np.concatenate(al), np.concatenate(sc), np.concatenate(lb))


def run(seed):
    S = build_sets(cfg, seed)
    N = cfg.N
    client = [S["tr"][:, n].reshape(-1, WIN * NCH) for n in range(N)]
    out = {}
    # local-only
    t0 = time.time()
    loc = [train_local(MLPAE(), client[n], seed * 100 + n, 25) for n in range(N)]
    out["Local only"] = evaluate_models([m.eval() for m in loc], S)
    out["Local only"]["comm_bytes"] = 0
    # centralised
    cen = train_local(MLPAE(), np.concatenate(client), seed, 25).eval()
    out["Centralised"] = evaluate_models([cen] * N, S)
    raw_bytes = sum(len(c) for c in client) * NCH * 4          # raw slots (one slot per window) uploaded once
    out["Centralised"]["comm_bytes"] = raw_bytes
    # FedAvg
    torch.manual_seed(seed)
    g = MLPAE()
    n_par = sum(p.numel() for p in g.parameters())
    for r in range(ROUNDS):
        states, wts = [], []
        for n in range(N):
            m = copy.deepcopy(g); train_local(m, client[n], seed * 1000 + r * 10 + n, LOC_EP)
            states.append(m.state_dict()); wts.append(len(client[n]))
        w = np.array(wts) / sum(wts)
        g.load_state_dict({k: sum(float(w[i]) * states[i][k] for i in range(N)) for k in states[0]})
    g.eval()
    out["FedAvg"] = evaluate_models([g] * N, S)
    out["FedAvg"]["comm_bytes"] = ROUNDS * N * 2 * n_par * 4
    out["n_params"] = n_par
    out["time_s"] = time.time() - t0
    print(time.strftime("%H:%M:%S"), seed, {k: round(v["f1"], 3) for k, v in out.items() if isinstance(v, dict)}, flush=True)
    json.dump(out, open(f"results/security/fed_s{seed}.json", "w"))


if __name__ == "__main__":
    os.makedirs("results/security", exist_ok=True)
    for s in map(int, sys.argv[1:]):
        run(s)
