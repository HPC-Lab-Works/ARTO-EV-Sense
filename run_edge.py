"""Edge inference cost: parameters, model size, single-sample latency (CPU, 1 thread), int8 dynamic quantisation."""
import sys, json, time, os, pickle, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
from evsim.security import *
from evsim.forecast import LSTMForecaster, W as FW, NF, H as FH
cfg = SimConfig()


def n_params(m): return sum(p.numel() for p in m.parameters())


def lat(fn, reps=1500, warm=100):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def quant(m):
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(m), {nn.Linear, nn.LSTM}, dtype=torch.qint8)


import copy


def size_kb(m):
    import io
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.getbuffer().nbytes / 1024


def run(seed):
    d = pickle.load(open(f"results/models/det_s{seed}.pkl", "rb"))
    ae = [x for x in d["dets"] if x.name.startswith("Autoencoder")][0].model.eval()
    lae = [x for x in d["dets"] if x.name.startswith("LSTM")][0].model.eval()
    fc = LSTMForecaster(); fc.load_state_dict(torch.load(f"results/models/lstm_s{seed}.pt")); fc.eval()
    ppo = torch.load(f"results/models/ppo_s{seed}.pt", weights_only=False)["net"].eval()
    out = {}
    x_ae = torch.randn(1, WIN * NCH); x_seq = torch.randn(1, WIN, NCH)
    x_fc = (torch.randn(1, FW, NF), torch.randn(1, FH), torch.randn(1))
    x_pi = torch.randn(1, 16)
    x_net = torch.randn(cfg.N, WIN * NCH)            # one slot of all stations at a gateway
    cases = {"Autoencoder (edge, per station)": (ae, lambda m: (lambda: m(x_ae))),
             "Autoencoder (10 stations batch)": (ae, lambda m: (lambda: m(x_net))),
             "LSTM autoencoder": (lae, lambda m: (lambda: m(x_seq))),
             "LSTM forecaster": (fc, lambda m: (lambda: m(*x_fc))),
             "PPO actor": (ppo, lambda m: (lambda: m(x_pi)))}
    with torch.no_grad():
        for name, (m, mk) in cases.items():
            q = quant(m)
            p50, p95 = lat(mk(m)); q50, q95 = lat(mk(q))
            out[name] = dict(params=n_params(m), size_kb=size_kb(m), lat_p50_ms=p50, lat_p95_ms=p95, q_size_kb=size_kb(q), q_lat_p50_ms=q50, q_lat_p95_ms=q95)
        # detection with quantised AE
        S = build_sets(cfg, seed)
        te = S["te"].reshape(-1, WIN, NCH); lab = S["lab"].reshape(-1); va = S["va"].reshape(-1, WIN, NCH)
        def f1_of(model):
            def sc(W):
                X = torch.as_tensor(W.reshape(len(W), -1)); return ((model(X) - X) ** 2).mean(1).numpy()
            thr = np.quantile(sc(va), 0.99); s = sc(te); return det_metrics(s > thr, s, lab)
        out["ae_f1_fp32"] = f1_of(ae)["f1"]; out["ae_f1_int8"] = f1_of(quant(ae))["f1"]
        # end-to-end slot pipeline latency at a gateway: window scoring of 10 stations + forecast + policy
        pipe = lambda: (ae(x_net), fc(*x_fc), ppo(x_pi))
        out["pipeline_p50_ms"], out["pipeline_p95_ms"] = lat(pipe)
    return out


if __name__ == "__main__":
    res = []
    for s in map(int, sys.argv[1:]):
        res.append(run(s)); print(time.strftime("%H:%M:%S"), "edge seed", s, flush=True)
    json.dump(res, open("results/edge.json", "w"))
