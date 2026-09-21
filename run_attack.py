"""Closed-loop robustness under attack (10 seeds): controllers with and without detector-based screening."""
import sys, json, time, os, pickle, numpy as np, torch
torch.set_num_threads(1)
from evsim.config import SimConfig
from evsim.agents import TorchPolicy, Static, TOU, RuleBased
from evsim.forecast import load_forecaster
from evsim.secure import run_closed_loop
cfg = SimConfig()
METRICS = ["reward", "profit", "wait_min", "utilization", "over_events", "over_kwh", "served_frac", "energy_kwh", "peak_ratio"]
SCEN = {"none": (0, 0.4, False), "FDI": (1, 0.4, False), "FDI-mask": (1, 1.0, True), "MITM": (2, 0.4, False), "DDoS": (3, 0.4, False)}


def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)


def load_pol(kind, seed):
    d = torch.load(f"results/models/{kind}_s{seed}.pt", weights_only=False)
    return TorchPolicy(d["kind"], d["net"].eval(), cfg)


def run(seed):
    fc = load_forecaster(seed, "results/models", cfg)
    det = pickle.load(open(f"results/models/det_s{seed}.pkl", "rb")); sc = det["scaler"]
    D = {d.name: d for d in det["dets"]}
    main = json.load(open(f"results/control/main_s{seed}.json"))
    pars = main["_params"]
    ctrl = {"Static": Static(cfg), "Best fixed price": Static(cfg, pars["fixed_m"]), "TOU": TOU(cfg, tuple(pars["tou"])), "Rule-based": RuleBased(cfg, pars["rule"]),
            "DQN": load_pol("dqn", seed), "SAC": load_pol("sac", seed), "TD3": load_pol("td3", seed), "PPO": load_pol("ppo", seed)}
    rows = []                                   # (controller, screening) combos
    for c in ctrl: rows.append((c, None))
    for dn in D: rows.append(("PPO", dn))
    rows.append(("Rule-based", "Autoencoder (ARTO-EV)"))
    res = {}
    es = 60000 + seed
    for sname, (at, fr, mk) in SCEN.items():
        for cname, dn in rows:
            S, dd = run_closed_loop(cfg, ctrl[cname], es, sc, D[dn] if dn else None, atype=at, frac=fr, E=30, forecaster=fc,
                                    forecast_mode="lstm", mask=mk)
            res[f"{sname}|{cname}|{dn}"] = dict(m={k: [float(x) for x in S[k]] for k in METRICS}, det=dd)
        log(f"seed {seed} scenario {sname} done")
    json.dump(res, open(f"results/control/attack_s{seed}.json", "w"))


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        run(s)
