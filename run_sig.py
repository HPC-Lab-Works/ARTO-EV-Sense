"""Closed-loop evaluation of command authentication (real Ed25519 signatures, nonces and policy bounds) under attack.
Usage: python run_sig.py SEED [SEED ...]  -> results/control/sig_s{seed}.json"""
import sys, json, time, os, pickle, numpy as np, torch
torch.set_num_threads(1)
from evsim.config import SimConfig
from evsim.agents import TorchPolicy, RuleBased
from evsim.forecast import load_forecaster
from evsim.secure import run_closed_loop, CmdChannel
cfg = SimConfig()
METRICS = ["reward", "profit", "wait_min", "utilization", "over_events", "over_kwh", "served_frac"]
# scenario -> (attack type, fraction of stations, channel variant)
SCEN = {"none": (0, 0.4, "forge"), "MITM, forged command": (2, 0.4, "forge"), "MITM, stolen gateway key": (2, 0.4, "stolen"),
        "Replay of an off-peak command": (2, 0.4, "replay"), "DDoS": (3, 0.4, "forge")}
# defense -> (channel mode, fallback, screening detector)
DEF = {"No defense": ("none", "hold", None), "AE screening": ("none", "hold", "Autoencoder (ARTO-EV)"),
       "Signature only": ("sig", "hold", None), "Signature, nonce, policy": ("full", "hold", None),
       "Signature, nonce, policy, safe fallback": ("full", "safe", None),
       "Full (contract rules and AE screening)": ("full", "hold", "Autoencoder (ARTO-EV)")}


def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)


def load_pol(kind, seed):
    d = torch.load(f"results/models/{kind}_s{seed}.pt", weights_only=False)
    return TorchPolicy(d["kind"], d["net"].eval(), cfg)


def run(seed, E=30):
    fc = load_forecaster(seed, "results/models", cfg)
    det = pickle.load(open(f"results/models/det_s{seed}.pkl", "rb")); sc = det["scaler"]
    D = {d.name: d for d in det["dets"]}
    pars = json.load(open(f"results/control/main_s{seed}.json"))["_params"]
    ctrl = {"PPO": load_pol("ppo", seed), "Rule-based": RuleBased(cfg, pars["rule"])}
    only = os.environ.get("CTRL")                       # optional restriction, results are merged into an existing file
    if only:
        ctrl = {k: v for k, v in ctrl.items() if k in only.split(",")}
    path = f"results/control/sig_s{seed}.json"
    res = json.load(open(path)) if (only and os.path.exists(path)) else {}
    es = 60000 + seed
    for sname, (at, fr, var) in SCEN.items():
        for cname, pol in ctrl.items():
            for dname, (mode, fb, dn) in DEF.items():
                chan = CmdChannel(E, cfg.N, cfg.C, mode=mode, fallback=fb, variant=var)
                S, dd = run_closed_loop(cfg, pol, es, sc, D[dn] if dn else None, atype=at, frac=fr, E=E, forecaster=fc,
                                        forecast_mode="lstm", chan=chan)
                res[f"{sname}|{cname}|{dname}"] = dict(m={k: [float(x) for x in S[k]] for k in METRICS}, det=dd)
        log(f"seed {seed} scenario {sname} done")
    json.dump(res, open(path, "w"))


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        run(s)
