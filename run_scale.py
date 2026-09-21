"""Scalability (zero-shot transfer to larger networks, N up to 500) and robustness to distribution shift (10 seeds)."""
import sys, json, time, os, resource, numpy as np, torch
torch.set_num_threads(1)
from evsim.config import SimConfig
from evsim.agents import TorchPolicy, Static, TOU, RuleBased, evaluate
from evsim.forecast import load_forecaster
from evsim.env import EVNetEnv
METRICS = ["reward", "profit", "wait_min", "utilization", "over_events", "over_kwh", "served_frac", "peak_ratio"]


def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)


def load_pol(kind, seed, cfg):
    d = torch.load(f"results/models/{kind}_s{seed}.pt", weights_only=False)
    return TorchPolicy(d["kind"], d["net"].eval(), cfg)


def run(seed):
    base = SimConfig()
    main = json.load(open(f"results/control/main_s{seed}.json")); pars = main["_params"]
    out = {"scale": {}, "shift": {}}
    for N in (10, 50, 100, 500):
        cfg = SimConfig(N=N)
        fc = load_forecaster(seed, "results/models", cfg)
        ctrl = {"Static": Static(cfg), "Best fixed price": Static(cfg, pars["fixed_m"]), "TOU": TOU(cfg, tuple(pars["tou"])), "Rule-based": RuleBased(cfg, pars["rule"])}
        for k in ("dqn", "sac", "td3", "ppo"):
            ctrl[k.upper()] = load_pol(k, seed, cfg)
        E = 10 if N < 500 else 4
        r = {}
        for name, c in ctrl.items():
            t0 = time.perf_counter()
            S = evaluate(c, cfg, 70000 + seed, fc, "lstm", E=E)
            r[name] = {k: [float(x) for x in S[k]] for k in METRICS}
            r[name]["wall_s_per_day"] = (time.perf_counter() - t0) / E
        # pure environment step cost (static policy, no forecaster overhead in the loop) and policy inference cost
        env = EVNetEnv(cfg, E=1, forecast_mode="persist"); o = env.reset(1)
        t0 = time.perf_counter()
        for t in range(96): o, _, _ = env.step(np.ones(1), np.full(1, cfg.C))
        r["env_step_ms"] = (time.perf_counter() - t0) / 96 * 1e3
        r["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        out["scale"][str(N)] = r
        log(f"seed {seed} N={N}: PPO reward/day {np.mean(r['PPO']['reward']):.1f} rule {np.mean(r['Rule-based']['reward']):.1f}")
    cfg = SimConfig(); fc = load_forecaster(seed, "results/models", cfg)
    ctrl = {"Static": Static(cfg), "Best fixed price": Static(cfg, pars["fixed_m"]), "TOU": TOU(cfg, tuple(pars["tou"])), "Rule-based": RuleBased(cfg, pars["rule"])}
    for k in ("dqn", "sac", "td3", "ppo"):
        ctrl[k.upper()] = load_pol(k, seed, cfg)
    shifts = {"nominal": {}, "demand +30%": dict(demand_mult=1.3), "heat wave (+8 C)": dict(heat=8.0), "feeder load +10%": dict(load_scale=1.10),
              "price sensitivity x1.7": dict(wmult=0.6)}
    for sn, kw in shifts.items():
        r = {}
        for name, c in ctrl.items():
            S = evaluate(c, cfg, 80000 + seed, fc, "lstm", E=30, **kw)
            r[name] = {k: [float(x) for x in S[k]] for k in METRICS}
        out["shift"][sn] = r
    log(f"seed {seed} shifts done")
    json.dump(out, open(f"results/control/scale_s{seed}.json", "w"))


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        run(s)
