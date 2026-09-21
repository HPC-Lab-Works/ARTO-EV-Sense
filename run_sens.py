"""Sensitivity of the control comparison to the simulator assumptions (retrained, one factor at a time).
Usage: python run_sens.py SEED [SEED ...]   -> results/sens/sens_s{seed}.json
For every configuration and seed: heuristics are re-tuned on validation days, PPO and TD3 are retrained with the
same budget and hyperparameters as in the main study, and all controllers are evaluated on the held-out days of the seed
(the same exogenous days as in the main study; only the changed parameter differs). The headroom forecaster does not
depend on the changed parameters (headroom is exogenous), so the forecaster of the seed is reused."""
import sys, json, time, os, numpy as np, torch
torch.set_num_threads(1)
from dataclasses import replace
from evsim.config import SimConfig
from evsim.agents import Static, TOU, RuleBased, evaluate, train_agent, rule_param_sample
from evsim.forecast import load_forecaster

STEPS = int(os.environ.get("STEPS", 300_000))
METRICS = ["reward", "profit", "wait_min", "utilization", "over_events", "over_kwh", "served_frac"]
LR = json.load(open("results/tuned_lr.json"))
BASE = SimConfig()
CONFIGS = {
    "demand x0.75": dict(demand_scale=BASE.demand_scale * 0.75),
    "demand x1.3": dict(demand_scale=BASE.demand_scale * 1.3),
    "steeper price response": dict(w_price=tuple(w * 0.6 for w in BASE.w_price)),
    "flatter price response": dict(w_price=tuple(w * 1.6 for w in BASE.w_price)),
    "wait weight 3": dict(w_wait=3.0),
    "wait weight 12": dict(w_wait=12.0),
    "overload weight 1": dict(w_over=1.0),
    "overload weight 5": dict(w_over=5.0),
}
os.makedirs("results/sens", exist_ok=True)


def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)


def pack(S): return {k: [float(x) for x in S[k]] for k in METRICS}


def tune(cfg, fc, seed):
    vs = 90000 + seed
    best_tou, bt = None, -1e9
    for mo in (0.6, 0.8, 1.0):
        for mm in (1.0, 1.1, 1.2):
            for mp in (1.2, 1.4, 1.6):
                r = evaluate(TOU(cfg, (mo, mm, mp)), cfg, vs, fc, "lstm", E=30)["reward"].mean()
                if r > bt: bt, best_tou = r, (mo, mm, mp)
    best_m, bm = 1.0, -1e9
    for mm in np.arange(0.8, 1.65, 0.1):
        r = evaluate(Static(cfg, mm), cfg, vs, fc, "lstm", E=30)["reward"].mean()
        if r > bm: bm, best_m = r, float(round(mm, 2))
    rng = np.random.default_rng(seed); best_rule, br = None, -1e9
    for _ in range(80):
        p = rule_param_sample(rng)
        r = evaluate(RuleBased(cfg, p), cfg, vs, fc, "lstm", E=30)["reward"].mean()
        if r > br: br, best_rule = r, p
    return best_tou, best_rule, best_m


def run(seed):
    fc = load_forecaster(seed, "results/models", BASE)
    path = f"results/sens/sens_s{seed}.json"
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name, kw in CONFIGS.items():
        if name in out: continue
        cfg = replace(BASE, **kw)
        es = 50000 + seed
        t0 = time.time()
        tou_p, rule_p, fixm = tune(cfg, fc, seed)
        r = {"Static": pack(evaluate(Static(cfg), cfg, es, fc, "lstm", E=30)),
             "Best fixed price": pack(evaluate(Static(cfg, fixm), cfg, es, fc, "lstm", E=30)),
             "TOU": pack(evaluate(TOU(cfg, tou_p), cfg, es, fc, "lstm", E=30)),
             "Rule-based": pack(evaluate(RuleBased(cfg, rule_p), cfg, es, fc, "lstm", E=30))}
        for kind, nm in (("ppo", "PPO"), ("td3", "TD3")):
            pol, _ = train_agent(kind, seed, cfg, fc, "lstm", steps=STEPS, lr=LR[kind], log=lambda m: None)
            r[nm] = pack(evaluate(pol, cfg, es, fc, "lstm", E=30))
        out[name] = r
        json.dump(out, open(path, "w"))
        log(f"seed {seed} [{name}] " + " ".join(f"{k}={np.mean(v['reward']):.1f}" for k, v in r.items()) + f" ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        run(s)
