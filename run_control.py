"""Control experiments. Usage: python run_control.py main SEED | ablate SEED | tune SEED_TUNE
Trains every controller with the same environment-step budget and evaluates on held-out days (common random numbers)."""
import sys, json, time, os, numpy as np, torch
torch.set_num_threads(1)
from evsim.config import SimConfig
from evsim.agents import *
from evsim.forecast import load_forecaster

cfg = SimConfig()
STEPS = int(os.environ.get("STEPS", 300_000))
METRICS = ["reward", "profit", "wait_min", "utilization", "over_events", "over_kwh", "served_frac", "renege_frac", "energy_kwh", "peak_ratio"]
os.makedirs("results/control", exist_ok=True); os.makedirs("results/models", exist_ok=True)
LR = json.load(open("results/tuned_lr.json")) if os.path.exists("results/tuned_lr.json") else {"ppo": 3e-4, "dqn": 3e-4, "sac": 3e-4, "td3": 3e-4}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def pack(S):
    return {k: [float(x) for x in S[k]] for k in METRICS}


def tune_static_ctrl(fc, seed):
    """small equal-effort tuning of the hand-written controllers on validation days (seed 90000+seed)."""
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
    return best_tou, best_rule, bt, br, best_m


def save_pol(pol, path):
    torch.save(dict(kind=pol.kind, net=pol.net, fixed_m=pol.fixed_m, fixed_n=pol.fixed_n), path)


if __name__ == "__main__":
    mode, seed = sys.argv[1], int(sys.argv[2])
    if mode == "tune":
        fc = load_forecaster(0, "results/models", cfg)
        out = {}
        for kind in ["ppo", "dqn", "sac", "td3"]:
            for lr in (3e-4, 1e-3):
                rs = []
                for s in (100 + seed,):
                    pol, _ = train_agent(kind, s, cfg, fc, "lstm", steps=100_000, lr=lr)
                    rs.append(float(evaluate(pol, cfg, 90000 + s, fc, "lstm", E=30)["reward"].mean()))
                log(f"tune {kind} lr {lr} seed {seed}: {np.mean(rs):.2f}")
                out[f"{kind}_{lr}"] = float(np.mean(rs))
        json.dump(out, open(f"results/tune_{seed}.json", "w"))
    elif mode == "main":
        fc = load_forecaster(seed, "results/models", cfg)
        es = 50000 + seed
        res = {}
        t0 = time.time()
        tou_p, rule_p, bt, br, fixm = tune_static_ctrl(fc, seed)
        res["Static"] = pack(evaluate(Static(cfg), cfg, es, fc, "lstm", E=30))
        res["Best fixed price"] = pack(evaluate(Static(cfg, fixm), cfg, es, fc, "lstm", E=30))
        res["TOU"] = pack(evaluate(TOU(cfg, tou_p), cfg, es, fc, "lstm", E=30))
        res["Rule-based"] = pack(evaluate(RuleBased(cfg, rule_p), cfg, es, fc, "lstm", E=30))
        res["_params"] = dict(fixed_m=fixm, tou=list(tou_p), rule={k: float(v) for k, v in rule_p.items()})
        log(f"seed {seed} heuristics tuned ({time.time()-t0:.0f}s): TOU val {bt:.2f} rule val {br:.2f}")
        hist = {}
        for kind, name in [("ppo", "PPO (ARTO-EV)"), ("dqn", "DQN"), ("sac", "SAC"), ("td3", "TD3")]:
            t1 = time.time()
            pol, h = train_agent(kind, seed, cfg, fc, "lstm", steps=STEPS, lr=LR[kind], log=lambda m: None)
            res[name] = pack(evaluate(pol, cfg, es, fc, "lstm", E=30))
            res[name + "_train_s"] = time.time() - t1
            hist[name] = h
            save_pol(pol, f"results/models/{kind}_s{seed}.pt")
            log(f"seed {seed} {name}: reward {np.mean(res[name]['reward']):.2f} wait {np.mean(res[name]['wait_min']):.1f} over {np.mean(res[name]['over_events']):.2f} ({time.time()-t1:.0f}s)")
        res["_hist"] = hist
        json.dump(res, open(f"results/control/main_s{seed}.json", "w"))
    elif mode == "ablate":
        fc = load_forecaster(seed, "results/models", cfg)
        es = 50000 + seed
        res = {}
        variants = [("PPO, persistence forecast", "persist", None, None), ("PPO, oracle forecast", "oracle", None, None),
                    ("PPO, no forecast", "none", None, None), ("PPO, price fixed (m=1)", "lstm", 1.0, None),
                    ("PPO, availability fixed (all chargers)", "lstm", None, cfg.C)]
        for name, fm, fmx, fnx in variants:
            t1 = time.time()
            pol, _ = train_agent("ppo", seed, cfg, fc, fm, steps=STEPS, lr=LR["ppo"], fixed_m=fmx, fixed_n=fnx)
            res[name] = pack(evaluate(pol, cfg, es, fc, fm, E=30))
            log(f"seed {seed} {name}: reward {np.mean(res[name]['reward']):.2f} ({time.time()-t1:.0f}s)")
        json.dump(res, open(f"results/control/ablate_s{seed}.json", "w"))
