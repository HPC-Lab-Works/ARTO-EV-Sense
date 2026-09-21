"""Ledger experiments: throughput/latency grid, tamper tests, and in-loop overhead of ledgering one day of control."""
import sys, json, time, os, copy, numpy as np, torch
torch.set_num_threads(1)
from evsim.ledger import *
from evsim.config import SimConfig
from evsim.env import EVNetEnv
from evsim.forecast import load_forecaster
cfg = SimConfig()
os.makedirs("results/ledger", exist_ok=True)


def fresh(nv=4, bs=50, seed=0, n_st=10):
    L = Ledger(n_validators=nv, block_size=bs, rng=np.random.default_rng(seed))
    ids = [Identity(f"st{i}") for i in range(n_st)]
    for i in ids: assert L.submit(L.register(i))
    return L, ids


def cmd(L, ids, k, m=1.2, n=4):
    i = ids[k % len(ids)]
    return L.make_tx(i, "command", {"m": m, "nact": n, "slot": k})


def perf(seed):
    out = {}
    for nv in (4, 7, 10):
        for bs in (10, 50, 100):
            L, ids = fresh(nv, bs, seed)
            for k in range(bs * 40):
                assert L.submit(cmd(L, ids, k))
            while L.pool: L.commit()
            cm = np.array(L.stats["commit_ms"]); vs = np.array(L.stats["verify_s"][10:]); sg = np.array(L.stats["sign_s"][10:])
            out[f"nv{nv}_bs{bs}"] = dict(commit_ms_mean=float(cm.mean()), commit_ms_p95=float(np.percentile(cm, 95)),
                                        throughput_tps=float(bs / (cm.mean() / 1000.0)), sign_us=float(sg.mean() * 1e6),
                                        verify_us=float(vs.mean() * 1e6), bytes_per_tx=float(L.stats["bytes"] / (bs * 40)))
    return out


def tamper(seed, trials=100):
    rng = np.random.default_rng(seed)
    res = {}
    def base():
        L, ids = fresh(4, 20, int(rng.integers(1 << 30)))
        for k in range(100): L.submit(cmd(L, ids, k))
        while L.pool: L.commit()
        return L, ids
    det = {k: 0 for k in ["payload_edit", "signature_swap", "block_deleted", "blocks_reordered", "forged_block_rehashed",
                          "unregistered_sender", "replay", "out_of_policy_command"]}
    for _ in range(trials):
        L, ids = base()
        ch = copy.deepcopy(L.chain); b = int(rng.integers(1, len(ch))); t = int(rng.integers(0, len(ch[b]["txs"])))
        ch[b]["txs"][t]["payload"]["m"] = 0.6; det["payload_edit"] += not L.verify_chain(ch)[0]
        ch = copy.deepcopy(L.chain); ch[b]["txs"][0]["sig"], ch[b]["txs"][1]["sig"] = ch[b]["txs"][1]["sig"], ch[b]["txs"][0]["sig"]
        det["signature_swap"] += not L.verify_chain(ch)[0]
        ch = copy.deepcopy(L.chain); del ch[int(rng.integers(1, len(ch) - 1))]; det["block_deleted"] += not L.verify_chain(ch)[0]
        ch = copy.deepcopy(L.chain); i, j = rng.choice(np.arange(1, len(ch)), 2, replace=False); ch[i], ch[j] = ch[j], ch[i]
        det["blocks_reordered"] += not L.verify_chain(ch)[0]
        # attacker edits a tx AND recomputes Merkle root and block hash (but cannot re-sign): caught by signature check
        ch = copy.deepcopy(L.chain); ch[b]["txs"][t]["payload"]["m"] = 0.6; ch[b]["root"] = L.merkle(ch[b]["txs"])
        ch[b]["hash"] = H(canon({k: ch[b][k] for k in ("index", "prev", "root", "ts")}))
        for q in range(b + 1, len(ch)):
            ch[q]["prev"] = ch[q - 1]["hash"]; ch[q]["hash"] = H(canon({k: ch[q][k] for k in ("index", "prev", "root", "ts")}))
        det["forged_block_rehashed"] += not L.verify_chain(ch)[0]
        rogue = Identity("rogue"); L.pk["rogue"] = rogue.pk
        det["unregistered_sender"] += not L.submit(L.make_tx(rogue, "command", {"m": 1.0, "nact": 4}))
        tx = cmd(L, ids, 5); L.submit(tx); det["replay"] += not L.submit(tx)
        det["out_of_policy_command"] += not L.submit(L.make_tx(ids[0], "command", {"m": float(rng.uniform(1.9, 4)), "nact": 4}))
    return {k: v / trials for k, v in det.items()}


def inloop(seed):
    fc = load_forecaster(seed, "results/models", cfg)
    pol = torch.load(f"results/models/ppo_s{seed}.pt", weights_only=False)
    from evsim.agents import TorchPolicy
    pol = TorchPolicy(pol["kind"], pol["net"], cfg)
    env = EVNetEnv(cfg, E=1, forecast_fn=fc, forecast_mode="lstm")
    o = env.reset(50000 + seed)
    L, ids = fresh(4, 50, seed, cfg.N)
    sess_id = Identity("gateway")
    L.submit(L.register(sess_id))
    t_env = t_led = 0.0; n_tx = 0; prev_c = 0
    for t in range(96):
        t0 = time.perf_counter(); m, n = pol(o); o, r, d = env.step(m, n); t_env += time.perf_counter() - t0
        t0 = time.perf_counter()
        for k in range(cfg.N):
            L.submit(L.make_tx(ids[k], "command", {"m": float(m[0]), "nact": int(n[0]), "slot": t})); n_tx += 1
        comp = int(env.metrics["completed"][0] - prev_c); prev_c = env.metrics["completed"][0]
        for k in range(comp):
            L.submit(L.make_tx(sess_id, "session", {"energy_kwh": 12.0, "slot": t, "k": k})); n_tx += 1
        while L.pool: L.commit()
        t_led += time.perf_counter() - t0
    ok = L.verify_chain()[0]
    cm = np.array(L.stats["commit_ms"])
    return dict(env_policy_s=t_env, ledger_cpu_s=t_led, tx=n_tx, blocks=len(cm), commit_ms_mean=float(cm.mean()), commit_ms_p95=float(np.percentile(cm, 95)),
                chain_bytes=L.stats["bytes"], audit_ok=bool(ok), slot_seconds=900.0)


if __name__ == "__main__":
    seeds = list(map(int, sys.argv[1:])) or list(range(10))
    out = {"perf": [], "tamper": [], "inloop": []}
    for s in seeds:
        out["perf"].append(perf(s)); out["tamper"].append(tamper(s))
        if os.path.exists(f"results/models/ppo_s{s}.pt"):
            out["inloop"].append(inloop(s))
        print(time.strftime("%H:%M:%S"), "ledger seed", s, flush=True)
    json.dump(out, open("results/ledger/ledger.json", "w"))
