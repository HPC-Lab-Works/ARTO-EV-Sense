"""Security testbed: 10 seeds. Trains detectors on benign data only and evaluates on attacked held-out days."""
import sys, json, time, os, pickle, numpy as np, torch
torch.set_num_threads(1)
from evsim.security import *
cfg = SimConfig()
os.makedirs("results/security", exist_ok=True)


def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)


def run(seed):
    S = build_sets(cfg, seed)
    tr, va, te = [S[k].reshape(-1, WIN, NCH) for k in ("tr", "va", "te")]
    rtr, rva, rte = [S["raw_" + k].reshape(-1, WIN, NCH) for k in ("tr", "va", "te")]
    lab = S["lab"].reshape(-1); E, N, T = S["lab"].shape
    plan = S["plan"]
    stl = np.repeat(plan["stealth"][:, :, None], T, 2).reshape(-1)
    res = {"prevalence": float((lab > 0).mean()), "n_windows": int(len(lab)),
           "attack_share": {k: float((lab == v).mean()) for k, v in ATT.items() if v}}
    dets = []
    for D in [AEDetector(seed, 25), LSTMAEDetector(seed, 12), IFDetector(seed), OCSVMDetector(seed), SignatureIDS()]:
        t0 = time.time()
        if isinstance(D, SignatureIDS):
            D.fit(rtr); D.calibrate(rva); trn = time.time() - t0
            t1 = time.time(); s = D.score(rte); inf = (time.time() - t1) / len(rte)
        else:
            D.fit(tr); D.calibrate(va); trn = time.time() - t0
            t1 = time.time(); s = D.score(te); inf = (time.time() - t1) / len(te)
        a = s > D.thr
        m = det_metrics(a, s, lab)
        det_rate, lat = segment_latency(a.reshape(E, N, T), S["lab"], plan["start"], plan["type"], plan["dur"])
        m.update(segment_detection=det_rate, median_latency_slots=lat, train_s=trn, infer_us_per_window=inf * 1e6)
        f1 = lab == 1
        m["recall_FDI_stealth"] = float(a[f1 & stl].mean()) if (f1 & stl).any() else float("nan")
        m["recall_FDI_nonstealth"] = float(a[f1 & ~stl].mean()) if (f1 & ~stl).any() else float("nan")
        res[D.name] = m
        dets.append(D)
        log(f"seed {seed} {D.name}: F1 {m['f1']:.3f} acc {m['accuracy']:.3f} fpr {m['fpr']:.3f} auroc {m['auroc']:.3f} rec F/M/D {m['recall_FDI']:.2f}/{m['recall_MITM']:.2f}/{m['recall_DDoS']:.2f}")
        if D.name.startswith("Autoencoder"):
            # operating points and FDI split by stealth flag
            ops = {}
            for q in (0.95, 0.99, 0.999):
                thr = float(np.quantile(D.score(va), q)); aa = s > thr
                ops[str(q)] = dict(fpr=float(aa[lab == 0].mean()), recall=float(aa[lab > 0].mean()))
            res["ae_operating_points"] = ops
    pickle.dump(dict(dets=dets, scaler=S["sc"]), open(f"results/models/det_s{seed}.pkl", "wb"))
    json.dump(res, open(f"results/security/sec_s{seed}.json", "w"))


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        run(s)
