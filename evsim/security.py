"""Sensor and network telemetry, attack injection and anomaly detectors for the ARTO-EV testbench."""
import numpy as np, torch, torch.nn as nn, time
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from .config import SimConfig, STEPS_PER_DAY
from .env import EVNetEnv

SPD = STEPS_PER_DAY
WIN = 8
SENS = ["P", "V", "I", "Th"]
NETF = ["rate", "jit", "err", "auth", "pkt"]
NCH = len(SENS) + len(NETF) + 1      # 10 channels per slot: 4 sensor, 5 network, 1 physical-consistency residual
ATT = {"benign": 0, "FDI": 1, "MITM": 2, "DDoS": 3}


# ============================================================================ benign telemetry
class Telemetry:
    """Stateful per-slot telemetry model for E networks x N stations."""
    def __init__(self, E, N, rng, R):
        self.E, self.N, self.rng, self.R = E, N, rng, R
        self.v_off = rng.normal(0, 2.0, (1, N)); self.pf = rng.uniform(0.97, 0.99, (1, N))
        self.gain = rng.normal(1.0, 0.004, (1, N)); self.a_th = rng.uniform(0.0005, 0.0007, (1, N))
        self.theta = None
        self.prev = None

    def step(self, Ps, occ, temp, rho):
        rng, E, N = self.rng, self.E, self.N
        P = (Ps + 0.3) * self.gain * (1 + rng.normal(0, 0.008, (E, N))) + rng.normal(0, 0.12, (E, N))
        V = 400.0 * (1 - 0.05 * np.clip(rho, 0, 1.3))[:, None] + self.v_off + rng.normal(0, 0.8, (E, N))
        I = (Ps + 0.3) * 1000.0 / (np.sqrt(3) * V * self.pf) * (1 + rng.normal(0, 0.008, (E, N)))
        target = temp[:, None] + self.a_th * (Ps * 1000.0 / (np.sqrt(3) * 400 * self.pf)) ** 2
        self.theta = target.copy() if self.theta is None else self.theta + 0.25 * (target - self.theta)
        Th = self.theta + rng.normal(0, 0.3, (E, N))
        rate = 8 + 3 * occ + rng.normal(0, 0.6, (E, N))
        jit = rng.lognormal(np.log(4.0), 0.25, (E, N))
        err = np.abs(rng.normal(0.4, 0.15, (E, N)))
        auth = rng.poisson(0.3, (E, N)).astype(float)
        pkt = rng.normal(240, 10, (E, N))
        return dict(P=P, V=V, I=I, Th=Th, rate=rate, jit=jit, err=err, auth=auth, pkt=pkt)


# ============================================================================ attacks
class AttackPlan:
    """One optional attack per (env, station): type, start slot, duration and severity parameters."""
    def __init__(self, rng, E, N, p_attack=0.6, types=(1, 2, 3), tmin=8, tmax=80, dmin=6, dmax=20, station_mask=None,
                 fixed_window=None):
        self.E, self.N = E, N
        has = rng.random((E, N)) < p_attack
        if station_mask is not None:
            has &= station_mask
        ty = rng.choice(types, size=(E, N))
        self.type = np.where(has, ty, 0)
        if fixed_window is not None:            # (start[E], dur[E]) shared by all stations of an env
            self.start = np.repeat(fixed_window[0][:, None], N, 1); self.dur = np.repeat(fixed_window[1][:, None], N, 1)
        else:
            self.dur = rng.integers(dmin, dmax + 1, (E, N))
            self.start = rng.integers(tmin, tmax, (E, N))
        self.sev = rng.random((E, N))
        self.mode = rng.integers(0, 3, (E, N))          # FDI: 0 scale, 1 bias, 2 ramp
        self.sign = rng.choice([-1.0, 1.0], (E, N))
        self.stealth = rng.random((E, N)) < 0.5
        self.mitm_p = rng.random((E, N)) < 0.5
        self.rng = rng
        self.eff_log = []

    def active(self, t):
        return (self.type > 0) & (t >= self.start) & (t < self.start + self.dur)

    def labels(self, T):
        lab = np.zeros((self.E, self.N, T), int)
        for t in range(T):
            lab[:, :, t] = np.where(self.eff_log[t], self.type, 0)
        return lab

    def apply(self, r, t, prev_rep, pf, gain=None):
        """Return attacked readings dict (copy). prev_rep: previous reported readings (for DDoS staleness)."""
        rng = self.rng
        a = self.active(t)
        out = {k: v.copy() for k, v in r.items()}
        if not a.any():
            return out, a
        el = np.clip(t - self.start, 0, None) / np.maximum(self.dur, 1)
        # ---- FDI on P (and consistently on I when stealthy)
        f = a & (self.type == 1)
        scale = 1 + self.sign * (0.15 + 0.5 * self.sev)
        Pf = np.where(self.mode == 0, r["P"] * scale,
                      np.where(self.mode == 1, r["P"] + self.sign * (2 + 18 * self.sev), r["P"] + self.sign * (2 + 18 * self.sev) * el))
        Pf = np.maximum(Pf, 0.0)
        out["P"] = np.where(f, Pf, out["P"])
        eff = a.copy()
        eff = np.where(f, np.abs(Pf - r["P"]) > 1.0, eff)       # FDI counts only when the reading is materially changed
        If = Pf * 1000.0 / (np.sqrt(3) * r["V"] * pf)
        out["I"] = np.where(f & self.stealth, If, out["I"])
        # ---- MITM: relay delay and re-encapsulation, sometimes payload tampering
        m = a & (self.type == 2)
        out["jit"] = np.where(m, r["jit"] * (1.8 + 4 * self.sev), out["jit"])
        out["pkt"] = np.where(m, r["pkt"] + 20 + 100 * self.sev, out["pkt"])
        out["auth"] = np.where(m, r["auth"] + rng.poisson(0.4 + 1.6 * self.sev), out["auth"])
        out["err"] = np.where(m, r["err"] + 0.3 + 2 * self.sev, out["err"])
        out["rate"] = np.where(m, r["rate"] * (1 + 0.1 * self.sev), out["rate"])
        mp = m & self.mitm_p
        out["P"] = np.where(mp, r["P"] * (1 + self.sign * (0.05 + 0.10 * self.sev)), out["P"])
        # ---- DDoS: flooding, errors, stale telemetry
        d = a & (self.type == 3)
        out["rate"] = np.where(d, r["rate"] * (4 + 36 * self.sev), out["rate"])
        out["jit"] = np.where(d, r["jit"] * (2 + 6 * self.sev), out["jit"])
        out["err"] = np.where(d, r["err"] + 3 + 22 * self.sev, out["err"])
        out["auth"] = np.where(d, r["auth"] + rng.poisson(2 + 8 * self.sev), out["auth"])
        out["pkt"] = np.where(d, r["pkt"] * (0.7 - 0.4 * self.sev), out["pkt"])
        if prev_rep is not None:
            stale = d & (rng.random(d.shape) < np.minimum(0.9, 0.1 + out["err"] / 30.0))
            for k in SENS:
                out[k] = np.where(stale, prev_rep[k], out[k])
        return out, eff


# ============================================================================ windows / features
def to_feat(r):
    """dict of [E,N] -> [E,N,NCH] transformed features for one slot."""
    Ph = np.sqrt(3) * r["V"] * r["I"] * 0.98 / 1000.0
    resid = (r["P"] - Ph) / (0.3 + 0.02 * np.abs(Ph))          # preprocessing-layer consistency check P vs V*I
    return np.stack([r["P"], r["V"], r["I"], r["Th"], np.log(np.maximum(r["rate"], 1e-3)), np.log(np.maximum(r["jit"], 1e-3)),
                     np.log1p(np.maximum(r["err"], 0)), np.log1p(r["auth"]), r["pkt"], resid], -1)


def windows(F):
    """F [E,N,T,NCH] -> [E,N,T,WIN,NCH] (front padded with the first slot)."""
    pad = np.concatenate([np.repeat(F[:, :, :1], WIN - 1, 2), F], 2)
    T = F.shape[2]
    return np.stack([pad[:, :, k:k + T] for k in range(WIN)], 3)


class Scaler:
    def fit(self, W):
        x = W.reshape(-1, W.shape[-1]); self.mu, self.sd = x.mean(0), x.std(0) + 1e-6; return self
    def __call__(self, W):
        return ((W - self.mu) / self.sd).astype(np.float32)


# ============================================================================ data generation
def rollout_benign_and_attacked(cfg, seed, E, forecast_mode="persist", attack_kwargs=None, with_attacks=True,
                                station_offset_seed=None):
    """Roll out E days under a random behaviour policy and generate telemetry (benign and attacked).
    Returns F [E,N,T,NCH] attacked features, Fb benign features, labels [E,N,T], stype [N]."""
    rng = np.random.default_rng(seed)
    env = EVNetEnv(cfg, E=E, forecast_mode=forecast_mode)
    o = env.reset(seed + 1, record=True)
    N, R = cfg.N, env.R
    mb = rng.uniform(0.8, 1.5, E); nb = rng.choice([3, 4, 5, 6], E)
    tel = Telemetry(E, N, np.random.default_rng((station_offset_seed if station_offset_seed is not None else 777)), R)
    tel.rng = rng
    plan = AttackPlan(rng, E, N, **(attack_kwargs or {})) if with_attacks else None
    F, Fb, prev = [], [], None
    for t in range(SPD):
        m = np.clip(mb * (1 + 0.15 * rng.standard_normal(E)), 0.6, 1.8)
        n = np.clip(nb + rng.integers(-1, 2, E), 2, cfg.C)
        env.step(m, n)
        rho = (env.load[:, t] + env.P_prev - env.pv[:, t]) / R
        occ = env.occupancy()
        r = tel.step(env.P_station, occ, env.exo["temp"][:, SPD + t], rho)
        Fb.append(to_feat(r))
        if plan is not None:
            r2, eff = plan.apply(r, t, prev, tel.pf)
            plan.eff_log.append(eff)
            prev = r2
            F.append(to_feat(r2))
        else:
            F.append(Fb[-1])
    F = np.stack(F, 2); Fb = np.stack(Fb, 2)
    lab = plan.labels(SPD) if plan is not None else np.zeros((E, N, SPD), int)
    if plan is not None:      # window label: any effectively attacked slot inside the 8-slot window (type of the latest one)
        padl = np.concatenate([np.zeros((E, N, WIN - 1), int), lab], 2)
        wl = np.zeros_like(lab)
        for k in range(WIN):
            seg = padl[:, :, k:k + SPD]
            wl = np.where(seg > 0, seg, wl)
        lab = wl
        S = dict(start=plan.start, dur=plan.dur, type=plan.type, stealth=plan.stealth, mode=plan.mode)
        return F, Fb, lab, env.stype, S
    return F, Fb, lab, env.stype, None


def build_sets(cfg, seed, n_train=32, n_val=8, n_test=24):
    """Train/val (benign only) and test (with attacks) windows, all as [n,WIN*NCH] arrays with station ids."""
    _, Ftr, _, st, _ = rollout_benign_and_attacked(cfg, seed * 10 + 1, n_train, with_attacks=False)
    _, Fva, _, _, _ = rollout_benign_and_attacked(cfg, seed * 10 + 2, n_val, with_attacks=False)
    Fte, Fte_b, lab, _, plan = rollout_benign_and_attacked(cfg, seed * 10 + 3, n_test, with_attacks=True)
    sc = Scaler().fit(windows(Ftr))
    prep = lambda F: sc(windows(F))                                 # [E,N,T,WIN,NCH]
    return dict(tr=prep(Ftr), va=prep(Fva), te=prep(Fte), lab=lab, sc=sc, stype=st, plan=plan,
                raw_tr=windows(Ftr), raw_va=windows(Fva), raw_te=windows(Fte))


# ============================================================================ detectors
class Detector:
    name = "base"
    def flat(self, W):
        return W.reshape(-1, W.shape[-2] * W.shape[-1])
    def calibrate(self, va, q=0.99):
        self.thr = float(np.quantile(self.score(va), q)); return self
    def alarm(self, W):
        return self.score(W) > self.thr


class MLPAE(nn.Module):
    def __init__(self, d=WIN * NCH, h=64, z=24):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, z), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))
    def forward(self, x):
        return self.dec(self.enc(x))


def _train_ae(model, X, seed, epochs=25, lr=2e-3, bs=512):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.as_tensor(X); n = len(X)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            x = X[perm[i:i + bs]]
            loss = ((model(x) - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


class AEDetector(Detector):
    name = "Autoencoder (ARTO-EV)"
    def __init__(self, seed=0, epochs=25, model=None):
        self.seed, self.epochs, self.model = seed, epochs, model
    def fit(self, tr):
        X = self.flat(tr)
        self.model = _train_ae(MLPAE(), X, self.seed, self.epochs); return self
    def score(self, W):
        X = torch.as_tensor(self.flat(W))
        with torch.no_grad():
            return ((self.model(X) - X) ** 2).mean(1).numpy()


class LSTMAE(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.enc = nn.LSTM(NCH, h, batch_first=True); self.dec = nn.LSTM(h, h, batch_first=True); self.out = nn.Linear(h, NCH)
    def forward(self, x):
        _, (hn, _) = self.enc(x)
        z = hn[-1][:, None, :].repeat(1, x.shape[1], 1)
        y, _ = self.dec(z)
        return self.out(y)


class LSTMAEDetector(Detector):
    name = "LSTM autoencoder"
    def __init__(self, seed=0, epochs=12): self.seed, self.epochs = seed, epochs
    def fit(self, tr):
        self.model = _train_ae(LSTMAE(), tr.reshape(-1, WIN, NCH), self.seed, self.epochs, lr=3e-3); return self
    def score(self, W):
        X = torch.as_tensor(W.reshape(-1, WIN, NCH))
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                x = X[i:i + 8192]; out.append(((self.model(x) - x) ** 2).mean((1, 2)).numpy())
        return np.concatenate(out)


class IFDetector(Detector):
    name = "Isolation forest"
    def __init__(self, seed=0): self.seed = seed
    def fit(self, tr):
        self.m = IsolationForest(n_estimators=200, max_samples=1024, random_state=self.seed, n_jobs=2).fit(self.flat(tr)); return self
    def score(self, W):
        return -self.m.score_samples(self.flat(W))


class OCSVMDetector(Detector):
    name = "One-class SVM"
    def __init__(self, seed=0, n_fit=5000): self.seed, self.n_fit = seed, n_fit
    def fit(self, tr):
        X = self.flat(tr); rng = np.random.default_rng(self.seed)
        X = X[rng.choice(len(X), min(self.n_fit, len(X)), replace=False)]
        self.m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02).fit(X); return self
    def score(self, W):
        X = self.flat(W); out = []
        for i in range(0, len(X), 20000):
            out.append(-self.m.decision_function(X[i:i + 20000]))
        return np.concatenate(out)


class SignatureIDS(Detector):
    """Threshold/signature IDS: per-feature 99.9 % benign limits on the network features and sensor ranges, plus a
    physical-consistency rule P = sqrt(3) V I pf. Fixed rules, no learning; alarm if any rule fires on the latest slot."""
    name = "Signature IDS"
    def fit(self, raw_tr, q=0.999):
        last = raw_tr[:, -1, :].reshape(-1, NCH) if raw_tr.ndim == 3 else raw_tr[..., -1, :].reshape(-1, NCH)
        self.hi = np.quantile(last, q, axis=0); self.lo = np.quantile(last, 1 - q, axis=0)
        use_hi = [0, 1, 3] + [4, 5, 6, 7]        # P,V,Th upper + network features upper
        self.mask_hi = np.zeros(NCH, bool); self.mask_hi[use_hi] = True
        self.mask_lo = np.zeros(NCH, bool); self.mask_lo[[1, 8]] = True   # V lower, packet size lower
        self.mask_hi[8] = True
        r = self._resid(last); self.rthr = np.quantile(r, q)
        self.med = np.median(last, 0); return self
    @staticmethod
    def _resid(L):
        P, V, I = L[:, 0], L[:, 1], L[:, 2]
        Ph = np.sqrt(3) * V * I * 0.98 / 1000.0
        return np.abs(P - Ph) / (0.3 + 0.02 * np.abs(Ph))
    def score(self, raw):
        L = raw.reshape(-1, WIN, NCH).reshape(-1, NCH)                   # every slot of every window
        span = np.maximum(self.hi - self.med, 1e-6); spanl = np.maximum(self.med - self.lo, 1e-6)
        up = ((L - self.med) / span)[:, self.mask_hi].max(1)
        dn = ((self.med - L) / spanl)[:, self.mask_lo].max(1)
        cons = self._resid(L) / max(self.rthr, 1e-6)
        s = np.maximum.reduce([up, dn, cons]).reshape(-1, WIN)
        return s.max(1)                                                   # window alarm if any slot violates a rule


# ============================================================================ metrics
def auroc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def det_metrics(alarm, score, lab):
    """alarm/score/lab flattened arrays (lab int 0..3)."""
    y = lab > 0
    tp = (alarm & y).sum(); fp = (alarm & ~y).sum(); tn = (~alarm & ~y).sum(); fn = (~alarm & y).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
    out = dict(accuracy=(tp + tn) / len(y), precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9), fpr=fpr,
               bal_acc=(rec + 1 - fpr) / 2, auroc=auroc(score, y))
    for k, v in ATT.items():
        if v:
            mk = lab == v
            out[f"recall_{k}"] = float(alarm[mk].mean()) if mk.any() else float("nan")
    return {k: float(v) for k, v in out.items()}


def segment_latency(alarm, lab_3d, plan_start, plan_type, plan_dur):
    """alarm [E,N,T] bool. Returns (segment detection rate, median latency in slots among detected)."""
    E, N, T = alarm.shape
    lat = []; det = 0; tot = 0
    for e in range(E):
        for n in range(N):
            if plan_type[e, n] == 0: continue
            s, d = plan_start[e, n], plan_dur[e, n]
            seg = alarm[e, n, s:min(s + d, T)]
            tot += 1
            if seg.any():
                det += 1; lat.append(int(np.argmax(seg)))
    return det / max(tot, 1), (float(np.median(lat)) if lat else float("nan"))
