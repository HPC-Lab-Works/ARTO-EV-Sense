"""Closed-loop evaluation of controllers under sensor/network attacks, with optional detector-based screening."""
import numpy as np
from .config import SimConfig, STEPS_PER_DAY
from .env import EVNetEnv
from .security import (Telemetry, AttackPlan, to_feat, Scaler, SignatureIDS, WIN, NCH, ATT)
import time
from .ledger import Identity, canon, verify

SPD = STEPS_PER_DAY
SAFE_M, SAFE_N = 1.2, 4          # edge fallback command applied by a gateway whose stream is flagged


class CmdChannel:
    """Command path from the gateway to the stations, with optional authentication (real Ed25519 signatures).
    mode: 'none' (commands are executed as received), 'sig' (signature check only), 'full' (signature, per-station nonce
    freshness and policy bounds, i.e. the rules of the ledger contract). variant selects how the attacker manipulates the
    command channel of an attacked station (attack type 2): 'forge' (payload replaced, no valid signature), 'stolen'
    (payload replaced and re-signed with the gateway key), 'replay' (a previously signed off-peak command is resent).
    A command that fails verification is replaced by the last executed command ('hold') or by the safe command ('safe')."""
    REPLAY_SLOT = 8

    def __init__(self, E, N, C, mode="none", fallback="hold", variant="forge", m_lo=0.6, m_hi=1.8):
        self.E, self.N, self.C, self.mode, self.fallback, self.variant = E, N, C, mode, fallback, variant
        self.m_lo, self.m_hi = m_lo, m_hi
        self.gw = Identity("gateway")
        self.last_nonce = -np.ones((E, N), int)
        self.cap = {}
        self.cap_m = self.cap_n = None
        self.st = dict(sign_s=0.0, verify_s=0.0, n_sign=0, n_verify=0, rej_bad_signature=0, rej_replay=0, rej_out_of_policy=0,
                       rej_legit=0, acc_malicious=0, delivered_malicious=0)

    def _body(self, msg):
        return canon({"sender": "gateway", "type": "command", "payload": msg["payload"], "nonce": msg["nonce"]})

    def _sign(self, e, i, m, n, t):
        payload = {"m": round(float(m), 4), "nact": int(n), "slot": int(t), "station": int(i), "env": int(e)}
        msg = {"payload": payload, "nonce": int(t)}
        t0 = time.perf_counter(); msg["sig"] = self.gw.sign(self._body(msg)); self.st["sign_s"] += time.perf_counter() - t0
        self.st["n_sign"] += 1
        return msg

    def _verify(self, e, i, msg):
        t0 = time.perf_counter(); ok = verify(self.gw.pk, msg["sig"], self._body(msg)); self.st["verify_s"] += time.perf_counter() - t0
        self.st["n_verify"] += 1
        if not ok:
            return False, "bad_signature"
        if self.mode == "full":
            if msg["nonce"] <= self.last_nonce[e, i]:
                return False, "replay"
            p = msg["payload"]
            if not (self.m_lo <= p["m"] <= self.m_hi and 1 <= p["nact"] <= self.C):
                return False, "out_of_policy"
            self.last_nonce[e, i] = msg["nonce"]
        return True, "ok"

    def deliver(self, t, m_vec, n_vec, act, typ, prev_m, prev_n, rng, safe=(1.2, 4)):
        E, N, C = self.E, self.N, self.C
        # the gateway issues commands inside the action range (the same clipping that the simulator applies to any command)
        m_vec = np.clip(np.asarray(m_vec, float), self.m_lo, self.m_hi); n_vec = np.clip(np.asarray(n_vec).astype(int), 1, C)
        m_st = np.repeat(m_vec[:, None], N, 1); n_st = np.repeat(n_vec.astype(float)[:, None], N, 1)
        if t == self.REPLAY_SLOT:
            self.cap_m, self.cap_n = m_st.copy(), n_st.copy()
        mitm = act & (typ == 2)
        lost = (act & (typ == 3)) & (rng.random((E, N)) < 0.8)
        if self.mode == "none":
            if self.variant == "replay" and self.cap_m is not None:
                m_rx = np.where(mitm, self.cap_m, m_st); n_rx = np.where(mitm, self.cap_n, n_st)
            else:
                m_rx = np.where(mitm, 0.6, m_st); n_rx = np.where(mitm, C, n_st)
            self.st["delivered_malicious"] += int(mitm.sum()); self.st["acc_malicious"] += int(mitm.sum())
            return np.where(lost, prev_m, m_rx), np.where(lost, prev_n, n_rx)
        m_rx = m_st.copy(); n_rx = n_st.copy(); acc = np.zeros((E, N), bool)
        for e in range(E):
            for i in range(N):
                if lost[e, i]:
                    continue
                msg = self._sign(e, i, m_st[e, i], n_st[e, i], t)
                if t == self.REPLAY_SLOT:
                    self.cap[(e, i)] = msg
                if mitm[e, i]:
                    self.st["delivered_malicious"] += 1
                    if self.variant == "forge":
                        msg = {"payload": dict(msg["payload"], m=0.6, nact=C), "nonce": msg["nonce"], "sig": msg["sig"]}
                    elif self.variant == "stolen":
                        msg = self._sign(e, i, 0.6, C, t)
                    elif self.variant == "replay":
                        msg = self.cap[(e, i)]
                ok, why = self._verify(e, i, msg)
                if ok:
                    acc[e, i] = True; m_rx[e, i] = msg["payload"]["m"]; n_rx[e, i] = msg["payload"]["nact"]
                    if mitm[e, i]:
                        self.st["acc_malicious"] += 1
                else:
                    self.st["rej_" + why] += 1
                    if not mitm[e, i]:
                        self.st["rej_legit"] += 1
        fb_m, fb_n = (prev_m, prev_n) if self.fallback == "hold" else (np.full((E, N), safe[0]), np.full((E, N), safe[1], float))
        m_out = np.where(acc, m_rx, np.where(lost, prev_m, fb_m)); n_out = np.where(acc, n_rx, np.where(lost, prev_n, fb_n))
        return m_out, n_out


def _score(det, raw_win, sc_win):
    if isinstance(det, SignatureIDS):
        return det.score(raw_win)
    return det.score(sc_win)


def run_closed_loop(cfg, policy, eval_seed, scaler, detector=None, atype=0, frac=0.4, E=30, forecaster=None,
                    forecast_mode="lstm", tel_seed=777, window=(48, 16), sever=None, fixed_start=None, mask=False, chan=None):
    """Run E days. atype 0 none, 1 FDI, 2 MITM, 3 DDoS. detector None => no screening.
    Returns (metrics dict of arrays [E], detection stats dict)."""
    env = EVNetEnv(cfg, E=E, forecast_fn=forecaster, forecast_mode=forecast_mode)
    o = env.reset(eval_seed)
    N, C, R = cfg.N, cfg.C, env.R
    rng = np.random.default_rng(eval_seed + 31337)
    tel = Telemetry(E, N, np.random.default_rng(tel_seed), R); tel.rng = rng
    if atype:
        start = rng.integers(48, 72, E); dur = np.full(E, window[1])
        plan = AttackPlan(rng, E, N, p_attack=frac, types=(atype,), fixed_window=(start, dur))
        if sever is not None:
            plan.sev[:] = sever
        if mask:                                    # FDI that hides load: every affected reading reduced to 35-50 % of its true value
            plan.sign[:] = -1.0; plan.mode[:] = 0; plan.sev[:] = rng.uniform(0.7, 1.0, (E, N))
    else:
        plan = AttackPlan(rng, E, N, p_attack=0.0, types=(1,))
    buf_raw = np.zeros((E, N, WIN, NCH)); buf_sc = np.zeros((E, N, WIN, NCH), np.float32)
    P_rep = np.zeros((E, N)); last_valid = np.zeros((E, N)); flagged = np.zeros((E, N), bool)
    prev_rep = None; prev_m = np.ones((E, N)); prev_n = np.full((E, N), float(C))
    tp = fp = tn = fn = 0
    n_flag_slots = 0
    first = True
    for t in range(SPD):
        Pobs = np.maximum(P_rep - 0.3, 0).sum(1) if t > 0 else np.zeros(E)
        obs = env.obs(P_report=Pobs)
        m, n = policy(obs)
        act = plan.active(t)
        if chan is None:
            m_st = np.repeat(np.asarray(m, float)[:, None], N, 1); n_st = np.repeat(np.asarray(n, float)[:, None], N, 1)
            # command channel attacks
            mitm = act & (plan.type == 2)
            ddos = act & (plan.type == 3)
            m_st = np.where(mitm, 0.6, m_st); n_st = np.where(mitm, C, n_st)                 # tampered: cheap price, all chargers
            lost = ddos & (rng.random((E, N)) < 0.8)
            m_st = np.where(lost, prev_m, m_st); n_st = np.where(lost, prev_n, n_st)       # command lost: station keeps the last one
        else:
            m_st, n_st = chan.deliver(t, m, n, act, plan.type, prev_m, prev_n, rng)
        if detector is not None:                                                        # flagged gateway -> safe local command
            m_st = np.where(flagged, SAFE_M, m_st); n_st = np.where(flagged, SAFE_N, n_st)
        prev_m, prev_n = m_st, n_st
        env.step(m_st, n_st)
        rho = (env.load[:, t] + env.P_prev - env.pv[:, t]) / R
        r = tel.step(env.P_station, env.occupancy(), env.exo["temp"][:, SPD + t], rho)
        r2, eff = plan.apply(r, t, prev_rep, tel.pf)
        prev_rep = r2
        f = to_feat(r2)
        buf_raw = np.concatenate([buf_raw[:, :, 1:], f[:, :, None]], 2) if t > 0 else np.repeat(f[:, :, None], WIN, 2)
        if detector is not None:
            sc = scaler(buf_raw).astype(np.float32)
            s = _score(detector, buf_raw.reshape(-1, WIN, NCH), sc.reshape(-1, WIN, NCH))
            alarm = (s > detector.thr).reshape(E, N)
        else:
            alarm = np.zeros((E, N), bool)
        y = eff
        if detector is not None:
            tp += (alarm & y).sum(); fp += (alarm & ~y).sum(); tn += (~alarm & ~y).sum(); fn += (~alarm & y).sum()
        flagged = alarm
        rep = r2["P"].copy()
        if detector is not None:
            rep = np.where(alarm, last_valid, rep)             # sample-and-hold replacement of flagged readings
            last_valid = np.where(alarm, last_valid, rep)
        else:
            last_valid = rep
        P_rep = rep
    S = env.summary()
    det = dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    if chan is not None:
        det["chan"] = dict(chan.st)
    return S, det
