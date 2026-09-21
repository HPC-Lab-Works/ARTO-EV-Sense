"""Vectorised EV charging network simulator (numpy).

E independent copies of an N-station network step in lock-step. One episode = one day = 96 slots of 15 min.
Action (network level, shared by all stations): price multiplier m and number of active chargers per station.
"""
import numpy as np
from .config import SimConfig, STEPS_PER_DAY, DT_H, OBS_DIM
from .gen import gen_exo, arrival_rate_per_hour, station_types

SPD = STEPS_PER_DAY


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class EVNetEnv:
    def __init__(self, cfg: SimConfig = None, E=16, forecast_fn=None, forecast_mode="persist"):
        self.cfg = cfg or SimConfig()
        self.E = E
        self.forecast_fn = forecast_fn          # callable(exo) -> [E,T,H] forecast of headroom made at each slot
        self.forecast_mode = forecast_mode      # 'lstm' | 'persist' | 'oracle' | 'none'
        c = self.cfg
        self.K = c.C + c.Q
        self.stype = station_types(c.N)
        self.tier = np.array(c.tier)[self.stype]
        self.w_st = np.array(c.w_price)[self.stype]
        self.R = c.R_per_station * c.N
        self.pv_peak = c.pv_per_station * c.N
        self.record = False

    # ------------------------------------------------------------------ reset
    def reset(self, seed, dow0=None, demand_mult=1.0, heat=0.0, load_scale=1.0, wmult=1.0, record=False):
        c, E, N, K = self.cfg, self.E, self.cfg.N, self.K
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 7919)
        exo = gen_exo(rng, E, c, days=2, dow0=dow0, heat=heat, load_scale=load_scale)
        self.exo = exo
        sl = slice(SPD, 2 * SPD)
        self.head = exo["head"][:, sl]
        self.load = exo["load"][:, sl]
        self.pv = exo["pv"][:, sl]
        self.tou = exo["tou"][:, sl]
        self.hour = exo["hour"][0, sl]
        self.weekend = exo["weekend"][:, sl]
        dm = np.exp(rng.normal(0, c.demand_day_sigma, size=(E, 1)) + rng.normal(0, 0.08, size=(E, N)))
        self.rate = arrival_rate_per_hour(self.hour, self.weekend, self.stype) * dm[:, :, None] * c.demand_scale * demand_mult
        # local demand surges (events) on ~35 % of days: x1.8-2.5 arrivals at 30-60 % of stations for 2-4 h
        for e in range(E):
            if rng.random() < 0.35:
                st = rng.uniform(8, 20); du = rng.uniform(2, 4)
                sel = rng.random(N) < rng.uniform(0.3, 0.6)
                win = (self.hour >= st) & (self.hour < st + du)
                self.rate[e][np.ix_(sel, win)] *= rng.uniform(1.8, 2.5)
        self.w_eff = self.w_st * wmult
        # forecasts of headroom for the next H slots, made at each slot of the episode day
        H = c.forecast_horizon
        if self.forecast_mode == "none":
            self.fc = np.zeros((E, SPD, H))
        elif self.forecast_mode == "oracle":
            idx = np.minimum(np.arange(SPD)[:, None] + SPD + 1 + np.arange(H)[None, :], 2 * SPD - 1)
            # beyond the episode day the last known value is repeated
            self.fc = exo["head"][:, idx]
        elif self.forecast_mode == "persist":
            self.fc = np.repeat(self.head[:, :, None], H, axis=2)
        else:
            self.fc = self.forecast_fn(exo)[:, sl, :]
        # vehicle slots
        self.valid = np.zeros((E, N, K), bool)
        self.at_port = np.zeros((E, N, K), bool)
        self.rem = np.zeros((E, N, K))
        self.vpmax = np.zeros((E, N, K))
        self.wait = np.zeros((E, N, K), np.int16)
        self.pat = np.zeros((E, N, K), np.int16)
        self.t = 0
        self.prev_m = np.full(E, 1.0)
        self.prev_alpha = np.ones(E)
        self.P_prev = np.zeros(E)
        self.P_station = np.zeros((E, N))
        self.arr_prev = np.zeros(E)
        self.wait_frac_prev = np.zeros(E)
        self.metrics = {k: np.zeros(E) for k in [
            "revenue", "cost", "energy", "wait_slots", "admitted", "balked", "reneged", "completed",
            "over_events", "over_kwh", "occ_sum", "peak_ratio", "reward", "arrivals"]}
        self.record = record
        self.trace = {"P": [], "occ": [], "Ptot": []} if record else None
        return self.obs()

    # ------------------------------------------------------------------ observation
    def occupancy(self):
        return (self.valid & self.at_port).sum(-1) / self.cfg.C     # [E,N]

    def obs(self, P_report=None):
        """16-dim network-level observation. P_report overrides the reported previous charging load [E] (kW)."""
        c, t = self.cfg, min(self.t, SPD - 1)
        R = self.R
        Pobs = self.P_prev if P_report is None else P_report
        h = self.hour[t]
        fc = self.fc[:, t, :]
        queued = (self.valid & ~self.at_port).sum(-1)
        o = np.zeros((self.E, OBS_DIM))
        o[:, 0] = np.sin(2 * np.pi * h / 24)
        o[:, 1] = np.cos(2 * np.pi * h / 24)
        o[:, 2] = self.weekend[:, t]
        o[:, 3] = self.head[:, t] / R
        o[:, 4] = fc.mean(1) / R
        o[:, 5] = fc.min(1) / R
        o[:, 6] = Pobs / R
        o[:, 7] = (self.head[:, t] - Pobs) / R
        o[:, 8] = self.occupancy().mean(1)
        o[:, 9] = queued.mean(1) / c.Q
        o[:, 10] = self.wait_frac_prev
        o[:, 11] = self.arr_prev / 3.0
        o[:, 12] = self.tou[:, t] / 0.22
        o[:, 13] = self.pv[:, t] / self.pv_peak
        o[:, 14] = (self.prev_m - 1.2) / 0.6
        o[:, 15] = self.prev_alpha
        return o

    # ------------------------------------------------------------------ internals
    def _promote(self):
        c = self.cfg
        for _ in range(c.C):
            cnt = (self.valid & self.at_port).sum(-1)
            queued = self.valid & ~self.at_port
            key = np.where(queued, self.wait.astype(float), -1.0)
            j = key.argmax(-1)
            has = queued.any(-1) & (cnt < c.C)
            if not has.any():
                break
            ee, nn = np.nonzero(has)
            self.at_port[ee, nn, j[ee, nn]] = True

    def step(self, m, nact):
        """m: [E] price multiplier; nact: [E] int active chargers per station (1..C)."""
        c, E, N, K = self.cfg, self.E, self.cfg.N, self.K
        t = self.t
        m = np.clip(np.asarray(m, float), c.m_lo, c.m_hi)
        nact = np.clip(np.asarray(nact).astype(int), 1, c.C)
        m_st = m if m.ndim == 2 else np.repeat(m[:, None], N, 1)          # [E,N] per-station values
        n_st = nact if nact.ndim == 2 else np.repeat(nact[:, None], N, 1)
        m = m_st.mean(1)                                                   # network-level summaries for the state
        nact = n_st.mean(1)
        self._promote()
        # ---- arrivals (binomial thinning => Poisson-like), common random numbers via fixed-size draws
        share = _sig((c.x_out - m_st) / self.w_eff[None, :]) / _sig((c.x_out - 1.0) / self.w_eff[None, :])
        lam_slot = self.rate[:, :, t] * share * DT_H
        u = self.rng.random((E, N, c.amax))
        z = self.rng.standard_normal((E, N, c.amax))
        pv_u = self.rng.random((E, N, c.amax))
        pat_u = self.rng.integers(c.patience_slots[0], c.patience_slots[1] + 1, size=(E, N, c.amax))
        s2 = np.log(1 + (c.e_sd / c.e_mean) ** 2)
        e_req = np.clip(np.exp(np.log(c.e_mean) - s2 / 2 + np.sqrt(s2) * z), c.e_min, c.e_max)
        cp = np.cumsum(c.v_pmax_prob)
        vp = np.where(pv_u < cp[0], c.v_pmax[0], np.where(pv_u < cp[1], c.v_pmax[1], c.v_pmax[2]))
        arr = u < (lam_slot / c.amax)[:, :, None]
        n_arr = arr.sum((1, 2)).astype(float)
        balk = np.zeros(E)
        adm = np.zeros(E)
        for a in range(c.amax):
            mk = arr[:, :, a]
            if not mk.any():
                continue
            free = ~self.valid
            has_free = free.any(-1)
            idx = free.argmax(-1)
            ok = mk & has_free
            balk += (mk & ~has_free).sum(1)
            ee, nn = np.nonzero(ok)
            if len(ee) == 0:
                continue
            ii = idx[ee, nn]
            cnt_port = (self.valid & self.at_port).sum(-1)[ee, nn]
            self.valid[ee, nn, ii] = True
            self.at_port[ee, nn, ii] = cnt_port < c.C
            self.rem[ee, nn, ii] = e_req[ee, nn, a]
            self.vpmax[ee, nn, ii] = np.minimum(vp[ee, nn, a], c.pmax_kw)
            self.wait[ee, nn, ii] = 0
            self.pat[ee, nn, ii] = pat_u[ee, nn, a]
            adm += ok.sum(1)
        # ---- charging allocation: shortest-remaining-first among plugged vehicles, at most nact per station
        plugged = self.valid & self.at_port
        key = np.where(plugged, self.rem, np.inf)
        rank = key.argsort(-1).argsort(-1)
        charging = plugged & (rank < n_st[:, :, None])
        P = np.minimum(self.vpmax, self.rem / DT_H) * charging
        Ps = P.sum(-1)                                   # [E,N] kW
        Pt = Ps.sum(1)                                   # [E]
        # ---- economics and grid
        price = m_st * c.p_ref * self.tier[None, :]
        rev = (price * Ps * DT_H).sum(1)
        pv_t, tou_t, head_t = self.pv[:, t], self.tou[:, t], self.head[:, t]
        cost = tou_t * np.maximum(0.0, Pt - pv_t) * DT_H
        excess = np.maximum(0.0, Pt - head_t)
        over_kwh = excess * DT_H
        over_evt = (excess > c.overload_tol * self.R).astype(float)
        # ---- state update
        self.rem -= P * DT_H
        notch = self.valid & ~charging
        wait_cnt = notch.sum((1, 2)).astype(float)
        self.wait += notch.astype(np.int16)
        done = self.valid & (self.rem <= 1e-6)
        renege = self.valid & ~done & (self.wait > self.pat)
        leave = done | renege
        present = self.valid.sum((1, 2)).astype(float)
        self.valid &= ~leave
        self.at_port &= ~leave
        self.rem[leave] = 0
        reward = (rev - cost + c.w_serve * Ps.sum(1) * DT_H - c.w_wait * wait_cnt * DT_H - c.w_over * over_kwh) / (c.r_scale * N)
        # ---- bookkeeping
        M = self.metrics
        M["revenue"] += rev
        M["cost"] += cost
        M["energy"] += Ps.sum(1) * DT_H
        M["wait_slots"] += wait_cnt
        M["admitted"] += adm
        M["arrivals"] += n_arr
        M["balked"] += balk
        M["reneged"] += renege.sum((1, 2))
        M["completed"] += done.sum((1, 2))
        M["over_events"] += over_evt
        M["over_kwh"] += over_kwh
        M["occ_sum"] += plugged.sum((1, 2)) / (N * c.C)
        M["peak_ratio"] = np.maximum(M["peak_ratio"], Pt / np.maximum(head_t, 1.0))
        M["reward"] += reward
        self.P_prev = Pt
        self.P_station = Ps
        self.prev_m = m
        self.prev_alpha = nact / c.C
        self.arr_prev = adm / N
        self.wait_frac_prev = wait_cnt / np.maximum(present, 1.0)
        if self.record:
            self.trace["P"].append(Ps.copy())
            self.trace["occ"].append(plugged.sum(-1) / c.C)
            self.trace["Ptot"].append(Pt.copy())
        self.t += 1
        done_ep = self.t >= SPD
        return self.obs(), reward, done_ep

    def summary(self):
        """Episode metrics per env [E] (call after 96 steps)."""
        M = self.metrics
        adm = np.maximum(M["admitted"], 1.0)
        out = {
            "reward": M["reward"].copy(),
            "profit": M["revenue"] - M["cost"],
            "revenue": M["revenue"].copy(),
            "energy_kwh": M["energy"].copy(),
            "wait_min": M["wait_slots"] * DT_H * 60.0 / adm,
            "utilization": M["occ_sum"] / SPD,
            "over_events": M["over_events"].copy(),
            "over_kwh": M["over_kwh"].copy(),
            "served_frac": M["completed"] / adm,
            "renege_frac": M["reneged"] / adm,
            "admitted": M["admitted"].copy(),
            "peak_ratio": M["peak_ratio"].copy(),
        }
        return out


def action_to_env(a, C, cfg: SimConfig):
    """Continuous action a in [-1,1]^2 -> (m, nact)."""
    a = np.clip(a, -1, 1)
    m = cfg.m_lo + (a[..., 0] + 1) / 2 * (cfg.m_hi - cfg.m_lo)
    alpha = 1.0 / C + (a[..., 1] + 1) / 2 * (1 - 1.0 / C)
    nact = np.maximum(1, np.rint(alpha * C)).astype(int)
    return m, nact
