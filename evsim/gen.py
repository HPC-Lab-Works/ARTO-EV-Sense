"""Synthetic exogenous generator: temperature, cloud cover, PV, feeder base load, tariff and demand profiles."""
import numpy as np
from .config import SimConfig, STEPS_PER_DAY, DT_H

SPD = STEPS_PER_DAY


def _g(h, mu, sd):
    return np.exp(-0.5 * ((h - mu) / sd) ** 2)


def _ar1(rng, shape, phi, sigma):
    """AR(1) noise along the last axis with stationary std sigma."""
    e = rng.standard_normal(shape) * sigma * np.sqrt(1 - phi ** 2)
    out = np.empty(shape)
    out[..., 0] = rng.standard_normal(shape[:-1]) * sigma
    for k in range(1, shape[-1]):
        out[..., k] = phi * out[..., k - 1] + e[..., k]
    return out


def base_profile(hour, weekend):
    """Feeder base load as a fraction of the transformer rating (before temperature effect)."""
    wd = 0.30 + 0.16 * _g(hour, 8.0, 2.0) + 0.10 * _g(hour, 13.0, 3.0) + 0.30 * _g(hour, 19.5, 2.2)
    we = 0.29 + 0.10 * _g(hour, 11.0, 3.5) + 0.24 * _g(hour, 19.5, 2.8)
    return np.where(weekend, we, wd)


def tou_price(hour, cfg: SimConfig):
    p = np.full_like(hour, cfg.tou_mid, dtype=float)
    p = np.where((hour < 6) | (hour >= 22), cfg.tou_offpeak, p)
    p = np.where((hour >= 17) & (hour < 22), cfg.tou_peak, p)
    return p


def station_types(N):
    """0 residential, 1 commercial, 2 highway; proportions 4:3:3."""
    t = np.zeros(N, dtype=int)
    n0, n1 = int(round(0.4 * N)), int(round(0.3 * N))
    t[n0:n0 + n1] = 1
    t[n0 + n1:] = 2
    return t


def arrival_rate_per_hour(hour, weekend, stype):
    """Vehicles per hour per station at reference price. hour [T], weekend [E,T] bool, stype [N] -> [E,N,T]."""
    h = hour[None, None, :]
    we = weekend[:, None, :]
    res_wd = 0.3 + 3.2 * _g(h, 19.5, 2.0) + 0.8 * _g(h, 7.5, 1.5)
    res_we = 0.4 + 2.2 * _g(h, 14.0, 4.0) + 2.0 * _g(h, 19.5, 2.5)
    com_wd = 0.15 + 3.6 * _g(h, 8.5, 1.5) + 1.2 * _g(h, 13.0, 2.0)
    com_we = 0.10 + 0.8 * _g(h, 13.0, 3.0) + 0 * h
    hwy_wd = 0.3 + 2.2 * _g(h, 12.0, 4.0) + 1.8 * _g(h, 17.0, 2.5)
    hwy_we = 0.3 + 3.0 * _g(h, 13.0, 4.0) + 0 * h
    res = np.where(we, res_we, res_wd)
    com = np.where(we, com_we, com_wd)
    hwy = np.where(we, hwy_we, hwy_wd)
    st = stype[None, :, None]
    return np.where(st == 0, res, np.where(st == 1, com, hwy))


def gen_exo(rng, E, cfg: SimConfig, days=2, dow0=None, heat=0.0, load_scale=1.0):
    """Generate exogenous series for E environments over `days` days.

    Returns dict with arrays [E, T] (T = days*96): hour, weekend, temp, cloud, pv, load (feeder base load),
    headroom A = R - load + pv, tou (grid price). R and PV scale with N.
    heat: additive temperature shift (deg C) used by the robustness experiment.
    """
    N = cfg.N
    T = days * SPD
    t = np.arange(T)
    hour = (t % SPD) / 4.0
    if dow0 is None:
        dow0 = rng.integers(0, 7, size=E)
    dow = (dow0[:, None] + t[None, :] // SPD) % 7
    weekend = dow >= 5
    R = cfg.R_per_station * N
    # temperature: seasonal mean + diurnal cycle + day offsets + slow AR noise
    tmean = rng.uniform(2.0, 32.0, size=E)[:, None]
    day_off = np.repeat(rng.normal(0, 2.0, size=(E, days)), SPD, axis=1)
    temp = tmean + 6.0 * np.sin(2 * np.pi * (hour[None, :] - 9.0) / 24.0) + day_off + _ar1(rng, (E, T), 0.98, 1.0) + heat
    # cloud cover factor in [0.15, 1]
    z = _ar1(rng, (E, T), 0.985, 1.2) + rng.normal(0.8, 0.8, size=(E, 1))
    cloud = 0.15 + 0.85 / (1 + np.exp(-z))
    pv_clear = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0, None) ** 1.3
    pv = cfg.pv_per_station * N * pv_clear[None, :] * cloud
    # feeder base load (fraction of R), scaled noise so that aggregate variability shrinks with N
    prof = base_profile(hour[None, :], weekend)
    heat_eff = 0.004 * np.abs(temp - 18.0)
    noise = _ar1(rng, (E, T), 0.95, 0.035 * np.sqrt(10.0 / max(N, 10)))
    spikes = (rng.random((E, T)) < 0.004) * rng.uniform(0.04, 0.10, size=(E, T))
    # grid-stress events (e.g. feeder contingency or demand-response window): +8..22 % of R for 2-4 h, on ~30 % of days
    ev = np.zeros((E, T))
    for d in range(days):
        has = rng.random(E) < 0.30
        st = rng.uniform(10, 20, size=E)
        du = rng.uniform(2, 4, size=E)
        mag = rng.uniform(0.08, 0.22, size=E)
        hh = hour[None, :]
        w = np.where((hh >= st[:, None]) & (hh < (st + du)[:, None]), 1.0, 0.0) * (t[None, :] // SPD == d)
        ev += (has * mag)[:, None] * w
    load = R * load_scale * (prof + heat_eff) * (1 + noise) + R * spikes + R * ev
    head = R - load + pv
    return dict(hour=np.tile(hour, (E, 1)), weekend=weekend, temp=temp, cloud=cloud, pv=pv, load=load,
                head=head, tou=tou_price(np.tile(hour, (E, 1)), cfg), R=R, dow=dow)
