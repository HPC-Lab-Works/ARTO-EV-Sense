"""Global constants of the ARTO-EV synthetic testbench.

All numbers are ASSUMPTIONS chosen to be in plausible ranges (see README); they are not fitted to any
real charging log. They are collected here so that every experiment uses a single source of truth.
"""
from dataclasses import dataclass, field

STEPS_PER_DAY = 96          # 15-minute slots
DT_H = 0.25                 # slot length in hours


@dataclass
class SimConfig:
    # topology
    N: int = 10                    # stations on the feeder
    C: int = 6                     # chargers (ports) per station
    Q: int = 6                     # waiting spots per station
    pmax_kw: float = 22.0          # charger power limit
    # feeder / grid (per station shares, so that R scales with N)
    R_per_station: float = 100.0   # transformer rating share [kW]
    pv_per_station: float = 25.0   # PV peak share [kW]
    overload_tol: float = 0.02     # overload event if excess > tol * R
    # tariff (grid purchase price, USD/kWh) and retail reference
    tou_offpeak: float = 0.10
    tou_mid: float = 0.14
    tou_peak: float = 0.22
    p_ref: float = 0.30            # retail reference price at m = 1
    tier: tuple = (0.9, 1.0, 1.15)  # price tier of residential, commercial, highway
    # price response (logit on the price multiplier m)
    x_out: float = 1.15
    w_price: tuple = (0.20, 0.25, 0.30)
    # demand scale (calibration knob) and day-to-day noise
    demand_scale: float = 1.9
    demand_day_sigma: float = 0.15
    amax: int = 6                  # max arrivals per station per slot (binomial thinning)
    # vehicles
    e_mean: float = 16.0
    e_sd: float = 8.0
    e_min: float = 4.0
    e_max: float = 50.0
    v_pmax: tuple = (7.4, 11.0, 22.0)
    v_pmax_prob: tuple = (0.3, 0.4, 0.3)
    patience_slots: tuple = (2, 6)   # uniform integer range, inclusive (30-90 min)
    # reward (USD)
    w_serve: float = 0.25          # service value per kWh delivered (USD/kWh)
    w_wait: float = 6.0            # per waiting vehicle-hour
    w_over: float = 2.0            # per overload kWh
    r_scale: float = 2.0           # USD per station per slot used to scale reward to ~O(1)
    # action ranges
    m_lo: float = 0.6
    m_hi: float = 1.8
    forecast_horizon: int = 4


ACTION_M_LEVELS = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
ACTION_N_LEVELS = (2, 3, 4, 6)
OBS_DIM = 16
