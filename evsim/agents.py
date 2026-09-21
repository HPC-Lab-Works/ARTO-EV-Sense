"""Controllers: static, TOU, rule-based (forecast aware) and DRL agents (PPO, DQN, SAC, TD3) implemented in PyTorch."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fnn, time
from .config import SimConfig, ACTION_M_LEVELS, ACTION_N_LEVELS, OBS_DIM, STEPS_PER_DAY
from .env import EVNetEnv, action_to_env

SPD = STEPS_PER_DAY
HID = 128


# =============================================================== non-learned controllers
class Static:
    """Fixed price multiplier m and all chargers enabled (m = 1 is the untuned reference)."""
    def __init__(self, cfg, m=1.0): self.cfg, self.m = cfg, m
    def __call__(self, o):
        return np.full(len(o), float(self.m)), np.full(len(o), self.cfg.C)


def _period(o):
    """tariff period from the observed grid price (o[:,12] = price/0.22): 0 off-peak, 1 mid, 2 peak."""
    return np.where(o[:, 12] > 0.9, 2, np.where(o[:, 12] > 0.55, 1, 0))


class TOU:
    def __init__(self, cfg, m=(0.8, 1.1, 1.4)): self.cfg, self.m = cfg, np.array(m)
    def __call__(self, o):
        return self.m[_period(o)], np.full(len(o), self.cfg.C)


class RuleBased:
    """Forecast-aware heuristic: tariff-period price plus surcharges and charger caps when the forecast margin is low."""
    def __init__(self, cfg, p):
        self.cfg, self.p = cfg, p
    def __call__(self, o):
        p = self.p
        s = o[:, 5] - o[:, 6]                       # forecast min headroom minus present charging load (fraction of R)
        m = np.array([p["m_off"], p["m_mid"], p["m_peak"]])[_period(o)]
        m = m + np.where(s < p["th_hi"], p["dm1"], 0.0) + np.where(s < p["th_lo"], p["dm2"], 0.0)
        n = np.where(s < p["th_lo"], p["n_low"], np.where(s < p["th_hi"], p["n_mid"], self.cfg.C))
        return m, n


def rule_param_sample(rng):
    return dict(m_off=rng.choice([0.6, 0.8, 1.0]), m_mid=rng.choice([0.9, 1.0, 1.1, 1.2]), m_peak=rng.choice([1.1, 1.3, 1.5, 1.7]),
                th_hi=rng.uniform(0.15, 0.40), th_lo=rng.uniform(0.0, 0.15), dm1=rng.choice([0.0, 0.1, 0.2, 0.3]),
                dm2=rng.choice([0.1, 0.2, 0.4, 0.6]), n_mid=int(rng.choice([3, 4, 5])), n_low=int(rng.choice([2, 3])))


# =============================================================== networks
def mlp(i, o, h=HID, act=nn.ReLU):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, h), act(), nn.Linear(h, o))


class Replay:
    def __init__(self, cap, od, ad):
        self.o = np.zeros((cap, od), np.float32); self.a = np.zeros((cap, ad), np.float32)
        self.r = np.zeros(cap, np.float32); self.o2 = np.zeros((cap, od), np.float32); self.d = np.zeros(cap, np.float32)
        self.cap, self.n, self.i = cap, 0, 0
    def add(self, o, a, r, o2, d):
        k = len(o); idx = (self.i + np.arange(k)) % self.cap
        self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx] = o, a, r, o2, d
        self.i = (self.i + k) % self.cap; self.n = min(self.n + k, self.cap)
    def sample(self, bs, rng):
        j = rng.integers(0, self.n, bs)
        t = lambda x: torch.from_numpy(x[j])
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


# =============================================================== policies used at evaluation time
class TorchPolicy:
    """Wraps a trained network; returns (m, nact)."""
    def __init__(self, kind, net, cfg, fixed_m=None, fixed_n=None):
        self.kind, self.net, self.cfg, self.fixed_m, self.fixed_n = kind, net, cfg, fixed_m, fixed_n
    def __call__(self, o):
        with torch.no_grad():
            x = torch.as_tensor(o, dtype=torch.float32)
            if self.kind == "dqn":
                k = self.net(x).argmax(-1).numpy()
                m = np.array(ACTION_M_LEVELS)[k // len(ACTION_N_LEVELS)]
                n = np.array(ACTION_N_LEVELS)[k % len(ACTION_N_LEVELS)]
            else:
                a = self.net(x).numpy()
                m, n = action_to_env(a, self.cfg.C, self.cfg)
        if self.fixed_m is not None: m = np.full(len(o), self.fixed_m)
        if self.fixed_n is not None: n = np.full(len(o), self.fixed_n)
        return m, n


class Actor(nn.Module):
    """Deterministic action (tanh of mean) used at evaluation for PPO / SAC."""
    def __init__(self, body, squash=True):
        super().__init__(); self.body, self.squash = body, squash
    def forward(self, x):
        y = self.body(x)
        return torch.tanh(y) if self.squash else y


class SacDet(nn.Module):
    def __init__(self, pi):
        super().__init__(); self.pi = pi
    def forward(self, x):
        return torch.tanh(self.pi(x).chunk(2, -1)[0])


# =============================================================== training
def make_train_env(cfg, E, forecaster, forecast_mode):
    return EVNetEnv(cfg, E=E, forecast_fn=forecaster, forecast_mode=forecast_mode)


def _act_env(kind, a_or_k, cfg, fixed_m, fixed_n):
    if kind == "dqn":
        k = a_or_k
        m = np.array(ACTION_M_LEVELS)[k // len(ACTION_N_LEVELS)]
        n = np.array(ACTION_N_LEVELS)[k % len(ACTION_N_LEVELS)]
    else:
        m, n = action_to_env(a_or_k, cfg.C, cfg)
    if fixed_m is not None: m = np.full(len(m), fixed_m)
    if fixed_n is not None: n = np.full(len(n), fixed_n)
    return m, n


def train_agent(kind, seed, cfg, forecaster, forecast_mode="lstm", steps=400_000, E=16, lr=3e-4, fixed_m=None, fixed_n=None,
                log=None, updates_per_step=4, warmup=5000, eval_hook=None):
    """Train one DRL agent for `steps` environment transitions. Returns TorchPolicy and training log."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed + 12345)
    env = make_train_env(cfg, E, forecaster, forecast_mode)
    od, ad = OBS_DIM, 2
    nA = len(ACTION_M_LEVELS) * len(ACTION_N_LEVELS)
    gamma = 0.99
    hist = []
    t0 = time.time()
    ep = 0
    base_seed = 1_000_000 + seed * 100_000

    if kind == "ppo":
        pi = mlp(od, ad, act=nn.Tanh); vf = mlp(od, 1, act=nn.Tanh)
        log_std = nn.Parameter(torch.full((ad,), -0.7))
        opt = torch.optim.Adam(list(pi.parameters()) + list(vf.parameters()) + [log_std], lr=lr)
        n_iter = steps // (SPD * E)
        for it in range(n_iter):
            for g in opt.param_groups: g["lr"] = lr * (1 - it / n_iter)
            o = env.reset(base_seed + ep); ep += 1
            O = np.zeros((SPD, E, od), np.float32); A = np.zeros((SPD, E, ad), np.float32); LP = np.zeros((SPD, E), np.float32)
            V = np.zeros((SPD, E), np.float32); Rw = np.zeros((SPD, E), np.float32)
            for t in range(SPD):
                with torch.no_grad():
                    x = torch.as_tensor(o, dtype=torch.float32)
                    mu = pi(x); std = log_std.exp()
                    a = mu + std * torch.randn_like(mu)
                    lp = (-0.5 * ((a - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
                    v = vf(x).squeeze(-1)
                O[t], A[t], LP[t], V[t] = x.numpy(), a.numpy(), lp.numpy(), v.numpy()
                m, n = _act_env("ppo", np.clip(a.numpy(), -1, 1), cfg, fixed_m, fixed_n)
                o, r, d = env.step(m, n)
                Rw[t] = r
            # GAE, terminal at t = SPD (value 0)
            adv = np.zeros((SPD, E), np.float32); last = 0.0
            for t in reversed(range(SPD)):
                nv = V[t + 1] if t + 1 < SPD else 0.0
                delta = Rw[t] + gamma * nv - V[t]
                last = delta + gamma * 0.95 * last
                adv[t] = last
            ret = adv + V
            fl = lambda x: torch.as_tensor(x.reshape(SPD * E, *x.shape[2:]))
            bO, bA, bLP, bAdv, bRet = fl(O), fl(A), fl(LP), fl(adv), fl(ret)
            bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
            nn_ = SPD * E
            for _ in range(10):
                perm = torch.randperm(nn_)
                for i in range(0, nn_, 512):
                    j = perm[i:i + 512]
                    mu = pi(bO[j]); std = log_std.exp()
                    lp = (-0.5 * ((bA[j] - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
                    ratio = (lp - bLP[j]).exp()
                    l_pi = -torch.min(ratio * bAdv[j], ratio.clamp(0.8, 1.2) * bAdv[j]).mean()
                    l_v = 0.5 * (vf(bO[j]).squeeze(-1) - bRet[j]).pow(2).mean()
                    loss = l_pi + 0.5 * l_v
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vf.parameters()), 0.5)
                    opt.step()
            s = env.summary()
            if it % 10 == 0 or it == n_iter - 1:
                hist.append(dict(step=(it + 1) * SPD * E, ret=float(s["reward"].mean()), t=time.time() - t0))
                if log: log(f"ppo s{seed} it{it}/{n_iter} ret {s['reward'].mean():.2f} std {log_std.exp().mean().item():.2f} {time.time()-t0:.0f}s")
        return TorchPolicy("ppo", Actor(pi), cfg, fixed_m, fixed_n), hist

    # ---------------------------------------------------------------- off-policy agents
    cap = steps
    buf = Replay(cap, od, 1 if kind == "dqn" else ad)
    bs = 256
    if kind == "dqn":
        q = mlp(od, nA); q_t = mlp(od, nA); q_t.load_state_dict(q.state_dict())
        opt = torch.optim.Adam(q.parameters(), lr=lr)
    elif kind == "sac":
        pi = mlp(od, 2 * ad); q1, q2, q1t, q2t = mlp(od + ad, 1), mlp(od + ad, 1), mlp(od + ad, 1), mlp(od + ad, 1)
        q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
        log_alpha = torch.zeros(1, requires_grad=True)
        opt_pi = torch.optim.Adam(pi.parameters(), lr=lr)
        opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
        opt_a = torch.optim.Adam([log_alpha], lr=lr)
        def sample(x, det=False):
            mu, ls = pi(x).chunk(2, -1); ls = ls.clamp(-5, 2); std = ls.exp()
            u = mu if det else mu + std * torch.randn_like(mu)
            a = torch.tanh(u)
            lp = (-0.5 * ((u - mu) / std) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1) - (2 * (np.log(2) - u - Fnn.softplus(-2 * u))).sum(-1)
            return a, lp
    elif kind == "td3":
        pi = mlp(od, ad); pi_t = mlp(od, ad); pi_t.load_state_dict(pi.state_dict())
        q1, q2, q1t, q2t = mlp(od + ad, 1), mlp(od + ad, 1), mlp(od + ad, 1), mlp(od + ad, 1)
        q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
        opt_pi = torch.optim.Adam(pi.parameters(), lr=lr)
        opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    soft = lambda src, dst: [dp.data.mul_(0.995).add_(0.005 * sp.data) for sp, dp in zip(src.parameters(), dst.parameters())]

    done_steps, n_upd = 0, 0
    o = env.reset(base_seed + ep); ep += 1
    t_in_ep = 0
    while done_steps < steps:
        # ---- act
        if done_steps < warmup:
            a = rng.uniform(-1, 1, (E, ad)).astype(np.float32) if kind != "dqn" else rng.integers(0, nA, (E, 1))
        else:
            with torch.no_grad():
                x = torch.as_tensor(o, dtype=torch.float32)
                if kind == "dqn":
                    eps = max(0.05, 1.0 - done_steps / (0.3 * steps))
                    k = q(x).argmax(-1).numpy()
                    rnd = rng.random(E) < eps
                    k = np.where(rnd, rng.integers(0, nA, E), k)
                    a = k[:, None]
                elif kind == "sac":
                    a = sample(x)[0].numpy()
                else:
                    a = np.clip(torch.tanh(pi(x)).numpy() + 0.1 * rng.standard_normal((E, ad)), -1, 1)
        m, n = _act_env(kind, a[:, 0] if kind == "dqn" else a, cfg, fixed_m, fixed_n)
        o2, r, d = env.step(m, n)
        buf.add(o.astype(np.float32), np.asarray(a, np.float32), r.astype(np.float32), o2.astype(np.float32), np.full(E, float(d), np.float32))
        o = o2; done_steps += E; t_in_ep += 1
        if d:
            s = env.summary()
            if ep % 20 == 0:
                hist.append(dict(step=done_steps, ret=float(s["reward"].mean()), t=time.time() - t0))
                if log: log(f"{kind} s{seed} ep{ep} step {done_steps}/{steps} ret {s['reward'].mean():.2f} {time.time()-t0:.0f}s")
            o = env.reset(base_seed + ep); ep += 1; t_in_ep = 0
        # ---- learn
        if done_steps >= warmup:
            for _ in range(updates_per_step):
                ob, ab, rb, o2b, db = buf.sample(bs, rng)
                if kind == "dqn":
                    with torch.no_grad():
                        a2 = q(o2b).argmax(-1, keepdim=True)
                        y = rb + gamma * (1 - db) * q_t(o2b).gather(1, a2).squeeze(1)
                    qv = q(ob).gather(1, ab.long()).squeeze(1)
                    loss = Fnn.smooth_l1_loss(qv, y)
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
                    soft(q, q_t)
                elif kind == "sac":
                    alpha = log_alpha.exp().detach()
                    with torch.no_grad():
                        a2, lp2 = sample(o2b)
                        x2 = torch.cat([o2b, a2], 1)
                        y = rb + gamma * (1 - db) * (torch.min(q1t(x2), q2t(x2)).squeeze(1) - alpha * lp2)
                    xb = torch.cat([ob, ab], 1)
                    lq = Fnn.mse_loss(q1(xb).squeeze(1), y) + Fnn.mse_loss(q2(xb).squeeze(1), y)
                    opt_q.zero_grad(); lq.backward(); opt_q.step()
                    an, lp = sample(ob); xn = torch.cat([ob, an], 1)
                    lpi = (alpha * lp - torch.min(q1(xn), q2(xn)).squeeze(1)).mean()
                    opt_pi.zero_grad(); lpi.backward(); opt_pi.step()
                    la = -(log_alpha * (lp.detach() - 2.0).mean())
                    opt_a.zero_grad(); la.backward(); opt_a.step()
                    soft(q1, q1t); soft(q2, q2t)
                else:  # td3
                    with torch.no_grad():
                        a2 = (torch.tanh(pi_t(o2b)) + (0.2 * torch.randn_like(ab)).clamp(-0.5, 0.5)).clamp(-1, 1)
                        x2 = torch.cat([o2b, a2], 1)
                        y = rb + gamma * (1 - db) * torch.min(q1t(x2), q2t(x2)).squeeze(1)
                    xb = torch.cat([ob, ab], 1)
                    lq = Fnn.mse_loss(q1(xb).squeeze(1), y) + Fnn.mse_loss(q2(xb).squeeze(1), y)
                    opt_q.zero_grad(); lq.backward(); opt_q.step()
                    n_upd += 1
                    if n_upd % 2 == 0:
                        lpi = -q1(torch.cat([ob, torch.tanh(pi(ob))], 1)).mean()
                        opt_pi.zero_grad(); lpi.backward(); opt_pi.step()
                        soft(pi, pi_t); soft(q1, q1t); soft(q2, q2t)
    if kind == "dqn":
        return TorchPolicy("dqn", q, cfg, fixed_m, fixed_n), hist
    if kind == "sac":
        return TorchPolicy("sac", SacDet(pi), cfg, fixed_m, fixed_n), hist
    return TorchPolicy("td3", Actor(pi), cfg, fixed_m, fixed_n), hist


# =============================================================== evaluation
def evaluate(policy, cfg, eval_seed, forecaster=None, forecast_mode="lstm", E=30, obs_hook=None, **reset_kw):
    """Run E held-out days with common random numbers. Returns per-day metric arrays."""
    env = EVNetEnv(cfg, E=E, forecast_fn=forecaster, forecast_mode=forecast_mode)
    o = env.reset(eval_seed, **reset_kw)
    for t in range(SPD):
        m, n = policy(o)
        o, r, d = env.step(m, n)
    return env.summary()
