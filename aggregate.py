"""Aggregate raw result files into LaTeX tables, figures and a machine-readable summary (all numbers in the paper come from here)."""
import json, glob, os, numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

SEEDS = list(range(10))
SUM = {}


def ms(x, d=2, pct=False):
    x = np.asarray(x, float) * (100 if pct else 1)
    return f"{x.mean():.{d}f} $\\pm$ {x.std(ddof=1):.{d}f}"


def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k]); adj[k] = min(run, 1.0)
    return adj


def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(pat))]


def seedvals(res_list, ctrl, metric, agg=np.mean):
    return np.array([agg(r[ctrl][metric]) for r in res_list])


def write(name, s):
    open(f"tables/{name}.tex", "w").write(s)


# ------------------------------------------------------------------ forecasting
def forecasting():
    r = json.load(open("results/forecast.json"))
    names = [("persistence", "Persistence"), ("seasonal_naive", "Seasonal naive"), ("ridge", "Ridge autoregression"), ("mlp", "MLP"), ("lstm", "LSTM (ARTO-EV)")]
    rows = []
    for k, n in names:
        g = lambda m: [x[m] for x in r[k]]
        rows.append(f"{n} & {ms(g('mae_h1'),1)} & {ms(g('mae_h4'),1)} & {ms(g('rmse_all'),1)} & {ms(g('mape_all'),2)} \\\\")
        SUM.setdefault("forecast", {})[k] = {m: [float(np.mean(g(m))), float(np.std(g(m), ddof=1))] for m in ("mae_h1", "mae_h4", "rmse_all", "mape_all")}
    a = np.array([x["mae_all"] for x in r["lstm"]]); b = np.array([x["mae_all"] for x in r["mlp"]])
    SUM["forecast"]["lstm_vs_mlp_p"] = float(stats.mannwhitneyu(a, b).pvalue)
    a = np.array([x["mae_all"] for x in r["lstm"]]); b = np.array([x["mae_all"] for x in r["persistence"]])
    SUM["forecast"]["lstm_vs_persist_p"] = float(stats.mannwhitneyu(a, b).pvalue)
    write("tab_forecast", "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nModel & MAE, 15 min (kW) & MAE, 60 min (kW) & RMSE, all horizons (kW) & MAPE (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


# ------------------------------------------------------------------ control
CT = ["Static", "Best fixed price", "TOU", "Rule-based", "DQN", "SAC", "TD3", "PPO (ARTO-EV)"]
CM = [("reward", "Reward", 1, False, True), ("profit", "Profit (USD)", 0, False, True), ("wait_min", "Wait (min)", 1, False, False),
      ("utilization", "Utilization (\\%)", 1, True, True), ("over_events", "Overload events", 2, False, False),
      ("over_kwh", "Overload energy (kWh)", 1, False, False), ("served_frac", "Served (\\%)", 1, True, True)]


def control():
    R = load("results/control/main_s*.json")
    if not R: return
    tab = {}
    for c in CT:
        tab[c] = {m[0]: seedvals(R, c, m[0]) for m in CM}
    SUM["control"] = {c: {m: [float(v.mean()), float(v.std(ddof=1))] for m, v in d.items()} for c, d in tab.items()}
    # significance of PPO against each other controller (Mann-Whitney U over seed-level means, Holm within metric)
    sig = {}
    for m, _, _, _, hib in CM:
        ps = [stats.mannwhitneyu(tab["PPO (ARTO-EV)"][m], tab[c][m], alternative="two-sided").pvalue for c in CT[:-1]]
        adj = holm(ps)
        sig[m] = {c: dict(p=float(ps[i]), p_holm=float(adj[i]), cliff=cliffs(tab["PPO (ARTO-EV)"][m], tab[c][m])) for i, c in enumerate(CT[:-1])}
    SUM["control_sig"] = sig
    lines = []
    for c in CT:
        cells = []
        for m, _, d, pct, hib in CM:
            v = tab[c][m]; cell = ms(v, d, pct)
            best = (max if hib else min)(CT, key=lambda k: tab[k][m].mean())
            if c == best: cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(f"{c} & " + " & ".join(cells) + " \\\\")
    hdr = " & ".join(m[1] for m in CM)
    write("tab_control", "\\begin{tabular}{@{}l" + "c" * len(CM) + "@{}}\n\\toprule\nController & " + hdr + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    # training curves
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for name in ["PPO (ARTO-EV)", "DQN", "SAC", "TD3"]:
        H = [r["_hist"][name] for r in R]
        L = min(len(h) for h in H)
        x = np.array([h[i]["step"] for i in range(L) for h in H[:1]])
        y = np.array([[h[i]["ret"] for i in range(L)] for h in H])
        ax.plot(x / 1e3, y.mean(0), label=name); ax.fill_between(x / 1e3, y.mean(0) - y.std(0), y.mean(0) + y.std(0), alpha=0.2)
    for c, ls in [("Rule-based", "--"), ("TOU", ":")]:
        ax.axhline(tab[c]["reward"].mean(), color="k", ls=ls, lw=0.8, label=c)
    ax.set_xlabel("Training environment steps (thousands)"); ax.set_ylabel("Training episode reward (per day)"); ax.legend(fontsize=6, frameon=False)
    ax.set_ylim(bottom=max(ax.get_ylim()[0], 0))
    fig.tight_layout(); fig.savefig("figs/fig_training.pdf"); plt.close(fig)
    # training time
    SUM["train_time_s"] = {n: float(np.mean([r[n + "_train_s"] for r in R])) for n in CT[4:]}
    SUM["heuristic_params"] = [r["_params"] for r in R]


def ablation():
    R = load("results/control/ablate_s*.json"); M = load("results/control/main_s*.json")
    if not R or not M: return
    n = len(R)
    names = list(R[0].keys())
    rows = []
    full = {m[0]: np.array([np.mean(M[i]["PPO (ARTO-EV)"][m[0]]) for i in range(len(R))]) for m in CM}
    def row(label, d):
        return f"{label} & " + " & ".join(ms(d[m], dd, pct) for m, _, dd, pct, _ in [(x[0], x[1], x[2], x[3], x[4]) for x in [CM[0], CM[1], CM[2], CM[3], CM[4]]]) + " \\\\"
    rows.append(row("Full ARTO-EV (LSTM forecast)", full))
    SUM["ablation"] = {"full": {m: [float(full[m].mean()), float(full[m].std(ddof=1))] for m in full}}
    for nm in names:
        d = {m[0]: np.array([np.mean(r[nm][m[0]]) for r in R]) for m in CM}
        rows.append(row(nm.replace("PPO, ", "").capitalize().replace("Lstm", "LSTM"), d))
        SUM["ablation"][nm] = {m: [float(d[m].mean()), float(d[m].std(ddof=1)), float(stats.mannwhitneyu(d[m], full[m]).pvalue)] for m in d}
    hdr = " & ".join(x[1] for x in CM[:5])
    write("tab_ablation", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nVariant & " + hdr + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


# ------------------------------------------------------------------ security
DN = ["Rule-based IDS", "Isolation forest", "One-class SVM", "LSTM autoencoder", "Autoencoder (ARTO-EV)"]


def security():
    R = load("results/security/sec_s*.json")
    if not R: return
    ren = {"Signature IDS": "Rule-based IDS"}
    rr = [{ren.get(k, k): v for k, v in r.items()} for r in R]
    SUM["security"] = {"prevalence": float(np.mean([r["prevalence"] for r in rr])),
                       "attack_share": {k: float(np.mean([r["attack_share"][k] for r in rr])) for k in ("FDI", "MITM", "DDoS")}}
    lines1, lines2 = [], []
    for d in DN:
        g = lambda m: [r[d][m] for r in rr]
        SUM["security"][d] = {m: [float(np.nanmean(g(m))), float(np.nanstd(g(m), ddof=1))] for m in rr[0][d]}
        lines1.append(f"{d} & {ms(g('accuracy'),1,True)} & {ms(g('precision'),1,True)} & {ms(g('recall'),1,True)} & {ms(g('f1'),1,True)} & {ms(g('fpr'),1,True)} & {ms(g('auroc'),3)} \\\\")
        lines2.append(f"{d} & {ms(g('recall_FDI'),1,True)} & {ms(g('recall_FDI_nonstealth'),1,True)} & {ms(g('recall_FDI_stealth'),1,True)} & {ms(g('recall_MITM'),1,True)} & {ms(g('recall_DDoS'),1,True)} & {ms(g('segment_detection'),1,True)} & {ms(g('median_latency_slots'),1)} \\\\")
    write("tab_detect", "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nDetector & Accuracy (\\%) & Precision (\\%) & Recall (\\%) & F1 (\\%) & FPR (\\%) & AUROC \\\\\n\\midrule\n" + "\n".join(lines1) + "\n\\bottomrule\n\\end{tabular}")
    write("tab_detect_type", "\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\nDetector & FDI (all) & FDI, non-stealthy & FDI, stealthy & MITM & DDoS & Episodes flagged (\\%) & Median delay (slots) \\\\\n\\midrule\n" + "\n".join(lines2) + "\n\\bottomrule\n\\end{tabular}")
    ae = np.array([r["Autoencoder (ARTO-EV)"]["f1"] for r in rr])
    SUM["security"]["pvals_f1_ae_vs"] = {d: float(stats.mannwhitneyu(ae, [r[d]["f1"] for r in rr]).pvalue) for d in DN[:-1]}
    SUM["security"]["ae_op"] = {q: {k: float(np.mean([r["ae_operating_points"][q][k] for r in rr])) for k in ("fpr", "recall")} for q in ("0.95", "0.99", "0.999")}
    SUM["security"]["timing"] = {d: {k: float(np.mean([r[d][k] for r in rr])) for k in ("train_s", "infer_us_per_window")} for d in DN}
    # figure: recall by attack type
    fig, ax = plt.subplots(figsize=(3.6, 2.3)); w = 0.16
    for i, d in enumerate(DN):
        v = [np.mean([r[d][f"recall_{a}"] for r in rr]) * 100 for a in ("FDI", "MITM", "DDoS")]
        e = [np.std([r[d][f"recall_{a}"] for r in rr], ddof=1) * 100 for a in ("FDI", "MITM", "DDoS")]
        ax.bar(np.arange(3) + (i - 2) * w, v, w, yerr=e, label=d, capsize=1.5, error_kw={"lw": 0.6})
    ax.set_xticks(range(3)); ax.set_xticklabels(["FDI", "MITM", "DDoS"]); ax.set_ylabel("Recall at 1% benign FPR (%)"); ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("figs/fig_detect.pdf"); plt.close(fig)


def fed():
    R = load("results/security/fed_s*.json")
    if not R: return
    lines = []
    SUM["fed"] = {}
    for k in ("Local only", "FedAvg", "Centralised"):
        g = lambda m: [r[k][m] for r in R]
        SUM["fed"][k] = {m: [float(np.mean(g(m))), float(np.std(g(m), ddof=1))] for m in ("f1", "recall", "fpr", "auroc", "comm_bytes")}
        lines.append(f"{k} & {ms(g('f1'),1,True)} & {ms(g('recall'),1,True)} & {ms(g('fpr'),1,True)} & {ms(g('auroc'),3)} & {np.mean(g('comm_bytes'))/1e6:.2f} \\\\")
    SUM["fed"]["n_params"] = R[0]["n_params"]
    SUM["fed"]["p_fedavg_vs_local"] = float(stats.mannwhitneyu([r["FedAvg"]["f1"] for r in R], [r["Local only"]["f1"] for r in R]).pvalue)
    SUM["fed"]["p_fedavg_vs_central"] = float(stats.mannwhitneyu([r["FedAvg"]["f1"] for r in R], [r["Centralised"]["f1"] for r in R]).pvalue)
    write("tab_fed", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nTraining scheme & F1 (\\%) & Recall (\\%) & FPR (\\%) & AUROC & Data transferred (MB) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")


def attack():
    R = load("results/control/attack_s*.json")
    if not R: return
    scen = ["none", "FDI", "FDI-mask", "MITM", "DDoS"]
    keys = [("Static", None), ("Best fixed price", None), ("TOU", None), ("Rule-based", None), ("DQN", None), ("SAC", None), ("TD3", None), ("PPO", None),
            ("Rule-based", "Autoencoder (ARTO-EV)"), ("PPO", "Rule-based IDS"), ("PPO", "Isolation forest"), ("PPO", "One-class SVM"), ("PPO", "LSTM autoencoder"), ("PPO", "Autoencoder (ARTO-EV)")]
    sigmap = {"Signature IDS": "Rule-based IDS"}
    def get(r, s, c, d, m):
        dd = sigmap.get(d, d) if False else d
        kk = f"{s}|{c}|{('Signature IDS' if d == 'Rule-based IDS' else d)}"
        return np.mean(r[kk]["m"][m])
    labels = {("Rule-based", "Autoencoder (ARTO-EV)"): "Rule-based + AE screening", ("PPO", "Rule-based IDS"): "PPO + rule-based IDS", ("PPO", "Isolation forest"): "PPO + isolation forest",
              ("PPO", "One-class SVM"): "PPO + one-class SVM", ("PPO", "LSTM autoencoder"): "PPO + LSTM autoencoder", ("PPO", "Autoencoder (ARTO-EV)"): "PPO + AE screening (ARTO-EV)"}
    SUM["attack"] = {}
    lo, lr = [], []
    for c, d in keys:
        lab = labels.get((c, d), c if c != "PPO" else "PPO, no screening")
        co, cr = [], []
        for s in scen:
            ov = np.array([get(r, s, c, d, "over_events") for r in R]); rw = np.array([get(r, s, c, d, "reward") for r in R])
            ok = np.array([get(r, s, c, d, "over_kwh") for r in R]); wt = np.array([get(r, s, c, d, "wait_min") for r in R])
            SUM["attack"][f"{s}|{lab}"] = dict(over=[float(ov.mean()), float(ov.std(ddof=1))], reward=[float(rw.mean()), float(rw.std(ddof=1))],
                                               over_kwh=[float(ok.mean()), float(ok.std(ddof=1))], wait=[float(wt.mean()), float(wt.std(ddof=1))])
            co.append(ms(ov, 2)); cr.append(ms(rw, 1))
        lo.append(f"{lab} & " + " & ".join(co) + " \\\\"); lr.append(f"{lab} & " + " & ".join(cr) + " \\\\")
    lo.insert(8, "\\midrule"); lr.insert(8, "\\midrule")
    hdr = "Controller & No attack & FDI (40\\% of stations) & FDI, load masking (all stations) & MITM (40\\%) & DDoS (40\\%)"
    write("tab_attack", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n" + hdr + " \\\\\n\\midrule\n" + "\n".join(lo) + "\n\\bottomrule\n\\end{tabular}")
    write("tab_attack_reward", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n" + hdr + " \\\\\n\\midrule\n" + "\n".join(lr) + "\n\\bottomrule\n\\end{tabular}")
    # closed-loop detector stats for AE
    ae = {}
    for s in scen[1:]:
        tp = sum(r[f"{s}|PPO|Autoencoder (ARTO-EV)"]["det"]["tp"] for r in R); fn = sum(r[f"{s}|PPO|Autoencoder (ARTO-EV)"]["det"]["fn"] for r in R)
        fp = sum(r[f"{s}|PPO|Autoencoder (ARTO-EV)"]["det"]["fp"] for r in R); tn = sum(r[f"{s}|PPO|Autoencoder (ARTO-EV)"]["det"]["tn"] for r in R)
        ae[s] = dict(recall=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1))
    SUM["attack_ae_detection"] = ae
    # significance: PPO no screening vs PPO + AE per scenario (overload events)
    SUM["attack_sig"] = {}
    for s in scen:
        a = [get(r, s, "PPO", None, "over_events") for r in R]; b = [get(r, s, "PPO", "Autoencoder (ARTO-EV)", "over_events") for r in R]
        a2 = [get(r, s, "PPO", None, "reward") for r in R]; b2 = [get(r, s, "PPO", "Autoencoder (ARTO-EV)", "reward") for r in R]
        SUM["attack_sig"][s] = dict(p_over=float(stats.mannwhitneyu(a, b).pvalue), p_reward=float(stats.mannwhitneyu(a2, b2).pvalue))
    fig, ax = plt.subplots(figsize=(3.6, 2.3)); w = 0.25
    for i, (lab, c, d) in enumerate([("PPO, no screening", "PPO", None), ("PPO + AE screening", "PPO", "Autoencoder (ARTO-EV)"), ("Rule-based, no screening", "Rule-based", None)]):
        v = [np.mean([get(r, s, c, d, "over_events") for r in R]) for s in scen]
        e = [np.std([get(r, s, c, d, "over_events") for r in R], ddof=1) for s in scen]
        ax.bar(np.arange(5) + (i - 1) * w, v, w, yerr=e, label=lab, capsize=1.5, error_kw={"lw": 0.6})
    ax.set_xticks(range(5)); ax.set_xticklabels(["None", "FDI", "FDI-mask", "MITM", "DDoS"]); ax.set_ylabel("Overload events per day"); ax.set_ylim(0, 1.9); ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig("figs/fig_attack.pdf"); plt.close(fig)


def ledger():
    if not os.path.exists("results/ledger/ledger.json"): return
    L = json.load(open("results/ledger/ledger.json"))
    P = L["perf"]
    lines = []
    SUM["ledger"] = {}
    for nv in (4, 7, 10):
        for bs in (10, 50, 100):
            k = f"nv{nv}_bs{bs}"
            g = lambda m: [p[k][m] for p in P]
            lines.append(f"{nv} & {bs} & {ms(g('commit_ms_mean'),1)} & {ms(g('commit_ms_p95'),1)} & {np.mean(g('throughput_tps')):.0f} & {np.mean(g('bytes_per_tx')):.0f} \\\\")
            SUM["ledger"][k] = {m: float(np.mean(g(m))) for m in ("commit_ms_mean", "commit_ms_p95", "throughput_tps", "sign_us", "verify_us", "bytes_per_tx")}
    write("tab_ledger", "\\begin{tabular}{@{}cccccc@{}}\n\\toprule\nValidators & Block size (tx) & Commit latency, mean (ms) & Commit latency, p95 (ms) & Throughput (tx/s) & Storage (bytes/tx) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    T = L["tamper"]
    SUM["ledger"]["tamper"] = {k: float(np.mean([t[k] for t in T])) for k in T[0]}
    SUM["ledger"]["tamper_trials"] = 100 * len(T)
    if L["inloop"]:
        I = L["inloop"]
        SUM["ledger"]["inloop"] = {k: float(np.mean([i[k] for i in I])) for k in I[0] if k != "audit_ok"}
        SUM["ledger"]["inloop"]["audit_ok_all"] = all(i["audit_ok"] for i in I)


def edge():
    if not os.path.exists("results/edge.json"): return
    E = json.load(open("results/edge.json"))
    names = [k for k in E[0] if isinstance(E[0][k], dict)]
    lines = []; SUM["edge"] = {}
    for n in names:
        g = lambda m: [e[n][m] for e in E]
        lines.append(f"{n} & {E[0][n]['params']:,} & {np.mean(g('size_kb')):.1f} & {np.mean(g('lat_p50_ms')):.3f} & {np.mean(g('lat_p95_ms')):.3f} & {np.mean(g('q_size_kb')):.1f} & {np.mean(g('q_lat_p50_ms')):.3f} \\\\")
        SUM["edge"][n] = {m: float(np.mean(g(m))) for m in ("size_kb", "lat_p50_ms", "lat_p95_ms", "q_size_kb", "q_lat_p50_ms")}; SUM["edge"][n]["params"] = E[0][n]["params"]
    SUM["edge"]["ae_f1_fp32"] = float(np.mean([e["ae_f1_fp32"] for e in E])); SUM["edge"]["ae_f1_int8"] = float(np.mean([e["ae_f1_int8"] for e in E]))
    SUM["edge"]["pipeline_p50_ms"] = float(np.mean([e["pipeline_p50_ms"] for e in E])); SUM["edge"]["pipeline_p95_ms"] = float(np.mean([e["pipeline_p95_ms"] for e in E]))
    write("tab_edge", "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n & & \\multicolumn{3}{c}{FP32} & \\multicolumn{2}{c}{INT8 (dynamic)} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\nModel & Parameters & Size (kB) & p50 (ms) & p95 (ms) & Size (kB) & p50 (ms) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")


def scale():
    R = load("results/control/scale_s*.json")
    if not R: return
    Ns = ["10", "50", "100", "500"]
    lines = []; SUM["scale"] = {}
    for N in Ns:
        for c in ("Rule-based", "PPO"):
            rw = np.array([np.mean(r["scale"][N][c]["reward"]) for r in R])
            wt = np.array([np.mean(r["scale"][N][c]["wait_min"]) for r in R]); ov = np.array([np.mean(r["scale"][N][c]["over_events"]) for r in R])
            ut = np.array([np.mean(r["scale"][N][c]["utilization"]) for r in R]) * 100
            SUM["scale"][f"{N}|{c}"] = dict(reward_per_station=[float(rw.mean()), float(rw.std(ddof=1))], wait=[float(wt.mean()), float(wt.std(ddof=1))], over=[float(ov.mean()), float(ov.std(ddof=1))])
            lines.append(f"{N} & {c if c != 'PPO' else 'PPO (ARTO-EV)'} & {ms(rw,1)} & {ms(wt,1)} & {ms(ut,1)} & {ms(ov,2)} \\\\")
        SUM["scale"][f"{N}|env_step_ms"] = float(np.mean([r["scale"][N]["env_step_ms"] for r in R]))
        SUM["scale"][f"{N}|rss_mb"] = float(np.mean([r["scale"][N]["rss_mb"] for r in R]))
        SUM["scale"][f"{N}|wall_s_per_day_ppo"] = float(np.mean([r["scale"][N]["PPO"]["wall_s_per_day"] for r in R]))
        lines.append("\\addlinespace[2pt]")
    write("tab_scale", "\\begin{tabular}{@{}rlcccc@{}}\n\\toprule\nStations & Controller & Reward per day & Wait (min) & Utilization (\\%) & Overload events \\\\\n\\midrule\n" + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}")
    sh = list(R[0]["shift"].keys()); cs = ["Static", "Best fixed price", "TOU", "Rule-based", "DQN", "SAC", "TD3", "PPO"]
    lines = []; SUM["shift"] = {}
    for s in sh:
        cells = []
        for c in cs:
            v = np.array([np.mean(r["shift"][s][c]["reward"]) for r in R])
            SUM["shift"][f"{s}|{c}"] = [float(v.mean()), float(v.std(ddof=1)), float(np.mean([np.mean(r["shift"][s][c]["over_events"]) for r in R]))]
            cells.append(f"{v.mean():.1f}")
        best = int(np.argmax([SUM["shift"][f"{s}|{c}"][0] for c in cs]))
        cells[best] = "\\textbf{" + cells[best] + "}"
        label = s.replace("%", "\\%").replace("+8 C", "+8~\\textdegree{}C")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    write("tab_shift", "\\begin{tabular}{@{}l" + "c" * len(cs) + "@{}}\n\\toprule\nScenario & " + " & ".join(c if c != "PPO" else "PPO (ARTO-EV)" for c in cs) + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")


def profile_fig():
    import torch
    from evsim.config import SimConfig
    from evsim.agents import TorchPolicy, Static
    from evsim.forecast import load_forecaster
    from evsim.env import EVNetEnv
    cfg = SimConfig(); fc = load_forecaster(0, "results/models", cfg)
    d = torch.load("results/models/ppo_s0.pt", weights_only=False); pol = TorchPolicy(d["kind"], d["net"].eval(), cfg)
    out = {}
    for name, ctrl in [("Static", Static(cfg)), ("PPO (ARTO-EV)", pol)]:
        env = EVNetEnv(cfg, E=30, forecast_fn=fc, forecast_mode="lstm"); o = env.reset(50000, record=True)
        ms_, ns_ = [], []
        for t in range(96):
            m, n = ctrl(o); ms_.append(m.copy()); ns_.append(np.asarray(n, float).copy()); o, _, _ = env.step(m, n)
        out[name] = dict(P=np.array(env.trace["Ptot"]), m=np.array(ms_), n=np.array(ns_), head=env.head.copy(), S=env.summary())
    k = int(np.argmax(out["Static"]["S"]["over_events"]))
    hours = np.arange(96) / 4
    fig, ax = plt.subplots(3, 1, figsize=(3.6, 4.6), sharex=True)
    ax[0].plot(hours, out["Static"]["head"][k] / 1e3, "k--", lw=0.9, label="Headroom $G_t$")
    for name in out:
        ax[0].plot(hours, out[name]["P"][:, k] / 1e3, lw=1.0, label=name)
    ax[0].set_ylabel("Power (MW)"); ax[0].legend(fontsize=6, frameon=False, loc="lower left")
    ax[1].step(hours, out["PPO (ARTO-EV)"]["m"][:, k], where="post", color="C1"); ax[1].axhline(1.0, color="C0", lw=0.8); ax[1].set_ylabel("Price multiplier $m_t$")
    ax[2].step(hours, out["PPO (ARTO-EV)"]["n"][:, k], where="post", color="C1"); ax[2].axhline(6, color="C0", lw=0.8); ax[2].set_ylabel("Enabled chargers $n_t$"); ax[2].set_xlabel("Hour of day")
    fig.tight_layout(); fig.savefig("figs/fig_profile.pdf"); plt.close(fig)
    SUM["profile_day"] = dict(static_over=float(out["Static"]["S"]["over_events"][k]), ppo_over=float(out["PPO (ARTO-EV)"]["S"]["over_events"][k]),
                              static_wait=float(out["Static"]["S"]["wait_min"][k]), ppo_wait=float(out["PPO (ARTO-EV)"]["S"]["wait_min"][k]))


# ====================================================================================== sensitivity (retrained)
def sens():
    files = sorted(glob.glob("results/sens/sens_s*.json"))
    if not files: return
    S = {int(f.split("_s")[-1].split(".")[0]): json.load(open(f)) for f in files}
    order = ["demand x0.75", "demand x1.3", "steeper price response", "flatter price response", "wait weight 3", "wait weight 12",
             "overload weight 1", "overload weight 5"]
    seeds = sorted(S)
    done = [c for c in order if all(c in S[sd] for sd in seeds)]
    label = {"demand x0.75": "Demand $\\times 0.75$", "demand x1.3": "Demand $\\times 1.3$", "steeper price response": "Steeper price response ($w\\times 0.6$)",
             "flatter price response": "Flatter price response ($w\\times 1.6$)", "wait weight 3": "Waiting weight 3 (nominal 6)",
             "wait weight 12": "Waiting weight 12", "overload weight 1": "Overload weight 1 (nominal 2)", "overload weight 5": "Overload weight 5"}
    ctrls = ["Static", "Best fixed price", "TOU", "Rule-based", "PPO", "TD3"]
    M = {}   # config -> ctrl -> metric -> per-seed means
    nom = {sd: json.load(open(f"results/control/main_s{sd}.json")) for sd in seeds}
    M["nominal"] = {c: {m: np.array([np.mean(nom[sd]["PPO (ARTO-EV)" if c == "PPO" else c][m]) for sd in seeds]) for m in ("reward", "wait_min", "over_events", "utilization")} for c in ctrls}
    for cn in done:
        M[cn] = {c: {m: np.array([np.mean(S[sd][cn][c][m]) for sd in seeds]) for m in ("reward", "wait_min", "over_events", "utilization")} for c in ctrls}
    rows, rows2, summ = [], [], {}
    for cn in ["nominal"] + done:
        R = {c: M[cn][c]["reward"] for c in ctrls}
        bh = np.max([R["Best fixed price"], R["TOU"], R["Rule-based"]], axis=0)
        ppo_vs = 100 * (R["PPO"] - bh) / bh; td3_vs = 100 * (R["TD3"] - R["PPO"]) / R["PPO"]
        summ[cn] = dict(reward={c: [float(R[c].mean()), float(R[c].std(ddof=1))] for c in ctrls}, ppo_vs_best_heur_pct=[float(ppo_vs.mean()), float(ppo_vs.std(ddof=1))],
                        td3_vs_ppo_pct=[float(td3_vs.mean()), float(td3_vs.std(ddof=1))], n_ppo_above=int((ppo_vs > 0).sum()), n_td3_above=int((td3_vs > 0).sum()),
                        ppo_vs_pts=[float(x) for x in ppo_vs], td3_vs_pts=[float(x) for x in td3_vs],
                        wait={c: float(M[cn][c]["wait_min"].mean()) for c in ctrls}, over={c: float(M[cn][c]["over_events"].mean()) for c in ctrls})
        nm = "Nominal (same seeds)" if cn == "nominal" else label[cn]
        rows.append(f"{nm} & " + " & ".join(f"{R[c].mean():.1f}" for c in ctrls) + f" & {ppo_vs.mean():+.1f} ({int((ppo_vs>0).sum())}/{len(seeds)}) & {td3_vs.mean():+.1f} ({int((td3_vs>0).sum())}/{len(seeds)}) \\\\")
        rows2.append(f"{nm} & " + " & ".join(f"{M[cn][c]['wait_min'].mean():.1f} & {M[cn][c]['over_events'].mean():.2f}" for c in ("Best fixed price", "TOU", "Rule-based", "PPO", "TD3")) + " \\\\")
    rows.insert(1, "\\midrule"); rows2.insert(1, "\\midrule")
    write("tab_sens", "\\begin{tabular}{@{}lccccccll@{}}\n\\toprule\nConfiguration & Static & Best fixed & TOU & Rule-based & PPO & TD3 & PPO vs best heuristic (\\%) & TD3 vs PPO (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    hdr2 = " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for n in ("Best fixed", "TOU", "Rule-based", "PPO", "TD3"))
    write("tab_sens_ops", "\\begin{tabular}{@{}lcccccccccc@{}}\n\\toprule\n & " + hdr2 + " \\\\\n" + "".join(f"\\cmidrule(lr){{{2+2*k}-{3+2*k}}}" for k in range(5)) + "\nConfiguration & " + " & ".join("Wait & Over." for _ in range(5)) + " \\\\\n\\midrule\n" + "\n".join(rows2) + "\n\\bottomrule\n\\end{tabular}")
    SUM["sens"] = dict(seeds=seeds, configs=summ)
    # figure
    names = ["nominal"] + done
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for a, key, ttl in ((ax[0], "ppo_vs_pts", "PPO vs best heuristic (%)"), (ax[1], "td3_vs_pts", "TD3 vs PPO (%)")):
        for k, cn in enumerate(names):
            pts = summ[cn][key]; a.scatter(pts, [k] * len(pts), s=9, color="C0", alpha=0.6)
            a.scatter([np.mean(pts)], [k], marker="D", s=22, color="k")
        a.axvline(0, color="grey", lw=0.8); a.set_xlabel(ttl); a.set_yticks(range(len(names)))
    ax[0].set_yticklabels(["Nominal"] + [label[c].split(" (")[0].replace("$\\times ", "x").replace("$", "") for c in done]); ax[0].invert_yaxis()
    fig.tight_layout(); fig.savefig("figs/fig_sens.pdf"); plt.close(fig)


# ====================================================================================== command authentication in the loop
def sigexp():
    R = load("results/control/sig_s*.json")
    if not R: return
    scen = ["none", "MITM, forged command", "MITM, stolen gateway key", "Replay of an off-peak command", "DDoS"]
    defs = ["No defense", "AE screening", "Signature only", "Signature, nonce, policy", "Signature, nonce, policy, safe fallback", "Full (contract rules and AE screening)"]
    g = lambda r, s, c, d, m: float(np.mean(r[f"{s}|{c}|{d}"]["m"][m]))
    hdr = "Defense & No attack & MITM, forged command & MITM, stolen gateway key & Replay of an off-peak command & DDoS"
    tabs = {}
    for metric, fname, dd in (("over_events", "tab_sig_over", 2), ("reward", "tab_sig_reward", 1)):
        lines = []
        for ci, c in enumerate(("PPO", "Rule-based")):
            for d in defs:
                lines.append(f"{d} & " + " & ".join(ms([g(r, s, c, d, metric) for r in R], dd) for s in scen) + " \\\\")
            if ci == 0: lines.append("\\midrule")
        write(fname, "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n" + hdr + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    SUM["sig"] = {"n_seeds": len(R), "over": {}, "reward": {}, "p": {}, "chan": {}}
    for c in ("PPO", "Rule-based"):
        for m, key in (("over_events", "over"), ("reward", "reward")):
            for s in scen:
                for d in defs:
                    v = [g(r, s, c, d, m) for r in R]
                    SUM["sig"][key][f"{s}|{c}|{d}"] = [float(np.mean(v)), float(np.std(v, ddof=1))]
    # tests for PPO: contract rules vs no defense, and full vs AE screening, per attack scenario, Holm across the four attack scenarios
    for m in ("over_events", "reward"):
        for a, b, nm in (("Signature, nonce, policy", "No defense", "contract_vs_none"), ("Full (contract rules and AE screening)", "AE screening", "full_vs_screen"),
                         ("Signature only", "No defense", "sigonly_vs_none")):
            ps = [stats.mannwhitneyu([g(r, s, "PPO", a, m) for r in R], [g(r, s, "PPO", b, m) for r in R]).pvalue for s in scen[1:]]
            adj = holm(ps)
            SUM["sig"]["p"][f"{m}|{nm}"] = {s: [float(p), float(q)] for s, p, q in zip(scen[1:], ps, adj)}
    # processing statistics from the channel counters (PPO runs)
    tot = lambda s, d, k: np.array([r[f"{s}|PPO|{d}"]["det"]["chan"][k] for r in R], float)
    for d in ("Signature only", "Signature, nonce, policy"):
        for s in scen:
            SUM["sig"]["chan"][f"{s}|{d}"] = {k: float(tot(s, d, k).mean() / 30.0) for k in ("delivered_malicious", "acc_malicious", "rej_bad_signature", "rej_replay", "rej_out_of_policy", "rej_legit")}
    sg = np.concatenate([tot(s, "Signature, nonce, policy", "sign_s") / np.maximum(tot(s, "Signature, nonce, policy", "n_sign"), 1) for s in scen]) * 1e6
    vf = np.concatenate([tot(s, "Signature, nonce, policy", "verify_s") / np.maximum(tot(s, "Signature, nonce, policy", "n_verify"), 1) for s in scen]) * 1e6
    nmsg = np.concatenate([tot(s, "Signature, nonce, policy", "n_verify") / 30.0 for s in scen[:1]])
    SUM["sig"]["crypto"] = dict(sign_us=float(sg.mean()), verify_us=float(vf.mean()), msgs_per_day_none=float(nmsg.mean()),
                                legit_rejections_total=float(sum(np.array([r[f"{s}|{c}|{d}"]["det"]["chan"]["rej_legit"] for r in R]).sum() for s in scen for c in ("PPO", "Rule-based") for d in ("Signature only", "Signature, nonce, policy", "Signature, nonce, policy, safe fallback", "Full (contract rules and AE screening)"))))
    lines = []
    for s in scen[1:]:
        a = SUM["sig"]["chan"][f"{s}|Signature only"]; b = SUM["sig"]["chan"][f"{s}|Signature, nonce, policy"]
        dm = b["delivered_malicious"]
        if dm > 0:
            lines.append(f"{s} & {dm:.0f} & {100*a['acc_malicious']/dm:.0f} & {100*b['acc_malicious']/dm:.0f} & {b['rej_bad_signature']:.0f} & {b['rej_replay']:.0f} & {b['rej_out_of_policy']:.0f} \\\\")
        else:
            lines.append(f"{s} & 0 & -- & -- & 0 & 0 & 0 \\\\")
    write("tab_sig_reject", "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nScenario & Attacked-station commands per day & Accepted, signature only (\\%) & Accepted, full contract rules (\\%) & Rejected: bad signature & Rejected: stale nonce & Rejected: out of policy \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    # figure: PPO overloads and reward for four defenses
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
    use = ["No defense", "AE screening", "Signature, nonce, policy", "Full (contract rules and AE screening)"]
    labs = ["None", "AE screening", "Contract rules", "Contract rules + AE"]
    xs = np.arange(len(scen)); w = 0.2
    for a, key, yl in ((ax[0], "over", "Overload events per day"), (ax[1], "reward", "Episode reward")):
        for k, d in enumerate(use):
            mu = [SUM["sig"][key][f"{s}|PPO|{d}"][0] for s in scen]; sd = [SUM["sig"][key][f"{s}|PPO|{d}"][1] for s in scen]
            a.bar(xs + (k - 1.5) * w, mu, w, yerr=sd, capsize=1.5, label=labs[k], error_kw=dict(lw=0.6))
        a.set_xticks(xs); a.set_xticklabels(["None", "MITM\nforged", "MITM\nstolen key", "Replay", "DDoS"], fontsize=6.5); a.set_ylabel(yl)
    ax[1].set_ylim(100, 140); ax[0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig("figs/fig_sig.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in (forecasting, control, ablation, security, fed, attack, ledger, edge, scale, profile_fig, sens, sigexp):
        try:
            f()
        except Exception as e:
            import traceback; print("FAILED", f.__name__, e); traceback.print_exc()
    json.dump(SUM, open("results/summary.json", "w"), indent=1)
    print("ok", list(SUM.keys()))
