# ARTO-EV synthetic testbench

Code and raw results (ten seeds, trained models, tables and figures) for the study "ARTO-EV: A Sensor-Driven Reinforcement Learning Framework with Anomaly Detection
for Secure Real-Time Management of Electric Vehicle Charging Networks".

**Everything here is synthetic.** No real charging logs, traffic feeds, weather records or smart-meter data are used.
All parameters are assumptions in plausible ranges (see `evsim/config.py`, `evsim/gen.py`, `evsim/security.py`); they are not
fitted to field data. Absolute numbers are not predictions for a real network; only comparisons under identical
conditions are meaningful.

## Repository

This repository is intended for https://github.com/HPC-Lab-Works/ARTO-EV-Sense. It is released under the MIT licence (see `LICENSE`). `requirements.txt` lists the package versions used for the reported results. The trained models (about 70 MB) and the raw per-seed control and security results are stored as four compressed archives in `results/` (each below 25 MB, so the repository can be uploaded through the GitHub web interface); run `bash unpack_results.sh` once after cloning to restore `results/models`, `results/control` and `results/security` before running `aggregate.py`. Git LFS is not needed.

## Layout

| Path | Content |
|---|---|
| `evsim/config.py` | all simulator constants and reward weights |
| `evsim/gen.py` | exogenous generator: temperature, cloud/PV, feeder base load, grid-stress events, tariff, arrival profiles |
| `evsim/env.py` | vectorised charging-network simulator (E days x N stations in lock-step) |
| `evsim/forecast.py` | LSTM headroom forecaster and baselines (persistence, seasonal naive, ridge, MLP) |
| `evsim/agents.py` | Static, best-fixed-price, TOU, rule-based controllers; PPO, double DQN, SAC, TD3 (PyTorch); evaluation |
| `evsim/security.py` | telemetry model, FDI/MITM/DDoS attacks, autoencoder and baseline detectors, metrics |
| `evsim/secure.py` | closed-loop run under attack with detector-based screening, edge fallback and the authenticated command channel `CmdChannel` |
| `evsim/ledger.py` | in-process permissioned ledger: hash chain, Merkle roots, Ed25519, smart-contract rules, simulated PBFT commit |
| `run_forecast.py` | forecasting study (10 seeds) |
| `run_control.py` | `tune`, `main` (all controllers) and `ablate` modes |
| `run_security.py`, `run_fed.py` | detection study, federated vs local vs centralised autoencoders |
| `run_attack.py` | closed-loop robustness under attack, with and without screening |
| `run_sens.py` | retrained one-factor-at-a-time sensitivity study (demand, price response, reward weights); PPO and TD3 on 4 seeds |
| `run_sig.py` | closed-loop study of command authentication (real Ed25519 signatures, nonces, policy bounds; forgery, key compromise, replay, DDoS) |
| `run_scale.py` | zero-shot transfer to N = 50, 100, 500 stations and distribution-shift scenarios |
| `run_ledger.py`, `run_edge.py` | ledger throughput, tamper tests and in-loop overhead; edge inference cost |
| `aggregate.py` | builds all tables (`tables/`), figures (`figs/`) and `results/summary.json` from the raw result files |
| `pipeline.sh` | per-seed pipeline used for the reported results |
| `results/` | raw per-seed JSON files, trained models and tuning results (models and per-seed control and security files as `.tar.gz` archives) |
| `unpack_results.sh` | restores the archived results |

## Reproducing

```
pip install torch numpy scipy scikit-learn matplotlib cryptography
python run_forecast.py                       # ~12 min on 2 CPU cores
python run_control.py tune 0; python run_control.py tune 1   # learning-rate selection (results/tuned_lr.json)
./pipeline.sh 0 1 2 3 4 5 6 7 8 9            # or split the seeds over several workers
python run_sig.py 0 1 2 3 4 5 6 7 8 9        # ~4 min per seed
python run_sens.py 0 1 2 3                   # ~40 min per seed (8 configurations)
python run_ledger.py; python run_edge.py 0 1 2 3 4 5 6 7 8 9
python aggregate.py
```

The results reported in the paper were produced on 2 CPU cores without a GPU (Python 3.11, PyTorch 2.14). One seed of the
control experiment (four learning agents at 3e5 environment steps each) takes about 25 minutes on one core.

## Seeds and data separation

Seed `s` (0-9) determines the forecaster, detector and agent training data, the initialisation and exploration of every
learning agent, the tuning of the heuristic controllers, and the held-out evaluation days (`50000+s` for the main
comparison, `60000+s` attack scenarios, `70000+s` scalability, `80000+s` shifts). Validation days (`90000+s`) are used only for
tuning the heuristics. Learning-rate tuning uses seeds 100 and 101, which are not used in the evaluation.

## Known simplifications

* One network-level action (price multiplier and number of enabled chargers) is applied to all stations.
* Grid headroom is a soft constraint (penalised, counted as overload events), not enforced by curtailment.
* The ledger is a single-process prototype. Validator network delays are simulated; hashing and signature costs are measured.
* The "rule-based IDS" is a threshold monitor calibrated on benign statistics, not Snort or a public rule set.
* FedAvg is evaluated on detection quality and communication volume only; no privacy attack is implemented.
* Edge latencies are measured on a server-class CPU core and are not gateway-hardware measurements.
