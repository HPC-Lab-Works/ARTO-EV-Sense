#!/bin/sh
# usage: pipeline.sh seed [seed ...]  (one worker = one CPU core)
cd "$(dirname "$0")"
for s in "$@"; do
  python run_control.py main $s
  python run_security.py $s
  python run_fed.py $s
  python run_attack.py $s
  python run_scale.py $s
  python run_sig.py $s
  python run_control.py ablate $s
done
