#!/usr/bin/env bash
# Restores results/models, results/control and results/security from the archives in results/.
# The raw files are stored as archives only so that the repository can be uploaded through the GitHub web
# interface (at most 100 files and 25 MB per file per upload). Run this once after cloning.
set -e
cd "$(dirname "$0")/results"
for a in models_rl_ppo_dqn models_rl_sac_td3 models_lstm_det raw_control_security; do
  tar xzf "$a.tar.gz"
done
echo "Restored: $(ls models | wc -l) model files, $(ls control | wc -l) control files, $(ls security | wc -l) security files."
