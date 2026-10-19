#!/usr/bin/env bash
# Stage 1: learn which response mode each view needs.
#   start      Qwen3-VL-8B-Instruct
#   data       EVAThink-Dataset-Stage1 (1,766 strict pairs x 2 views = 3,532 rows)
#   reward     view-dependent table (reward_evathink.py defaults):
#                dense : correct+direct 2 | correct+think 1 | wrong+direct 0 | wrong+think -1
#                sparse: correct+think 2 | correct+direct 1 | wrong+think 0 | wrong+direct -1
#                malformed response -1
#   advantage  GRPO, divided by the group std
#   first-token regularizer (one-sided): wrong mode capped at p=0.5, right mode at p=0.85, beta=5
#   110 steps (one epoch); we used the step-110 checkpoint (= EVAThink-Stage1)
# Usage: MODEL_PATH=<Qwen3-VL-8B-Instruct> DATA_DIR=<EVAThink-Dataset-Stage1 with absolute frame paths> bash train_stage1.sh
set -euo pipefail
export EXP=${EXP:-evathink_stage1}
export NORM_ADV_STD=True
unset EVID_FIRSTTOK_CAP
export EVID_FIRSTTOK_CAP_WRONG=0.5 EVID_FIRSTTOK_CAP_RIGHT=0.85 EVID_FIRSTTOK_BETA=5
export SAVE_FREQ=${SAVE_FREQ:-20} TOTAL_STEPS=${TOTAL_STEPS:-110}
exec bash "$(dirname "$0")/train_common.sh" "$@"
