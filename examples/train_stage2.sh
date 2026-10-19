#!/usr/bin/env bash
# Stage 2: keep reasoning only where it pays off.
#   start      the Stage 1 step-110 checkpoint (EVAThink-Stage1)
#   data       EVAThink-Dataset-Stage2 (4,179 remaining gold pairs x 2 views = 8,358 rows)
#   reward     view-agnostic: correct+direct 1.1 | correct+think 1.0 | wrong 0; malformed response -1
#   advantage  GRPO without dividing by the group std (norm_adv_by_std_in_grpo=False)
#   first-token regularizer (symmetric): either mode capped at p=0.85, beta=5
#   we used the step-100 checkpoint (= EVAThink-Qwen3); optional validation every TEST_FREQ steps
# Usage: MODEL_PATH=<Stage 1 checkpoint (HF)> DATA_DIR=<EVAThink-Dataset-Stage2 with absolute frame paths> bash train_stage2.sh
set -euo pipefail
export EXP=${EXP:-evathink_stage2}
export NORM_ADV_STD=False
export REWARD_DENSE_OK_DIRECT=1.1 REWARD_DENSE_OK_THINK=1 REWARD_DENSE_NG_DIRECT=0 REWARD_DENSE_NG_THINK=0
export REWARD_SPARSE_OK_DIRECT=1.1 REWARD_SPARSE_OK_THINK=1 REWARD_SPARSE_NG_DIRECT=0 REWARD_SPARSE_NG_THINK=0
export REWARD_FORMAT_OK=0 REWARD_FORMAT_BAD=-1 REWARD_FORMAT_W=1 REWARD_ANSWER_W=1
unset EVID_FIRSTTOK_CAP_WRONG EVID_FIRSTTOK_CAP_RIGHT
export EVID_FIRSTTOK_CAP=0.85 EVID_FIRSTTOK_BETA=5
export SAVE_FREQ=${SAVE_FREQ:-20} TOTAL_STEPS=${TOTAL_STEPS:-100}
exec bash "$(dirname "$0")/train_common.sh" "$@"
