#!/usr/bin/env bash
# EVAThink data curation, end to end. Run step by step:  bash run_curation.sh <step>
#
#   prompts       datapool_mc.jsonl -> direct-answer prompts
#   frames        exact 32-frame extraction for every source video            (CPU)
#   direct        Qwen3-VL-8B direct answers x8 on the original videos        (accelerators)
#   stage1        score -> per-question accuracy -> dense/sparse view -> evidence-annotation prompts
#   anno          Qwen3.5-27B evidence localization on the sparse questions   (accelerators)
#   contra        build counterpart videos (sparse->dense clip, dense->sparse concat) + extract their frames (CPU)
#   contra_direct Qwen3-VL-8B direct answers x8 on the counterpart videos     (accelerators)
#   evid          score -> keep evidence-dependent pairs -> slow-thinking prompts for the sparse view
#   slow          Qwen3-VL-8B reasoning answers x8 on the sparse view         (accelerators)
#   final         score -> keep pairs where reasoning recovers the sparse view -> sample_final.jsonl (gold pairs)
#   trainset      Stage 1 (strict pairs) and Stage 2 (remaining gold pairs) verl datasets
#
# Settings (environment variables):
#   EVATHINK_WORK        working directory (default: <repo>/work)
#   DATAPOOL             datapool_mc.jsonl (default: $EVATHINK_WORK/datapool_mc.jsonl)
#   EVATHINK_VIDEO_ROOT  root of the relative video paths in the datapool (required for `prompts`)
#   EVATHINK_QWEN3VL     Qwen3-VL-8B-Instruct path (rollout model)
#   ANNOTATOR            Qwen3.5-27B path (evidence annotator)
#   NSHARD               inference shards run in parallel on this machine (default 1)
#   DEVICES_PER_SHARD    accelerators per shard = vLLM tensor parallel size (default 1; use 4 for the 27B annotator)
#   DEVICE_ENV           visible-devices variable (default CUDA_VISIBLE_DEVICES; ASCEND_RT_VISIBLE_DEVICES on Ascend)
#   SUFFIX               folder suffix of the counterpart videos (default _contra_qwen_new, as in the released data)
#   PY                   python interpreter (needs vllm, transformers, qwen_vl_utils, opencv, PyAV, pandas, pyarrow)
# Counterpart videos are written next to the source videos (<folder><SUFFIX>/), so the video root must be writable.
set -euo pipefail
C=$(cd "$(dirname "$0")" && pwd)
W=${EVATHINK_WORK:-$(dirname "$C")/work}; export EVATHINK_WORK=$W
DATAPOOL=${DATAPOOL:-$W/datapool_mc.jsonl}
PY=${PY:-python3}
NSHARD=${NSHARD:-1}; DPS=${DEVICES_PER_SHARD:-1}; DEVICE_ENV=${DEVICE_ENV:-CUDA_VISIBLE_DEVICES}
SUFFIX=${SUFFIX:-_contra_qwen_new}
ANNOTATOR=${ANNOTATOR:-Qwen/Qwen3.5-27B}
F=$W/full; R=$W/rollout; FR=$W/frames
mkdir -p "$F" "$R" "$W/logs"

# run_shards <script> <args...>: NSHARD processes, each on DPS devices, with --shard k --nshard NSHARD appended
run_shards() {
  local pids=() k
  for k in $(seq 0 $((NSHARD - 1))); do
    env "$DEVICE_ENV=$(seq -s, $((k * DPS)) $((k * DPS + DPS - 1)))" TP=$DPS \
      "$PY" "$@" --shard "$k" --nshard "$NSHARD" > "$W/logs/$(basename "$1" .py).$k.log" 2>&1 &
    pids+=($!)
  done
  for k in "${pids[@]}"; do wait "$k"; done
}
merge() { cat "$R/$1".shard*.jsonl > "$2"; echo "merged $(ls "$R/$1".shard*.jsonl | wc -l) shards -> $2"; }

step=${1:?usage: run_curation.sh prompts|frames|direct|stage1|anno|contra|contra_direct|evid|slow|final|trainset}
case $step in
prompts)
  : "${EVATHINK_VIDEO_ROOT:?set EVATHINK_VIDEO_ROOT to the root of the datapool video paths}"
  "$PY" "$C/build_prompt4rollout_direct.py" --input "$DATAPOOL" --video-root "$EVATHINK_VIDEO_ROOT" \
      --output "$F/prompt4rollout_direct.jsonl" ;;
frames)
  "$PY" "$C/recurate_extract_exact.py" --out "$W" --valid "$F/prompt4rollout_direct.jsonl" ;;
direct)
  # T=1.0 top_p=1.0 top_k=20 max_tokens=2048 n=8 (fixed in the script)
  PROBE_MODEL=${EVATHINK_QWEN3VL:-Qwen/Qwen3-VL-8B-Instruct} \
    run_shards "$C/recurate_direct_exact.py" --out "$W" --valid "$F/prompt4rollout_direct.jsonl" ;;
stage1)
  merge rollout_direct "$F/rollout_direct.jsonl"
  # valid = questions not solved 8/8; view: acc < 0.5 -> sparse, else dense
  "$PY" "$C/eval_rollout_acc_rule.py" -i "$F/rollout_direct.jsonl" -g "$DATAPOOL" -o "$F/rollout_direct_eval.jsonl" \
      -e "$F/rollout_direct_valid_nodur.jsonl"
  "$PY" "$C/add_duration_from_meta.py" --input "$F/rollout_direct_valid_nodur.jsonl" --frames-root "$FR" \
      --output "$F/rollout_direct_valid.jsonl"
  # evidence-localization prompts for the sparse questions, then switched to answers in seconds
  "$PY" "$C/build_prompt4anno_gt.py" --input "$F/rollout_direct_valid.jsonl" --gt "$DATAPOOL" \
      --output "$F/prompt4anno_gt.jsonl"
  "$PY" "$C/recurate_anno_time_prompt.py" --input "$F/prompt4anno_gt.jsonl" --output "$F/prompt4anno_gt_time.jsonl" ;;
anno)
  run_shards "$C/recurate_infer_exact.py" --input "$F/prompt4anno_gt_time.jsonl" --out "$W" --name anno_gt_time \
      --model "$ANNOTATOR" --n 1 --temperature 0 --max-tokens 256 --no-thinking
  merge anno_gt_time "$F/anno_gt_raw.jsonl"
  "$PY" "$C/convert_anno_time.py" --input "$F/anno_gt_raw.jsonl" --frames-root "$FR" --output "$F/anno_gt.jsonl" ;;
contra)
  mkdir -p "$W/tmp"
  # sparse questions: cut the annotated evidence segment -> dense counterpart
  "$PY" "$C/build_sample_sparse2dense.py" --valid "$F/rollout_direct_valid.jsonl" --gt-anno "$F/anno_gt.jsonl" \
      --rollout-prompt /nonexistent --suffix "$SUFFIX" --out "$F/contra_s2d.jsonl" --tmp-dir "$W/tmp"
  # dense questions: concatenate with another video of similar length -> sparse counterpart
  "$PY" "$C/build_sample_dense2sparse.py" --input "$F/rollout_direct_valid.jsonl" --suffix "$SUFFIX" \
      --out "$F/contra_d2s.jsonl" --tmp-dir "$W/tmp"
  "$PY" "$C/merge_contra.py" --s2d "$F/contra_s2d.jsonl" --d2s "$F/contra_d2s.jsonl" --suffix "$SUFFIX" \
      --output "$F/prompt4rollout_direct_contra.jsonl"
  "$PY" "$C/recurate_extract_exact.py" --out "$W" --valid "$F/prompt4rollout_direct_contra.jsonl" ;;
contra_direct)
  run_shards "$C/recurate_infer_exact.py" --input "$F/prompt4rollout_direct_contra.jsonl" --out "$W" \
      --name rollout_direct_contra --model "${EVATHINK_QWEN3VL:-Qwen/Qwen3-VL-8B-Instruct}" --n 8 --temperature 1.0 \
      --top-k 20 --max-tokens 512 ;;
evid)
  merge rollout_direct_contra "$F/rollout_direct_contra.jsonl"
  "$PY" "$C/eval_rollout_acc_rule_lite.py" -i "$F/rollout_direct_contra.jsonl" -g "$DATAPOOL" \
      -o "$F/rollout_direct_contra_eval.jsonl" -e "$F/rollout_direct_contra_valid.jsonl"
  # keep pairs with dense_acc > sparse_acc + 0.125
  "$PY" "$C/select_sample_evidrelated.py" --orig-eval "$F/rollout_direct_valid.jsonl" \
      --contra-eval "$F/rollout_direct_contra_valid.jsonl" --out "$F/sample_evidrelated.jsonl"
  "$PY" "$C/build_prompt4rollout_slow.py" --input "$F/sample_evidrelated.jsonl" --output "$F/prompt4rollout_slow.jsonl" ;;
slow)
  run_shards "$C/recurate_infer_exact.py" --input "$F/prompt4rollout_slow.jsonl" --out "$W" --name rollout_slow \
      --model "${EVATHINK_QWEN3VL:-Qwen/Qwen3-VL-8B-Instruct}" --n 8 --temperature 1.0 --top-k 20 --max-tokens 1024 ;;
final)
  merge rollout_slow "$F/rollout_slow.jsonl"
  "$PY" "$C/eval_rollout_acc_rule_lite.py" -i "$F/rollout_slow.jsonl" -g "$DATAPOOL" -o "$F/rollout_slow_eval.jsonl" \
      -e "$F/rollout_slow_valid.jsonl"
  # keep pairs with sparse_slow_acc - sparse_acc > 0.125
  "$PY" "$C/select_sample_final.py" --slow-valid "$F/rollout_slow_valid.jsonl" --evid "$F/sample_evidrelated.jsonl" \
      --out "$F/sample_final.jsonl" ;;
trainset)
  "$PY" "$C/build_trainset.py" --mode strict --final "$F/sample_final.jsonl" --source "$DATAPOOL" \
      --frames-root "$FR" --out "$W/datasets/stage1"
  "$PY" "$C/build_trainset.py" --mode rest --final "$F/sample_final.jsonl" --source "$DATAPOOL" \
      --frames-root "$FR" --exclude "$W/datasets/stage1/sample_strict.jsonl" --out "$W/datasets/stage2" ;;
*) echo "unknown step: $step"; exit 1 ;;
esac
