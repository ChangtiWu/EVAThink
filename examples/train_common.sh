#!/usr/bin/env bash
# Shared verl launch used by train_stage1.sh / train_stage2.sh. Do not call directly.
# These are the exact settings of our runs (4 nodes x 8 Ascend 910 64 GB, verl v0.8.0, vLLM rollout).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)

: "${MODEL_PATH:?}" "${DATA_DIR:?}" "${EXP:?}"
CKPT=${CKPT_DIR:-$ROOT/checkpoints}/$EXP
NNODES=${NNODES:-4}; NGPUS=${NGPUS_PER_NODE:-8}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}         # released datasets: train.abs.parquet (see README)
VAL_FILES=${VAL_FILES:-$TRAIN_FILE}                      # verl needs a val file even when TEST_FREQ=-1

# verl_ext first: provides evid_paired_main, ts_dataset.py, the reward, and sitecustomize.py (both patches).
# Append to PYTHONPATH, never overwrite it (on Ascend the CANN paths live there).
export PYTHONPATH="$ROOT/verl_ext:${PYTHONPATH:-}"
# Real timestamps: the dataset class looks up each frame folder here. Read inside Ray actors, so with a
# multi-node cluster export it (and everything else in this file) BEFORE `ray start` on every node.
export EVID_TS_MAP=${EVID_TS_MAP:-$DATA_DIR/ts_map.json}
export EVID_ENTROPY_CHUNK=${EVID_ENTROPY_CHUNK:-1}          # chunked entropy, avoids OOM with use_remove_padding=False
export REWARD_MAXLEN_TOKENS=1024
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$CKPT/tensorboard}

exec python3 -m evid_paired_main \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_STD} \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILES" \
    data.custom_cls.path="$ROOT/verl_ext/ts_dataset.py" \
    data.custom_cls.name=RealTimestampDataset \
    data.image_key=images \
    data.video_key=videos \
    data.train_batch_size=32 \
    data.max_prompt_length=20480 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    custom_reward_function.path="$ROOT/verl_ext/reward_evathink.py" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_model_len=22528 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.balance_batch=True \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=evathink \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$CKPT" \
    trainer.rollout_data_dir="$CKPT/rollouts" \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=$NNODES \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ:--1} \
    trainer.total_epochs=3 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    "$@"
