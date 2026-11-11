# EVAThink

<p align="center">
<a href=""><img src="assets/evathink-logo.png" alt="logo" width="400px"></a>
</p>

<p align="center"><b>EVAThink: Evidence Sufficiency-Guided Training for Adaptive Thinking in Video Understanding</b></p>

<p align="center">
    <a href="https://arxiv.org/abs/2611.13907", target="_blank"><img src="https://img.shields.io/badge/arXiv-2611.13907-b31b1b.svg"></a>
    <a href="https://huggingface.co/collections/MaplesWCT/evathink" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20-Hugging%20Face-blue?color=blue&logoColor=white" /></a>
</p>

Code for **EVAThink**, a video-language model trained with reinforcement learning to decide per question whether to
answer directly or to write a short evidence-grounded reasoning trace first.

The repository has two parts: the **data curation** that turns the question pool into paired training data, and a
small **verl extension** for the two-stage RL.

```
curation/           data curation pipeline (run_curation.sh drives it)
verl_ext/           verl v0.8.0 extension: entry point, paired sampler, real-timestamp dataset,
                    first-token regularizer, reward, runtime patches
examples/           training launch scripts for Stage 1 and Stage 2
```

Comments inside the source files are partly in Chinese.

## Requirements

- verl **v0.8.0**, Ray 2.48, vLLM 0.20.2 (vllm-ascend), transformers 5.5.3, torch 2.10 + torch_npu, `qwen_vl_utils`
- curation also needs OpenCV, PyAV, ffmpeg/x264 (to cut and concatenate videos), pandas, pyarrow
- hardware: Ascend 910 (64 GB). Training used 4 nodes × 8 cards

## Data curation

Every question is answered on **two views of the same evidence**: a *dense* view where the evidence fills the clip,
and a *sparse* view where it is diluted in a longer video. A pair is kept when the base model (Qwen3-VL-8B-Instruct)
answers the dense view directly much better than the sparse view, and when reasoning recovers the sparse view.

All model calls run on the **same 32 exactly decoded frames with their real timestamps** that training later uses,
so the measured signals describe what the policy actually sees.

`curation/run_curation.sh <step>` runs one step at a time. Set `EVATHINK_WORK`, `DATAPOOL`, `EVATHINK_VIDEO_ROOT`,
`EVATHINK_QWEN3VL`, `ANNOTATOR`, and for the inference steps `NSHARD` / `DEVICES_PER_SHARD` / `DEVICE_ENV`
(see the header of the script).

| Step | What it does | Main program | Output (in `$EVATHINK_WORK`) |
|---|---|---|---|
| `prompts` | direct-answer prompt for every pool question | `build_prompt4rollout_direct.py` | `full/prompt4rollout_direct.jsonl` |
| `frames` | 32 frames per video, sequential decoding (no seeking), real fps / frame indices in `meta.json` | `recurate_extract_exact.py` | `frames/<video_key>__32f/` |
| `direct` | Qwen3-VL-8B answers each question 8 times (T=1.0, top-k 20, ≤2048 tokens) | `recurate_direct_exact.py` | `rollout/rollout_direct.shard*.jsonl` |
| `stage1` | rule-based scoring; drop questions solved 8/8; accuracy < 0.5 → sparse view, else dense; evidence-localization prompts for sparse questions (answers in seconds) | `eval_rollout_acc_rule.py`, `add_duration_from_meta.py`, `build_prompt4anno_gt.py`, `recurate_anno_time_prompt.py` | `full/rollout_direct_valid.jsonl`, `full/prompt4anno_gt_time.jsonl` |
| `anno` | Qwen3.5-27B localizes the evidence (T=0, ≤256 tokens, thinking off); seconds → frame indices | `recurate_infer_exact.py`, `convert_anno_time.py` | `full/anno_gt.jsonl` |
| `contra` | sparse questions: cut the evidence segment → dense counterpart; dense questions: concatenate with another video of similar length → sparse counterpart (the curation prompt names the segment; the training prompt does not); extract their frames | `build_sample_sparse2dense.py`, `build_sample_dense2sparse.py`, `merge_contra.py`, `recurate_extract_exact.py` | `full/prompt4rollout_direct_contra.jsonl` |
| `contra_direct` | direct answers ×8 on the counterparts (≤512 tokens) | `recurate_infer_exact.py` | `rollout/rollout_direct_contra.shard*.jsonl` |
| `evid` | keep pairs with `dense_acc > sparse_acc + 0.125`; reasoning prompt for the sparse view | `eval_rollout_acc_rule_lite.py`, `select_sample_evidrelated.py`, `build_prompt4rollout_slow.py` | `full/sample_evidrelated.jsonl`, `full/prompt4rollout_slow.jsonl` |
| `slow` | reasoning answers ×8 on the sparse view (≤1024 tokens) | `recurate_infer_exact.py` | `rollout/rollout_slow.shard*.jsonl` |
| `final` | keep pairs with `sparse_slow_acc − sparse_acc > 0.125` (**gold pairs**) | `eval_rollout_acc_rule_lite.py`, `select_sample_final.py` | `full/sample_final.jsonl` |
| `trainset` | Stage 1 = strict pairs (`dense_acc ≥ 0.75` and gain ≥ 0.5); Stage 2 = remaining gold pairs | `build_trainset.py` (uses `assemble_trainset4verl.py`) | `datasets/stage1`, `datasets/stage2` |

Each training row holds the question, the response-mode instruction (`postamble.txt`), the 32 frame paths of one view,
and the curation signals in `extra_info`; `ts_map.json` maps every frame folder to its real fps and frame indices.
Counterpart videos are written next to the source videos (`<folder>_contra_qwen_new/`), so the video root must be
writable.

We ran the pool in two batches (25.7k and 92.5k questions) and merged them when building the training sets;
`build_trainset.py --final` accepts several `sample_final.jsonl` files for this.

## Training with verl

### How the extension plugs into verl

Everything is attached through verl's own extension points; **no verl source file is modified**.

| Piece | File | How it is attached |
|---|---|---|
| Paired sampling: the dense and sparse rows of a question land in the same optimizer step (each still forms its own GRPO group) | `paired_sampler.py`, `paired_task_runner.py` | run `python -m evid_paired_main` instead of `python -m verl.trainer.main_ppo`; same Hydra config |
| Real timestamps: Qwen3-VL writes frame times into the prompt, and a folder of JPEGs would otherwise get fake ones (every clip looks 15.2 s long) | `ts_dataset.py` | `data.custom_cls.path=<repo>/verl_ext/ts_dataset.py`, `data.custom_cls.name=RealTimestampDataset`, env `EVID_TS_MAP=<dataset>/ts_map.json` |
| Reward: format (start with `<answer>` or `Alright`, one valid answer tag at the end) + answer table by view and mode | `reward_evathink.py` | `custom_reward_function.path=<repo>/verl_ext/reward_evathink.py`, `custom_reward_function.name=compute_score`; table via env `REWARD_*` |
| First-token regularizer: penalizes the first token's probability above a cap so neither mode becomes absorbing | `firsttok_reg.py` | installed by the paired task runner around `ray_trainer.compute_advantage`; env `EVID_FIRSTTOK_*` |
| Runtime patches: ndarray→list fix for frame lists read from parquet; optional chunked entropy against OOM | `sitecustomize.py` | loaded automatically when `verl_ext/` is first on `PYTHONPATH` |

`paired_task_runner.py` contains a copy of verl v0.8.0's `TaskRunner.run` with only the sampler changed; it checks
on import whether your verl's `run` has drifted from that copy and warns. With another verl version, re-align it.

Environment variables:

| Variable | Meaning |
|---|---|
| `EVID_TS_MAP` | path of `ts_map.json` (required for real timestamps) |
| `EVID_PAIR_DISABLE=1` | fall back to verl's default sampler |
| `EVID_PAIR_KEY` / `EVID_PAIR_GROUP_SIZE` | pairing key in `extra_info` (default `problem_id`) / rows per group (default 2) |
| `EVID_FIRSTTOK_CAP` | symmetric cap c (Stage 2: 0.85) |
| `EVID_FIRSTTOK_CAP_WRONG` / `EVID_FIRSTTOK_CAP_RIGHT` | one-sided caps for the wrong / right mode of each view (Stage 1: 0.5 / 0.85); take precedence over `EVID_FIRSTTOK_CAP` |
| `EVID_FIRSTTOK_BETA` | regularizer strength β (we used 5) |
| `REWARD_{DENSE,SPARSE}_{OK,NG}_{DIRECT,THINK}`, `REWARD_FORMAT_OK/BAD` | reward table; defaults are the Stage 1 table |
| `EVID_ENTROPY_CHUNK=1` | chunked entropy (we needed it with `use_remove_padding=False`) |

All of these are read inside Ray actors. On a multi-node cluster, export them, together with `PYTHONPATH`,
**before** `ray start` on every node.

### Running

1. Install verl v0.8.0 and its rollout backend.
2. Get the data: either run the curation above, or download the released datasets and unpack their frames with
   absolute paths:
   ```bash
   hf download MaplesWCT/EVAThink-Dataset-Stage1 --repo-type dataset --local-dir data/stage1
   python data/stage1/extract_frames.py --abs        # -> data/stage1/train.abs.parquet
   ```
3. Launch (settings in `examples/train_common.sh` are the ones we used):
   ```bash
   # Stage 1, from Qwen3-VL-8B-Instruct, 110 steps
   MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct DATA_DIR=data/stage1 TRAIN_FILE=data/stage1/train.abs.parquet \
       bash examples/train_stage1.sh
   # Stage 2, from the Stage 1 step-110 checkpoint converted to HF format, 100 steps
   MODEL_PATH=<stage1-hf> DATA_DIR=data/stage2 TRAIN_FILE=data/stage2/train.abs.parquet \
       bash examples/train_stage2.sh
   ```
   Any extra Hydra override can be appended to the command line.

| | Stage 1 | Stage 2 |
|---|---|---|
| start | Qwen3-VL-8B-Instruct | Stage 1, step 110 |
| data | 1,766 strict pairs (3,532 rows) | 4,179 remaining gold pairs (8,358 rows) |
| reward (correct direct / correct think / wrong direct / wrong think) | dense 2 / 1 / 0 / −1, sparse 1 / 2 / −1 / 0 | 1.1 / 1.0 / 0 / 0 on both views |
| malformed response | −1 | −1 |
| advantage | GRPO, divided by group std | GRPO, not divided by std |
| first-token cap | wrong mode 0.5, right mode 0.85, β=5 | both modes 0.85, β=5 |
| checkpoint used | step 110 (one epoch) | step 100 |

Shared: 32 prompts × 8 rollouts per step, one update per step, lr 1e-6 constant, no KL, token-mean loss,
max response 1,024 tokens, T=1.0.

