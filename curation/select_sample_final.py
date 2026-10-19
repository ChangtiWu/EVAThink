#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""挑出最终的黄金样本对: sparse 视角上慢思考(slow)比直答(direct)正确率提升的题。

select_sample_evidrelated.py 挑出的每题有一个 dense 视角(直答就能答对)和一个 sparse 视角(直答答不好)。
随后对每题的 sparse 视角做慢思考 rollout(build_prompt4rollout_slow.py)。若 sparse 视角慢思考的正确率比直答
高出 --improve-threshold(默认 0.125), 说明推理能补上证据稀疏造成的差距, 该题两个视角一并保留; 否则丢弃。

输入:
  - --slow-valid: eval_rollout_acc_rule_lite.py 对慢思考 rollout 的产出, 每题一行, acc = sparse 视角慢思考正确率;
    行里已带 evid 阶段的 dense_acc / sparse_acc / gap / evid_case(慢思考 prompt 起一路透传)。
  - --evid: sample_evidrelated.jsonl, 每题两行(dense + sparse)。

输出: 命中的题从 sample_evidrelated.jsonl 原样拷两行(dense + sparse, 字段与顺序不变), 只多一个字段
  sparse_slow_acc(两行都写)。slow_gain = sparse_slow_acc - sparse_acc, 需要时现算。

用法:
  python select_sample_final.py --slow-valid rollout_slow_valid.jsonl --evid sample_evidrelated.jsonl       --out sample_final.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evathink_paths import WORK, VIDEO_ROOT, QWEN3VL  # noqa: E402
ROOT = WORK
DEFAULT_SLOW = f"{ROOT}/outputs/qwen3vl-8b/rollout_slow_valid.jsonl"
DEFAULT_EVID = f"{ROOT}/outputs/qwen3vl-8b/sample_evidrelated.jsonl"
DEFAULT_OUT = f"{ROOT}/outputs/qwen3vl-8b/sample_final.jsonl"


def load_valid(path):
    """读 *_valid.jsonl(每题一行、自带 acc), 返回 {pid: 整行}。

    也兼容 per-rollout 的 *_eval.jsonl:那种没有 acc, 就按 res 现聚合一个补上。"""
    rows = {}
    agg = defaultdict(lambda: [0, 0])
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        pid = d["problem_id"]
        if "acc" not in d:
            agg[pid][0] += int(d.get("res", 0))
            agg[pid][1] += 1
        rows.setdefault(pid, d)
    for pid, (nc, nt) in agg.items():
        rows[pid]["acc"] = nc / nt if nt else 0.0
        rows[pid]["_n"] = f"{nc}/{nt}"
    return rows


def load_evid_pairs(path):
    """读 sample_evidrelated.jsonl(每题两行), 按 problem_id 归成 {pid: {view: 行}}。"""
    pairs = defaultdict(dict)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        v = d.get("view")
        if v in ("dense", "sparse"):
            pairs[d["problem_id"]][v] = d
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="挑最终黄金样本对: sparse 视图 slow 相比 direct 正确率提升的题(sparse+dense 均保留)。")
    ap.add_argument("--slow-valid", default=DEFAULT_SLOW,
                    help="sparse 视图慢思考的结果(answer_extract_valid.jsonl, 每题一行含 acc)")
    ap.add_argument("--evid", default=DEFAULT_EVID,
                    help="证据相关样本对 jsonl(每题两行, 这里只取 dense 视图的视频路径)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="最终样本对输出 jsonl")
    ap.add_argument("--improve-threshold", type=float, default=0.125,
                    help="保留条件: slow_acc - sparse_direct_acc 需 > 此值(默认 0, 严格提升)")
    args = ap.parse_args()

    print(f"[load] slow ={args.slow_valid}", file=sys.stderr)
    slow = load_valid(args.slow_valid)
    print(f"[load] evid ={args.evid}", file=sys.stderr)
    pairs = load_evid_pairs(args.evid)
    print(f"[load] slow 题数={len(slow)}  evid 题数={len(pairs)}", file=sys.stderr)

    # ① 先按 slow_gain 判定哪些 problem_id 入选
    keep = {}                   # pid -> sparse_slow_acc
    no_evid = no_baseline = 0
    cnt = Counter()
    by_src = defaultdict(Counter)
    considered = 0
    for pid, s_row in slow.items():
        pair = pairs.get(pid)
        if pair is None or "dense" not in pair or "sparse" not in pair:
            no_evid += 1
            continue
        # direct 基线取 slow 行里透传的 sparse_acc, 缺失时退回 evid 那两行去取
        sparse_direct_acc = s_row.get("sparse_acc", pair["sparse"].get("acc"))
        if sparse_direct_acc is None:
            no_baseline += 1
            continue
        considered += 1

        slow_acc = float(s_row["acc"])
        slow_gain = slow_acc - float(sparse_direct_acc)
        if not (slow_gain > args.improve_threshold):
            continue

        keep[pid] = slow_acc
        case = s_row.get("evid_case") or pair["sparse"].get("evid_case")
        cnt[case] += 1
        by_src[case][str(s_row.get("data_source", "")).split("/")[0]] += 1

    # ② 再把 evid 文件从头过一遍, 命中的行原样拷出, 只补 sparse_slow_acc。
    #    走原文件顺序而不是自己排序:保证 dense/sparse 两行仍然相邻, 且与 evid 完全对齐。
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_line = 0
    gains = Counter()
    with open(args.evid) as fin, open(args.out, "w") as fo:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = d.get("problem_id")
            if pid not in keep:
                continue
            d["sparse_slow_acc"] = round(keep[pid], 4)
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_line += 1
            if d.get("view") == "sparse":       # 每题只统计一次
                gains[round(keep[pid] - float(d.get("sparse_acc", 0)), 2)] += 1

    # 统计
    n1 = cnt["sparse2dense"]
    n2 = cnt["dense2sparse"]
    print(f"\n[stat] slow 题数={len(slow)}  参与比较={considered}  无evid配对={no_evid}  无direct基线={no_baseline}")
    print(f"[stat] 保留条件: slow_acc - sparse_direct_acc > {args.improve_threshold}")
    print(f"[stat] sparse2dense(原稀疏->裁稠密) 命中: {n1}")
    print(f"[stat] dense2sparse(原稠密->拼稀疏) 命中: {n2}")
    print(f"[stat] 合计保留(黄金样本对): {n1 + n2} / {considered} 题 -> 写出 {n_line} 行(每题两行) -> {args.out}")
    for case in ("sparse2dense", "dense2sparse"):
        if by_src[case]:
            print(f"[stat] {case} 分源: {dict(by_src[case])}")

    # slow_gain 分布(= sparse_slow_acc - sparse_acc, 每题算一次)
    if gains:
        print(f"[stat] slow_gain 分布: {dict(sorted(gains.items()))}")


if __name__ == "__main__":
    main()
