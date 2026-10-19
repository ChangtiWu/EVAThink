#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""挑出证据相关(evidence-related)样本: 同一题在 dense 视角下比 sparse 视角答得好, 说明作答依赖证据的可见度。

判据: dense_acc > sparse_acc + --margin(默认 0.125)

每个 problem_id 有一个 dense 视角和一个 sparse 视角, 来源取决于原样本的 view:
  原样本 view == sparse -> 对比样本由 sparse2dense 裁出, view == dense
      => dense 来自对比视频, sparse 来自原视频
  原样本 view == dense  -> 对比样本由 dense2sparse 拼接, view == sparse
      => dense 来自原视频, sparse 来自对比视频

输入(每题一行、自带 acc 与 view):
    rollout_direct_valid.jsonl         原视频上的结果(eval_rollout_acc_rule.py, 只含 acc < 1 的题)
    rollout_direct_contra_valid.jsonl  对比视频上的结果(eval_rollout_acc_rule_lite.py)
  也兼容 per-rollout 的 *_eval.jsonl(没有 acc 字段时按 res 聚合)。

输出: 每个命中的 problem_id 写两行(dense 一行、sparse 一行, 相邻, dense 在前), 各带自己的
  video / prompt / view / acc, 另附两行共有的 dense_acc / sparse_acc / gap / evid_case。

用法:
  python select_sample_evidrelated.py --orig-eval rollout_direct_valid.jsonl       --contra-eval rollout_direct_contra_valid.jsonl --out sample_evidrelated.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evathink_paths import WORK, VIDEO_ROOT, QWEN3VL  # noqa: E402
ROOT = WORK
DEFAULT_CONTRA = f"{ROOT}/outputs/qwen3vl-8b/rollout_direct_contra_valid.jsonl"
DEFAULT_ORIG = f"{ROOT}/outputs/qwen3vl-8b/rollout_direct_valid.jsonl"
DEFAULT_OUT = f"{ROOT}/outputs/qwen3vl-8b/sample_evidrelated.jsonl"

# per-rollout 字段(聚合成 acc 后丢弃); prompt 保留, 两个视角的 prompt 不同
DROP_FIELDS = ("rollout_idx", "response", "prompt_tokens", "res")


def load_eval(path):
    """读结果 jsonl, 按 problem_id 归一成 {pid: {"acc", "meta"}}。

    两种输入:
      ① *_valid.jsonl —— 每题一行且自带 acc, 直接用
      ② *_eval.jsonl  —— 每行一个 rollout, 按 res 聚合成 acc
    meta = 该题任一行去掉 per-rollout 字段(元信息, 各 rollout 一致)。
    """
    agg = defaultdict(lambda: [0, 0])   # pid -> [n_correct, n_total]
    direct_acc = {}                     # pid -> 文件里自带的 acc
    meta = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        pid = d["problem_id"]
        if "acc" in d:
            direct_acc[pid] = float(d["acc"])
        else:
            agg[pid][0] += int(d.get("res", 0))
            agg[pid][1] += 1
        if pid not in meta:
            meta[pid] = {k: v for k, v in d.items() if k not in DROP_FIELDS}
    out = {}
    for pid, m in meta.items():
        if pid in direct_acc:
            acc = direct_acc[pid]
        else:
            nc, nt = agg[pid]
            acc = nc / nt if nt else 0.0
        out[pid] = {"acc": acc, "meta": m}
    return out


def main():
    ap = argparse.ArgumentParser(description="挑出稠密视图比稀疏视图答得更好的证据相关样本")
    ap.add_argument("--contra-eval", default=DEFAULT_CONTRA, help="对比样本结果 jsonl(valid 或 eval)")
    ap.add_argument("--orig-eval", default=DEFAULT_ORIG, help="原始样本结果 jsonl(valid 或 eval)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="命中样本输出 jsonl(每题两条)")
    ap.add_argument("--view-field", default="view", help="视图字段名(默认 view)")
    ap.add_argument("--margin", type=float, default=0.125,
                    help="判据 dense_acc > sparse_acc + margin;0=严格大于, 调大只留差距明显的")
    args = ap.parse_args()

    print(f"[load] contra={args.contra_eval}", file=sys.stderr)
    contra = load_eval(args.contra_eval)
    print(f"[load] orig={args.orig_eval}", file=sys.stderr)
    orig = load_eval(args.orig_eval)
    print(f"[load] contra 题数={len(contra)}  orig 题数={len(orig)}", file=sys.stderr)

    matched = missing = unknown_view = view_conflict = 0
    selected = []          # [(dense_rec, sparse_rec)]
    cnt = Counter()
    by_src = Counter()
    gaps = Counter()
    for pid, c in contra.items():
        o = orig.get(pid)
        if o is None:
            missing += 1
            continue
        matched += 1
        cv = c["meta"].get(args.view_field)
        ov = o["meta"].get(args.view_field)
        if cv not in ("dense", "sparse") or ov not in ("dense", "sparse"):
            unknown_view += 1
            continue
        if cv == ov:
            # 一题的两个视图必须一稠一稀; 相同说明上游构造或字段串了, 宁可丢掉也不硬凑
            view_conflict += 1
            continue

        if cv == "dense":                      # 对比样本是 sparse2dense 裁出来的
            dense_side, sparse_side = c, o
            case = "sparse2dense"              # 原样本 sparse -> 裁成 dense
        else:                                  # 对比样本是 dense2sparse 拼长的
            dense_side, sparse_side = o, c
            case = "dense2sparse"              # 原样本 dense -> 拼成 sparse

        dense_acc, sparse_acc = dense_side["acc"], sparse_side["acc"]
        if not (dense_acc > sparse_acc + args.margin):
            continue

        gap = dense_acc - sparse_acc
        cnt[case] += 1
        src = str(c["meta"].get("data_source", "")).split("/")[0]
        by_src[src] += 1
        gaps[round(gap, 2)] += 1

        shared = {"dense_acc": round(dense_acc, 4), "sparse_acc": round(sparse_acc, 4),
                  "gap": round(gap, 4), "evid_case": case}
        # 每题两条: 各带自己的 video / view / acc, 共有字段两条都写一份
        d_rec = dict(dense_side["meta"]);  d_rec["acc"] = round(dense_acc, 4);  d_rec.update(shared)
        s_rec = dict(sparse_side["meta"]); s_rec["acc"] = round(sparse_acc, 4); s_rec.update(shared)
        d_rec[args.view_field] = "dense"
        s_rec[args.view_field] = "sparse"
        selected.append((d_rec, s_rec))

    # 写出:按 case 再按 gap 降序; 同一题的两条【相邻】, dense 在前
    order = {"sparse2dense": 0, "dense2sparse": 1}
    selected.sort(key=lambda pr: (order.get(pr[0]["evid_case"], 9), -pr[0]["gap"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_line = 0
    with open(args.out, "w") as fo:
        for d_rec, s_rec in selected:
            for r in (d_rec, s_rec):
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_line += 1

    n1, n2 = cnt["sparse2dense"], cnt["dense2sparse"]
    print(f"\n[stat] 对比样本题数={len(contra)}  匹配到原样本={matched}  未匹配={missing}  "
          f"未知view={unknown_view}  两边view相同(丢弃)={view_conflict}")
    print(f"[stat] 判据: dense_acc > sparse_acc + {args.margin}")
    print(f"[stat] sparse2dense(原稀疏->裁稠密) 命中: {n1}")
    print(f"[stat] dense2sparse(原稠密->拼稀疏) 命中: {n2}")
    print(f"[stat] 合计命中 {n1 + n2} 题 -> 写出 {n_line} 行(每题两条) -> {args.out}")
    if by_src:
        print(f"[stat] 分源: {dict(by_src.most_common())}")
    if gaps:
        print(f"[stat] gap(dense_acc - sparse_acc) 分布: {dict(sorted(gaps.items()))}")


if __name__ == "__main__":
    main()
