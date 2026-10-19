#!/usr/bin/env python3
"""Merge the two kinds of counterpart videos into one prompt list for the counterpart direct rollouts.

    sparse2dense (build_sample_sparse2dense.py): clip cut around the evidence  -> the new view is dense
    dense2sparse (build_sample_dense2sparse.py): video concatenated with another -> the new view is sparse

Drops duplicated problem_ids, rows whose path does not carry the counterpart suffix, and missing / truncated videos.

    python merge_contra.py --s2d contra_s2d*.jsonl --d2s contra_d2s*.jsonl --suffix _contra_qwen_new \
        --output prompt4rollout_direct_contra.jsonl
"""
import argparse
import collections
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2d", nargs="+", default=[], help="build_sample_sparse2dense.py outputs")
    ap.add_argument("--d2s", nargs="+", default=[], help="build_sample_dense2sparse.py outputs")
    ap.add_argument("--suffix", default="_contra_qwen_new")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    seen, out = set(), []
    bad = collections.Counter()
    for files, view in ((a.s2d, "dense"), (a.d2s, "sparse")):
        for fp in files:
            for line in open(fp):
                d = json.loads(line)
                d.setdefault("view", view)
                if d["problem_id"] in seen:
                    bad["duplicate problem_id"] += 1
                    continue
                if a.suffix not in d["video"]:
                    bad["suffix missing"] += 1
                    continue
                if not os.path.exists(d["video"]) or os.path.getsize(d["video"]) < 2048:
                    bad["video missing or truncated"] += 1
                    continue
                seen.add(d["problem_id"])
                out.append(d)
    with open(a.output, "w") as fo:
        for d in out:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[contra] {len(out)} rows {dict(collections.Counter(d['view'] for d in out))}, dropped {dict(bad)} -> {a.output}")


if __name__ == "__main__":
    main()
