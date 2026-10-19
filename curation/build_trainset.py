#!/usr/bin/env python3
"""Build the two verl training sets from the curated gold pairs (sample_final.jsonl).

    --mode strict  Stage 1: pairs with dense_acc >= 0.75 and sparse_slow_acc - sparse_acc >= 0.5
    --mode rest    Stage 2: all gold pairs minus the problem_ids in --exclude (the Stage 1 set)

Rows are assembled by assemble_trainset4verl.main (question/options cut from each row's curation prompt, GT answer
from the datapool, the response-mode instruction POSTAMBLE, ts_map.json with the real frame times), but frames are
NOT extracted again: each view reuses the 32 frames that recurate_extract_exact.py extracted and on which the
accuracy signals were measured.

    python build_trainset.py --mode strict --final $W/full/sample_final.jsonl --source datapool_mc.jsonl \
        --frames-root $W/frames --out $W/datasets/stage1
    python build_trainset.py --mode rest --final $W/full/sample_final.jsonl --source datapool_mc.jsonl \
        --frames-root $W/frames --exclude $W/datasets/stage1/sample_strict.jsonl --out $W/datasets/stage2
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble_trainset4verl as A  # noqa: E402

META_KEYS = ("src_fps", "src_total_frames", "src_duration", "frames_indices")
FRAME_ROOTS = []


def _work(job):
    """Replacement for A._work: read the already extracted frames (first frame root that has them)."""
    mp4, n, _, _ = job
    d = f"{A.vid_key(mp4)}__{n}f"
    for fr in FRAME_ROOTS:
        try:
            m = json.load(open(f"{fr}/{d}/meta.json"))
        except Exception:  # noqa: BLE001  not in this root, try the next one
            continue
        if len(m["frames_indices"]) != n or not all(os.path.exists(f"{fr}/{d}/{k:03d}.jpg") for k in range(n)):
            return mp4, None, None, "incomplete frames"
        return mp4, [f"{fr}/{d}/{k:03d}.jpg" for k in range(n)], {k: m[k] for k in META_KEYS}, None
    return mp4, None, None, "no frames in any frame root"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("strict", "rest"), required=True)
    ap.add_argument("--final", nargs="+", required=True, help="one or more sample_final.jsonl (gold pairs)")
    ap.add_argument("--source", required=True, help="datapool_mc.jsonl (ground-truth answers)")
    ap.add_argument("--frames-root", nargs="+", required=True, help="frame roots written by recurate_extract_exact.py")
    ap.add_argument("--out", required=True, help="output dataset folder")
    ap.add_argument("--exclude", nargs="*", default=[], help="rest mode: jsonl files whose problem_ids are removed")
    ap.add_argument("--dense-pivot", type=float, default=0.75)
    ap.add_argument("--gain-pivot", type=float, default=0.5)
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "rest" and not a.exclude:
        ap.error("--mode rest needs --exclude")
    FRAME_ROOTS.extend(os.path.abspath(f) for f in a.frames_root)
    os.makedirs(a.out, exist_ok=True)

    drop = set()
    for fp in a.exclude:
        drop |= {str(json.loads(line)["problem_id"]) for line in open(fp) if line.strip()}
    rows, seen = [], set()
    for fp in a.final:
        rs = [json.loads(line) for line in open(fp) if line.strip()]
        if a.mode == "strict":
            keep = {str(r["problem_id"]) for r in rs if r["view"] == "dense" and r["dense_acc"] >= a.dense_pivot
                    and r["sparse_slow_acc"] - r["sparse_acc"] >= a.gain_pivot}
        else:
            keep = {str(r["problem_id"]) for r in rs} - drop
        assert not keep & seen, f"{fp} overlaps earlier inputs on {len(keep & seen)} problem_ids"
        seen |= keep
        rows += [r for r in rs if str(r["problem_id"]) in keep]
    sel = os.path.join(a.out, "sample_strict.jsonl" if a.mode == "strict" else "sample_rest.jsonl")
    with open(sel, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[trainset] {a.mode}: {len(seen)} problems / {len(rows)} rows -> {sel}", flush=True)

    A._work = _work
    A.main(["--input", sel, "--source", a.source, "--out", os.path.join(a.out, "train.parquet"),
            "--frames-root", FRAME_ROOTS[0], "--nproc", str(a.nproc)])
    with open(os.path.join(a.out, "postamble.txt"), "w") as f:
        f.write(A.POSTAMBLE)

    import pyarrow.parquet as pq
    trs = pq.read_table(os.path.join(a.out, "train.parquet")).to_pylist()
    per = collections.Counter(r["extra_info"]["problem_id"] for r in trs)
    man = dict(mode=a.mode, source=a.final, exclude=a.exclude, frame_roots=FRAME_ROOTS,
               criterion=(f"dense_acc >= {a.dense_pivot} and sparse_slow_acc - sparse_acc >= {a.gain_pivot}"
                          if a.mode == "strict" else "gold pairs minus --exclude"),
               rows=len(trs), problems=len(per), rows_per_problem=dict(collections.Counter(per.values())),
               cases=dict(collections.Counter(r["extra_info"]["evid_case"] for r in trs if r["extra_info"]["view"] == "dense")))
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[trainset] {a.out}: {len(trs)} rows / {len(per)} problems {man['cases']}")


if __name__ == "__main__":
    main()
