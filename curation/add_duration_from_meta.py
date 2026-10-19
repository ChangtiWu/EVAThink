#!/usr/bin/env python3
"""Add `duration` (seconds) to every row of a valid jsonl, read from the frame folder's meta.json.

build_sample_dense2sparse.py needs the duration of every video to pick a concatenation partner of similar length.
The frames are extracted by recurate_extract_exact.py, whose meta.json stores the true source duration.

    python add_duration_from_meta.py --input rollout_direct_valid_nodur.jsonl --frames-root $EVATHINK_WORK/frames \
        --output rollout_direct_valid.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble_trainset4verl as A  # noqa: E402
from recurate_extract_exact import NF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames-root", required=True)
    a = ap.parse_args()
    n = miss = 0
    cache = {}
    with open(a.output, "w") as fo:
        for line in open(a.input):
            if not line.strip():
                continue
            d = json.loads(line)
            v = d["video"]
            if v not in cache:
                try:
                    m = json.load(open(os.path.join(a.frames_root, f"{A.vid_key(v)}__{NF}f", "meta.json")))
                    cache[v] = float(m["src_duration"])
                except Exception:  # noqa: BLE001  no frames for this video
                    cache[v] = None
            d["duration"] = cache[v]
            miss += d["duration"] is None
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    print(f"[duration] {n} rows, {miss} without frames/meta.json -> {a.output}")


if __name__ == "__main__":
    main()
