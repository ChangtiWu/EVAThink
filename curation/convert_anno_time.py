#!/usr/bin/env python3
"""Convert the evidence annotations from seconds back to frame indices of the 32-frame view.

The annotator (Qwen3.5-27B, prompt from recurate_anno_time_prompt.py) sees the real timestamps of the 32 frames and
answers with {status, start_time, end_time, confidence} in seconds. build_sample_sparse2dense.py cuts clips by frame
index, so each answer is mapped with the video's real frame times (meta.json):

    start_frame = last  i with t_i <= start_time
    end_frame   = first i with t_i >= end_time
(the model sees one timestamp per pair of frames, so this keeps both frames of a pair inside the segment).

The converted JSON replaces `response` (the format build_sample_sparse2dense.py parses); the raw answer is kept in
`response_raw`.

    python convert_anno_time.py --input anno_gt_raw.jsonl --frames-root $EVATHINK_WORK/frames --output anno_gt.jsonl
"""
import argparse
import bisect
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble_trainset4verl as A  # noqa: E402
from build_sample_sparse2dense import parse_localization  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="annotation rollouts (recurate_infer_exact.py output, n=1)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames-root", required=True)
    a = ap.parse_args()
    st = collections.Counter()
    seen = set()
    seg = collections.Counter()
    with open(a.output, "w") as fo:
        for line in open(a.input):
            d = json.loads(line)
            if d["problem_id"] in seen:
                continue
            seen.add(d["problem_id"])
            ans = parse_localization(d.get("response")) or {}
            m = json.load(open(os.path.join(a.frames_root, f"{A.vid_key(d['video'])}__32f", "meta.json")))
            ts = [i / m["src_fps"] for i in m["frames_indices"]]
            status = str(ans.get("status", "")).strip().lower()
            try:
                t0, t1 = float(ans["start_time"]), float(ans["end_time"])
            except (KeyError, TypeError, ValueError):
                t0 = t1 = None
            conv = {"status": status or "unparseable", "confidence": ans.get("confidence")}
            if t0 is not None and status in ("localized", "partial"):
                if t0 > t1:
                    t0, t1 = t1, t0
                s = max(0, bisect.bisect_right(ts, t0 + 1e-6) - 1)
                e = min(len(ts) - 1, bisect.bisect_left(ts, t1 - 1e-6))
                conv.update(start_frame=s, end_frame=max(s, e), start_time=t0, end_time=t1)
                seg[min(e - s + 1, 33)] += 1
            elif status in ("localized", "partial"):
                conv["status"] = "unparseable"
            st[conv["status"]] += 1
            d["response_raw"] = d.get("response")
            d["response"] = json.dumps(conv, ensure_ascii=False)
            d["anno"] = conv
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[anno] {len(seen)} questions, status {dict(st)} -> {a.output}")
    print(f"[anno] segment length in frames {dict(sorted(seg.items()))}")


if __name__ == "__main__":
    main()
