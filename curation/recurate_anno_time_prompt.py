#!/usr/bin/env python3
"""把证据定位 prompt(build_prompt4anno_gt.py 产出, 要求输出 32 帧空间的帧号)改成按真实时间戳输出秒数的版本。

标注模型(Qwen3.5-27B)看到的是 <X.X seconds> 时间戳而不是帧号, 问帧号会有歧义(模型常把秒数当帧号写)。
只改与输出单位相关的句子(帧段 -> 时间段、rule 3、JSON 键名), 题干 / 选项 / GT / 其余规则不动;
得到的 start_time / end_time 由 convert_anno_time.py 用各视频 meta.json 的真实帧时刻换回 32 帧空间帧号,
build_sample_sparse2dense.py 照旧按帧号裁剪。
"""
import argparse, json

REPL = [
    ("The frame ids are defined by their order in this 32-frame video:\n\n"
     "- The first sampled frame is Frame 0.\n- The last sampled frame is Frame 31.\n",
     "Each frame is shown with its real timestamp (in seconds) in the original video.\n"),
    ("Your task is to locate the sampled frame range that contains the visual evidence",
     "Your task is to locate the time range that contains the visual evidence"),
    ("1. Output the minimal sampled frame range that visually supports the ground-truth answer.",
     "1. Output the minimal time range that visually supports the ground-truth answer."),
    ("3. Use frame ids (start from 0), not estimated timestamps.",
     "3. Use the timestamps shown in the video (in seconds), not frame indices."),
    ("6. Both start_frame and end_frame are included in the selected range.",
     "6. Both start_time and end_time are included in the selected range."),
    ('"start_frame": int,\n"end_frame": int,', '"start_time": float,\n"end_time": float,'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    n = 0
    with open(a.output, "w") as fo:
        for l in open(a.input):
            d = json.loads(l); p = d["prompt"]
            for old, new in REPL:
                assert p.count(old) == 1, (d["problem_id"], old[:50])
                p = p.replace(old, new)
            d["prompt"] = p
            fo.write(json.dumps(d, ensure_ascii=False) + "\n"); n += 1
    print(f"[anno-time] {n} 条 -> {a.output}")


if __name__ == "__main__":
    main()
