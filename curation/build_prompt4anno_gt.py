#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 sparse 视角的题构造证据定位(evidence localization)prompt。

输入: rollout_direct_valid.jsonl(eval_rollout_acc_rule.py 的产出, 每题一行, 带 acc / view / video)。
    它没有题干/选项/GT, 用 --gt(datapool_mc.jsonl)按 problem_id 回连 problem / options / answer。
    若输入本身带 problem/options/answer(datapool 格式), 则不需要 --gt。

只处理 view == "sparse" 的题(view 由 eval_rollout_acc_rule.py 按 --view-threshold 打好:
acc < 阈值 -> sparse, 否则 dense); 没有 view 字段的行不过滤。

输出: jsonl, 每行
    {"problem_id", "data_source", "video", "acc", "view", "prompt"}
prompt 要求按 32 帧空间输出证据帧段; 之后由 recurate_anno_time_prompt.py 改成按真实时间输出秒数。

用法:
    python build_prompt4anno_gt.py --input rollout_direct_valid.jsonl --gt datapool_mc.jsonl --output prompt4anno_gt.jsonl
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evathink_paths import WORK, VIDEO_ROOT, QWEN3VL  # noqa: E402
ROOT = WORK
DEFAULT_INPUT = f"{ROOT}/outputs/qwen3vl-8b/rollout_direct_valid.jsonl"
DEFAULT_OUTPUT = f"{ROOT}/outputs/qwen3vl-8b/prompt4anno_gt.jsonl"
DEFAULT_GT = f"{ROOT}/datasets/origin_dataset/datapool_mc.jsonl"
# videos 若是相对路径, 拼到这个根目录下得到绝对路径
DEFAULT_VIDEO_ROOT = VIDEO_ROOT

# 证据定位 prompt 模板。注意: 末尾的返回 JSON 示例里带有花括号, 因此这里
# 不能用 str.format(会与花括号冲突), 改用占位符 __QUESTION__/__OPTIONS__/__GT__ 替换。
PROMPT_TEMPLATE = """You are a video evidence localizer.

You will be given a video constructed from exactly 32 frames uniformly sampled from the original video and then arranged in chronological order.
The frame ids are defined by their order in this 32-frame video:

- The first sampled frame is Frame 0.
- The last sampled frame is Frame 31.

Your task is not to answer the question.
Your task is to locate the sampled frame range that contains the visual evidence needed to support the given ground-truth answer.

Question:
__QUESTION__

Options:
__OPTIONS__

Ground-truth answer:
__GT__

Rules:
1. Output the minimal sampled frame range that visually supports the ground-truth answer.
2. If the required evidence appears in multiple separated moments, output the smallest continuous range covering all required evidence.
3. Use frame ids (start from 0), not estimated timestamps.
4. Do not include irrelevant context unless it is needed to answer the question.
5. If the evidence is not visible in the provided frames, output status = "not_localized"; If the evidence is ambiguous or only partially visible, output status = "partial".
6. Both start_frame and end_frame are included in the selected range.

Return only valid JSON in the following format:

{
"status": "localized | partial | not_localized",
"start_frame": int,
"end_frame": int,
"confidence": "high | medium | low"
}"""


def clean_question(problem):
    """从 problem 中提取干净的问题文本: 去掉开头的 <video>/<image> 标签与结尾的 'Options:'。"""
    q = problem or ""
    q = re.sub(r"^\s*<(?:video|image)>\s*", "", q)
    q = re.sub(r"\s*Options?\s*:\s*$", "", q, flags=re.IGNORECASE)
    return q.strip()


def clean_answer(answer):
    """剥去 <answer>...</answer> 包裹, 返回内部文本。"""
    a = answer or ""
    m = re.search(r"<answer>(.*?)</answer>", a, flags=re.DOTALL)
    return (m.group(1) if m else a).strip()


def format_gt(gt_text, options):
    """若 gt 是选项字母(如 'B'), 展开为完整选项行('B. try to get her into water.'),
    便于定位器理解要为哪个答案找证据; 否则原样返回。"""
    key = gt_text.strip().rstrip(".").upper()
    for opt in options or []:
        head = str(opt).split(".", 1)[0].strip().upper()
        if key and key == head:
            return str(opt).strip()
    return gt_text


def to_abs(path, video_root):
    """视频路径归一: 已是绝对路径就原样返回; 相对路径拼到 video_root 下。"""
    r = (path or "").strip()
    if r.startswith("/"):
        return r
    while r.startswith("./"):
        r = r[2:]
    return os.path.normpath(os.path.join(video_root, r.lstrip("/")))


def build_prompt(entry):
    """用模板 + problem/options/answer 封装 prompt。"""
    question = clean_question(entry.get("problem", ""))
    options = entry.get("options") or []
    options_block = "\n".join(str(o) for o in options)
    gt = format_gt(clean_answer(entry.get("answer", "")), options)
    return (PROMPT_TEMPLATE
            .replace("__QUESTION__", question)
            .replace("__OPTIONS__", options_block)
            .replace("__GT__", gt))


def load_rows(path):
    """读 jsonl 或 JSON 数组, 统一返回 list[dict]。"""
    with open(path) as f:
        head = f.read(2048).lstrip()
        f.seek(0)
        if head.startswith("["):
            return json.load(f)
        return [json.loads(l) for l in f if l.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="封装视频证据定位 prompt 为 video_infer 用法B 的 jsonl。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="有效题 jsonl(含 problem_id/video/acc), 或老式的 datapool 标注")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 jsonl 路径")
    p.add_argument("--gt", default=DEFAULT_GT,
                   help="题干/选项/答案的来源(datapool jsonl 或 JSON 数组), 按 problem_id 回连; "
                        "输入自带 problem 字段时可传 '' 关掉")
    p.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT,
                   help="相对视频路径拼接的根目录(视频已是绝对路径时用不到)")
    p.add_argument("--only-mc", action="store_true",
                   help="仅保留 problem_type=='multiple choice' 的条目(GT 里带该字段时才生效)")
    p.add_argument("--check-exists", action="store_true",
                   help="跳过绝对路径不存在的视频(会对每条 stat, 挂载盘上较慢)")
    p.add_argument("--limit", type=int, default=0, help="仅处理前 N 条(调试用, 0=全部)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"[info] 读取输入: {args.input}", flush=True)
    data = load_rows(args.input)
    print(f"[info] 共 {len(data)} 条", flush=True)

    gt_map = {}
    if args.gt:
        gt_rows = load_rows(args.gt)
        gt_map = {d.get("problem_id"): d for d in gt_rows}
        print(f"[info] GT {len(gt_map)} 题 <- {args.gt}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    n_written = 0
    skip_view = skip_no_gt = skip_not_video = skip_no_mc = 0
    skip_no_video_path = skip_missing = skip_no_answer = 0

    with open(args.output, "w") as out:
        for row in data:
            # ── ① 只处理 view=="sparse" 的题(直答正确率低, 需要先定位证据帧)──
            #   view 由 eval_rollout_acc_rule.py 按 --view-threshold 打好; 没有 view 字段的行不过滤。
            acc = row.get("acc")
            view = row.get("view")
            if view is not None and view != "sparse":
                skip_view += 1
                continue

            # ── ② 题干/选项/答案:输入自带就用自己的, 否则按 problem_id 回连 GT ──
            pid = row.get("problem_id")
            src = row if "problem" in row else gt_map.get(pid)
            if src is None:
                skip_no_gt += 1
                continue

            if src.get("data_type", "video") != "video":
                skip_not_video += 1
                continue
            if args.only_mc and src.get("problem_type", "multiple choice") != "multiple choice":
                skip_no_mc += 1
                continue
            if not clean_answer(src.get("answer", "")):
                skip_no_answer += 1
                continue

            # ── ③ 视频路径:优先用输入里那份(已归一化过), 没有再从 GT 的 videos 取 ──
            video = row.get("video") or (src.get("videos") or [None])[0]
            if not video:
                skip_no_video_path += 1
                continue
            abs_path = to_abs(video, args.video_root)
            if args.check_exists and not os.path.isfile(abs_path):
                skip_missing += 1
                continue

            rec = {
                "problem_id": pid,
                "data_source": row.get("data_source") or src.get("data_source"),
                "video": abs_path,
                "prompt": build_prompt(src),
            }
            if acc is not None:
                rec["acc"] = acc
            if view is not None:
                rec["view"] = view
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            if args.limit and n_written >= args.limit:
                break

    print("=" * 56)
    print(f"[done] 写出 {n_written} 条 -> {args.output}")
    print(f"  跳过(view != sparse)={skip_view}  GT里找不到problem_id={skip_no_gt}")
    print(f"  跳过: 非video={skip_not_video}  无videos路径={skip_no_video_path}  "
          f"无answer={skip_no_answer}")
    if args.only_mc:
        print(f"  跳过(非多选题)={skip_no_mc}")
    if args.check_exists:
        print(f"  跳过(视频文件不存在)={skip_missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
