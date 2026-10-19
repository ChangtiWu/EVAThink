#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the direct-answer rollout prompts (step 1 of the curation) from datapool_mc.jsonl.

Input : datapool_mc.jsonl (one question per line; `videos` may be relative to --video-root).
Output: jsonl, one line per question: {problem_id, data_source, video (absolute path), prompt}.

    python build_prompt4rollout_direct.py --input datapool_mc.jsonl --video-root <video root> \
        --output $EVATHINK_WORK/full/prompt4rollout_direct.jsonl
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evathink_paths import WORK, VIDEO_ROOT, QWEN3VL  # noqa: E402
DEFAULT_INPUT = os.path.join(WORK, "datapool_mc.jsonl")
DEFAULT_OUTPUT = os.path.join(WORK, "full", "prompt4rollout_direct.jsonl")

# 直答 prompt 模板(用占位符 __QUESTION__ / __OPTIONS__ 替换)
PROMPT_TEMPLATE = """You are given a video. Please answer the question directly.

Question:
__QUESTION__

__OPTIONS__

Your final answer must be enclosed in <answer></answer>.
- For a multiple-choice question, put only the option letter inside <answer></answer>.

Examples:
<answer>A</answer>"""



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
    """若 gt 是选项字母(如 'B'), 展开为完整选项行; 否则原样返回。"""
    key = gt_text.strip().rstrip(".").upper()
    for opt in options or []:
        head = str(opt).split(".", 1)[0].strip().upper()
        if key and key == head:
            return str(opt).strip()
    return gt_text


def build_prompt(entry):
    """用模板 + problem/options/answer 封装 prompt。"""
    question = clean_question(entry.get("problem", ""))
    options = entry.get("options") or []
    options_block = "\n".join(str(o) for o in options)
    gt = format_gt(clean_answer(entry.get("answer", "")), options)
    return (PROMPT_TEMPLATE
            .replace("__QUESTION__", question)
            .replace("__OPTIONS__", options_block))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="封装视频证据定位 prompt 为 video_infer 用法B 的 jsonl。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="datapool jsonl(视频为绝对路径)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 jsonl 路径")
    p.add_argument("--video-root", default=VIDEO_ROOT,
                   help="relative video paths in --input are joined to this root")
    p.add_argument("--only-mc", action="store_true",
                   help="仅保留 problem_type=='multiple choice' 的条目(默认不过滤, "
                        "因为最终输入通常已只含多选题; 用默认混合文件试跑时建议加上)")
    p.add_argument("--check-exists", action="store_true",
                   help="跳过绝对路径不存在的视频(会对每条 stat, 挂载盘上较慢)")
    p.add_argument("--limit", type=int, default=0, help="仅处理前 N 条(调试用, 0=全部)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"[info] 读取 datapool: {args.input}", flush=True)
    with open(args.input) as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"[info] 共 {len(data)} 条", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    n_written = 0
    skip_not_video = skip_no_mc = skip_no_video_path = skip_missing = skip_no_answer = 0

    with open(args.output, "w") as out:
        for entry in data:
            if entry.get("data_type") != "video":
                skip_not_video += 1
                continue
            if args.only_mc and entry.get("problem_type") != "multiple choice":
                skip_no_mc += 1
                continue
            videos = entry.get("videos") or []
            if not videos:
                skip_no_video_path += 1
                continue
            if not clean_answer(entry.get("answer", "")):
                skip_no_answer += 1
                continue

            abs_path = videos[0]
            if args.video_root and not os.path.isabs(abs_path):
                abs_path = os.path.normpath(os.path.join(args.video_root, abs_path))
            if args.check_exists and not os.path.isfile(abs_path):
                skip_missing += 1
                continue

            rec = {
                "problem_id": entry.get("problem_id"),
                "data_source": entry.get("data_source"),
                "video": abs_path,
                "prompt": build_prompt(entry),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            if args.limit and n_written >= args.limit:
                break

    print("=" * 56)
    print(f"[done] 写出 {n_written} 条 -> {args.output}")
    print(f"  跳过: 非video={skip_not_video}  无videos路径={skip_no_video_path}  "
          f"无answer={skip_no_answer}")
    if args.only_mc:
        print(f"  跳过(非多选题)={skip_no_mc}")
    if args.check_exists:
        print(f"  跳过(视频文件不存在)={skip_missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
