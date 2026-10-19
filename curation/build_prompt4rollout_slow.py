#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为证据相关样本的 sparse 视角构造慢思考(先跨帧整合证据再作答)rollout prompt。

select_sample_evidrelated.py 挑出的是同一题 dense 视角明显比 sparse 视角答得好的样本。本脚本只取其中
view == "sparse" 的那条, 把指令换成慢思考, 用来检验推理能否在 sparse 视角上补回正确率。

题干与选项从样本自带的 direct prompt 里原样切出, 与跑过 rollout 的 prompt 逐字一致。
dense2sparse 样本(拼接视频)的 prompt 里有一句 "The video contains two separate event segments, please answer
the question based only on the {first|second} event segment.", 指明目标事件在前段还是后段, 本脚本把它原样放回
慢思考模板的首句之后。

输入: sample_evidrelated.jsonl(每题两条, 只用 view == "sparse" 的那条)
输出: jsonl, 每行 {problem_id, data_source, video(sparse 视角视频), prompt(慢思考), view="sparse",
      evid_case, dense_acc, sparse_acc, gap}
      sparse2dense 类的 sparse 视频是原视频, dense2sparse 类的是拼接后的对比视频。

用法:
  python build_prompt4rollout_slow.py --input sample_evidrelated.jsonl --output prompt4rollout_slow.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evathink_paths import WORK, VIDEO_ROOT, QWEN3VL  # noqa: E402
ROOT = WORK
DEFAULT_INPUT = f"{ROOT}/outputs/qwen3vl-8b/sample_evidrelated.jsonl"
DEFAULT_OUTPUT = f"{ROOT}/outputs/qwen3vl-8b/prompt4rollout_slow.jsonl"

# 慢思考 prompt 模板。__SEGMENT_HINT__ 只有 dense2sparse 样本才会填(见上)。
PROMPT_TEMPLATE = """You are given a video.__SEGMENT_HINT__

Do not answer immediately. Think step by step before answering.
If the frames are temporally jumpy, the evidence is scattered, or the visible content is not sufficient for an immediate answer, carefully reason over the observed cross-frame clues and integrate the evidence before giving your answer.
Do not invent details that are not supported by the visible frames.

**Note**:
Work through your reasoning first, then give your final answer.
**Your final answer MUST be enclosed in <answer></answer>.**
**For multiple-choice questions, put only the option letter inside the tags, e.g., <answer>A</answer>.**

**An Example**:
Question: What does the person add to the bowl?
Options:
A. Water
B. Milk
C. Flour
D. Oil

The sampled frames are not fully continuous, so I need to connect the visible clues across time. In an earlier frame, the person is holding a carton near the bowl. In a later frame, a white liquid is being poured into the bowl. The final frames show the bowl containing a white liquid. I do not see evidence of flour, oil, or clear water being added. The scattered visual clues together support that the person adds milk.
<answer>B</answer>

===

Now answer the actual question:

Question:
__QUESTION__

Options:
__OPTIONS__"""

# 从 direct prompt 里切题干与选项。direct 模板(build_prompt4rollout_direct.py)长这样:
#   You are given a video. Please answer the question directly.
#   [可选: The video contains two separate event segments, ...]
#   (空行)
#   Question:
#   <题干>
#   (空行)
#   <选项们>            <- 选项之间可能还有空行, 所以这里靠结尾锚点截断
#   (空行)
#   Your final answer must be enclosed in <answer></answer>.
QA_RE = re.compile(r"Question:\n(.*?)\n\n(.*?)\n\nYour final answer must be enclosed",
                   re.DOTALL)
# 两段提示句(ADD_LATTER / ADD_FORMER, 见 build_sample_dense2sparse.py)
HINT_RE = re.compile(r"The video contains two separate event segments,[^\n]*?event segment\.")


def split_direct_prompt(prompt):
    """从 direct prompt 里切出 (题干, 选项块, 两段提示句或 None)。切不出来返回 None。"""
    p = prompt or ""
    m = QA_RE.search(p)
    if not m:
        return None
    question = m.group(1).strip()
    options = m.group(2).strip()
    if not question or not options:
        return None
    h = HINT_RE.search(p)
    return question, options, (h.group(0) if h else None)


def build_prompt(question, options, hint):
    """填慢思考模板;hint 非空时原样放在首句之后(和 direct 里的位置一致)。"""
    return (PROMPT_TEMPLATE
            .replace("__SEGMENT_HINT__", "\n" + hint if hint else "")
            .replace("__QUESTION__", question)
            .replace("__OPTIONS__", options))


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="为证据相关样本的稀疏视图封装慢思考 rollout prompt。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="证据相关样本 jsonl(select_sample_evidrelated.py 产物, 每题两条)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 jsonl 路径")
    p.add_argument("--view", default="sparse", help="只处理该 view 的那条(默认 sparse)")
    p.add_argument("--limit", type=int, default=0, help="仅处理前 N 条(调试用, 0=全部)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"[info] 读取证据相关样本: {args.input}", flush=True)
    samples = load_jsonl(args.input)
    rows = [s for s in samples if s.get("view") == args.view]
    print(f"[info] 共 {len(samples)} 行, 其中 view=={args.view} 的 {len(rows)} 条", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    n_written = n_hint = 0
    skip_unparsed = skip_no_video = 0
    case_cnt = Counter()
    with open(args.output, "w") as out:
        for s in rows:
            parsed = split_direct_prompt(s.get("prompt"))
            if parsed is None:
                skip_unparsed += 1
                continue
            question, options, hint = parsed
            video = s.get("video")
            if not video:
                skip_no_video += 1
                continue

            rec = {
                "problem_id": s.get("problem_id"),
                "data_source": s.get("data_source"),
                "video": video,
                "prompt": build_prompt(question, options, hint),
                "view": args.view,
                "evid_case": s.get("evid_case"),
                "dense_acc": s.get("dense_acc"),
                "sparse_acc": s.get("sparse_acc"),
                "gap": s.get("gap"),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            n_hint += hint is not None
            case_cnt[s.get("evid_case")] += 1
            if args.limit and n_written >= args.limit:
                break

    print("=" * 60)
    print(f"[done] 写出 {n_written} 条 -> {args.output}")
    print(f"  其中带【两段提示句】的 {n_hint} 条(应等于 dense2sparse 的条数)")
    print(f"  evid_case 构成: {dict(case_cnt)}")
    print(f"  跳过: prompt 切不出题干/选项={skip_unparsed}  无视频路径={skip_no_video}")
    if skip_unparsed:
        print("  ⚠ 有切不出来的, 多半是 direct 模板变了 —— 去看 QA_RE 的锚点")
    return 0


if __name__ == "__main__":
    sys.exit(main())
