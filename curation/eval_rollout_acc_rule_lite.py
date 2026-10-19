#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估对比样本 / 慢思考 rollout 的正确率(lite 版), 给每条加 res(正确 1 / 错误 0)。

与 eval_rollout_acc_rule.py 的区别:
  · 不按阈值现算 view, 沿用记录里构造时就定好的 view(dense2sparse 出 sparse, sparse2dense 出 dense);
  · 多打一张按 view 分组的正确率表;
  · --effective-threshold 默认 2, 即导出时不按正确率过滤(全部题目导出)。

── response 选项解析规则 ──
  ① 回复里出现 answer/Answer(大小写不限, 带不带 <> 都算) -> 取它之后的第一个大写选项字母。
  ② 没出现 answer -> 取整个回复里第一个大写选项字母。
  只认大写, 且字母必须落在本题选项范围内(由 GT 的 options 决定); 默认还要求该字母是独立 token
  (前后不是字母), 避免 "Both" 被当成 B; 加 --any-letter 可退回"第一个大写字母"。

用法:
  python eval_rollout_acc_rule_lite.py -i rollout.jsonl -g datapool_mc.jsonl -o eval.jsonl -e valid.jsonl
"""

import sys
import argparse
import json
import os
import re
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evathink_paths import WORK, VIDEO_ROOT, QWEN3VL  # noqa: E402
DEFAULT_INPUT = os.path.join(WORK, "full", "rollout_direct.jsonl")
DEFAULT_GT = os.path.join(WORK, "datapool_mc.jsonl")

# 挑「有效题」时要从条目里去掉的 per-rollout 字段(用 acc 汇总取代)
ROLLOUT_FIELDS = ("rollout_idx", "response", "prompt_tokens", "res")

DEFAULT_LETTERS = "ABCD"        # options 缺失/解析不出时的兜底范围


def norm_answer(text):
    """GT 的 answer 归一化: 剥去 <answer> 标签、抽出选项字母、大写。

    GT 是标注好的干净字段(形如 <answer>B</answer>), 所以这里保持宽松匹配;
    response 的解析走 parse_choice, 规则严格得多, 两者不共用。
    """
    t = re.sub(r"</?answer>", "", str(text or ""), flags=re.I).strip()
    m = (re.fullmatch(r"\(?\s*([A-Za-z])\s*[).:]?", t)                      # 整体就是一个字母
         or re.search(r"(?:answer|option|答案|选项|选)\W{0,8}?([A-Ha-h])\b", t, re.I)  # answer is B
         or re.match(r"\s*\(?([A-Ha-h])[).:\s]", t)                        # 开头 "B." / "(B)"
         or re.search(r"\b([A-Ha-h])\b", t))                              # 兜底: 第一个孤立字母
    return m.group(1).upper() if m else t.upper()


def option_letters(options):
    """从 options 列表推出本题的合法选项字母, 如 ["A. 猫","B. 狗"] -> "AB"。

    优先读每个选项开头的字母(有的题目字母不从 A 连排); 读不出来就按个数
    退化成 A、B、C… 连排。返回大写字母串。
    """
    if not options:
        return DEFAULT_LETTERS
    letters = []
    for o in options:
        m = re.match(r"\s*\(?\s*([A-Z])\s*[.)、:,]", str(o))
        if m:
            letters.append(m.group(1))
    if len(letters) == len(options) and len(set(letters)) == len(letters):
        return "".join(letters)
    return "".join(chr(ord("A") + i) for i in range(min(len(options), 26)))


def parse_choice(text, letters, standalone=True):
    """从 response 抽选项字母; 抽不出返回 None。

    letters: 本题合法选项字母(如 "ABCD"), 只在这个范围里找, 且【严格大写】。
    standalone: True 时要求字母前后不是字母(把 Both / Also 这种词首大写排除掉)。
    """
    t = str(text or "")
    cls = "[" + re.escape(letters) + "]"
    pat = rf"(?<![A-Za-z]){cls}(?![A-Za-z])" if standalone else cls

    # ① 出现 answer/Answer(大小写不限, 带不带 <> 都算): 从它后面开始找
    m = re.search(r"answer", t, re.I)
    if m:
        hit = re.search(pat, t[m.end():])
        if hit:
            return hit.group(0)
        # answer 后面没有合法选项字母 -> 落回全文找(比直接判空更能捞回一些)

    # ② 没出现 answer, 或 answer 后面没找到: 整个回复里第一个大写选项字母
    hit = re.search(pat, t)
    return hit.group(0) if hit else None


def dump_effective(prob, prob_base, out_path, threshold):
    """挑出 acc < threshold 的题, 去掉 per-rollout 字段、加 acc, 每题一条写出。

    threshold 默认 2(--effective-threshold): acc 最大为 1, 所以默认全部导出; 传 1 只留非全对的题。
    view 沿用记录里已有的字段。返回 (写出条数, 全错题数, Counter{view: 题数})。
    """
    n = n_zero = 0
    views = Counter()
    with open(out_path, "w") as f:
        for key, (correct, total) in prob.items():
            acc = correct / total
            if acc < threshold:
                rec = dict(prob_base[key], acc=round(acc, 4))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if correct == 0:
                    n_zero += 1
                views[rec.get("view", "(无 view)")] += 1
    return n, n_zero, views


def load_gt(path):
    """GT 支持两种格式: datapool 那样的 jsonl(一行一条), 或 json 数组。
    返回 {problem_id: (答案字母, 本题合法选项字母)}。"""
    with open(path) as f:
        head = f.read(2048).lstrip()
        f.seek(0)
        if head.startswith("["):
            rows = json.load(f)
        else:
            rows = [json.loads(l) for l in f if l.strip()]
    return {d["problem_id"]: (norm_answer(d.get("answer")),
                              option_letters(d.get("options")))
            for d in rows}


def main():
    ap = argparse.ArgumentParser(description="评估 rollout 多选题正确率, 每条加 res(1/0)。")
    ap.add_argument("-i", "--input", default=DEFAULT_INPUT, help="rollout 结果 jsonl")
    ap.add_argument("-g", "--gt", default=DEFAULT_GT,
                    help="GT 文件(jsonl 或 json 数组, 含 problem_id/answer/options)")
    ap.add_argument("-o", "--output", default=None, help="输出 jsonl(默认 input 加 _eval)")
    ap.add_argument("-e", "--effective-output", default=None,
                    help="导出 jsonl(默认 input 加 _valid);写哪些题由 --effective-threshold 决定")
    ap.add_argument("--effective-threshold", type=float, default=2.0,
                    help="只导出 acc < 该值的题。★默认 2 = 不按正确率过滤(acc 上限是 1), "
                         "全部题目都写出;传 1 才是老口径(只要非全对), 传 0.5 只留难题")
    ap.add_argument("--any-letter", action="store_true",
                    help="不要求选项字母是独立 token(Both 里的 B 也会被当成选项)")
    ap.add_argument("--dump-unparsed", default=None,
                    help="把解析不出选项的 response 抽样写到这个文件, 便于回看规则漏了什么")
    args = ap.parse_args()
    out_path = args.output or os.path.splitext(args.input)[0] + "_eval.jsonl"
    eff_path = args.effective_output or os.path.splitext(args.input)[0] + "_valid.jsonl"
    standalone = not args.any_letter

    gt = load_gt(args.gt)
    print(f"[info] GT {len(gt):,} 题 <- {args.gt}")
    nopt = Counter(len(v[1]) for v in gt.values())
    print(f"[info] 选项数分布: {dict(sorted(nopt.items()))}")

    n = n_correct = n_no_gt = n_unparsed = 0
    by_src = defaultdict(lambda: [0, 0])          # data_source -> [correct, total]
    by_view = defaultdict(lambda: [0, 0])         # view -> [correct, total]
    prob = defaultdict(lambda: [0, 0])            # (data_source, problem_id) -> [correct, total]
    prob_base = {}                                # (data_source, problem_id) -> 去掉 per-rollout 字段的题目记录
    unparsed_samples = []
    with open(args.input) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            g = gt.get(d.get("problem_id"))
            if g is None:
                d["res"] = 0
                n_no_gt += 1
            else:
                ans, letters = g
                pick = parse_choice(d.get("response"), letters, standalone)
                if pick is None:
                    n_unparsed += 1
                    if len(unparsed_samples) < 200:
                        unparsed_samples.append(d)
                d["pred"] = pick                  # 记下抽到的选项, 便于回查误判
                d["res"] = int(pick is not None and pick == ans)
                n += 1
                n_correct += d["res"]
                src = d.get("data_source")
                key = (src, d.get("problem_id"))
                by_src[src][1] += 1
                by_src[src][0] += d["res"]
                _v = d.get("view", "(无 view)")
                by_view[_v][1] += 1
                by_view[_v][0] += d["res"]
                prob[key][1] += 1
                prob[key][0] += d["res"]
                prob_base.setdefault(key, {k: v for k, v in d.items()
                                           if k not in ROLLOUT_FIELDS and k != "pred"})
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"已写出 -> {out_path}")
    print(f"可评估 {n} 条 | 正确 {n_correct} | 正确率 {n_correct / n * 100:.2f}%" if n else "无可评估条目")
    if n_no_gt:
        print(f"⚠ {n_no_gt} 条在 GT 中找不到 problem_id (res 记 0)")
    if n_unparsed:
        print(f"⚠ {n_unparsed} 条({n_unparsed / n * 100:.2f}%)解析不出选项字母, 一律判错")
    if args.dump_unparsed and unparsed_samples:
        with open(args.dump_unparsed, "w") as f:
            for d in unparsed_samples:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  解析失败样例 {len(unparsed_samples)} 条 -> {args.dump_unparsed}")

    # 按 view 分组的正确率
    print("=" * 56)
    print("按 view 分组的样本级正确率(★对比实验主指标):")
    for v, (c, t) in sorted(by_view.items(), key=lambda x: -x[1][1]):
        print(f"  {v:<12} {c / t * 100:5.2f}%  ({c}/{t})")

    # 各 data_source 总体正确率
    print("=" * 56)
    print("各 data_source 总体正确率:")
    for src, (c, t) in sorted(by_src.items(), key=lambda x: -x[1][1]):
        print(f"  {src:<38} {c / t * 100:5.1f}%  ({c}/{t})")

    # 各 data_source 下, 按题聚合的「每题 rollout 正确数」分布占比
    dist = defaultdict(Counter)                   # src -> Counter{"c/t": 题数}
    nprob = Counter()
    for (src, _pid), (c, t) in prob.items():
        dist[src][f"{c}/{t}"] += 1
        nprob[src] += 1
    print("=" * 56)
    print("各 data_source 下 每题正确数(对/总) 的题目占比:")
    for src in sorted(dist, key=lambda s: -nprob[s]):
        print(f"[{src}]  共 {nprob[src]} 题")
        for label, k in sorted(dist[src].items(), key=lambda x: int(x[0].split("/")[0])):
            print(f"    {label:>5} 正确: {k:>5} 题  ({k / nprob[src] * 100:5.1f}%)")

    # 总体(所有 data_source 合并)每题正确数分布
    total_dist = Counter()
    for (_src, _pid), (c, t) in prob.items():
        total_dist[f"{c}/{t}"] += 1
    n_total = sum(total_dist.values())
    print("=" * 56)
    print(f"总体 每题正确数(对/总) 的题目占比:  共 {n_total} 题")
    for label, k in sorted(total_dist.items(), key=lambda x: int(x[0].split("/")[0])):
        print(f"    {label:>5} 正确: {k:>5} 题  ({k / n_total * 100:5.1f}%)")

    # 挑出有效题(只要不是全对), 每题一条 + acc, 写新文件
    n_eff, n_zero, views = dump_effective(prob, prob_base, eff_path, args.effective_threshold)
    print("=" * 56)
    _thr = args.effective_threshold
    _desc = "不卡正确率, 全部导出" if _thr > 1 else f"acc < {_thr}"
    print(f"导出({_desc}) {n_eff} / {n_total} 题 ({n_eff / n_total * 100:.1f}%) -> {eff_path}")
    # 三档分开报: 默认阈值下 n_eff 里也含全对的题
    n_full = sum(1 for (c, t) in prob.values() if t and c == t)
    n_mid = n_total - n_zero - n_full
    print(f"  全对(acc=1) {n_full} 题 ({n_full / n_total * 100:.1f}%) | "
          f"全错(acc=0) {n_zero} 题 ({n_zero / n_total * 100:.1f}%) | "
          f"中间 {n_mid} 题 ({n_mid / n_total * 100:.1f}%)")
    print(f"  有效题的 view 构成(沿用记录字段): "
          + ", ".join(f"{k} {v} 题({v / n_eff * 100:.1f}%)" for k, v in views.most_common())
          if n_eff else "  view 构成: 无有效题")


if __name__ == "__main__":
    main()
