# -*- coding: utf-8 -*-
"""EVAThink 奖励: 总分 R = 格式奖励 + 答案奖励(无长度项、无组内统计, 每条 rollout 的分数只由它自己决定)。

    思考判定: 回复以 <answer> 开头 → 直答; 否则(格式合格时必为 Alright 开头)→ 思考。

    格式奖励  合格 0 / 否则 −1, 四条全中才算合格:
                ① 全文 <answer></answer> 只出现一次(半个截断标签也算违规)
                ② </answer> 之后只允许空白
                ③ <answer></answer> 里正好是该题的一个合法选项字母(严格匹配)
                ④ 开头(去除前导空白后)必须是 <answer> 或 Alright, 大小写严格匹配

    答案奖励  只在 <answer></answer> 里找答案; 格式不合格则跳过(记 0)。默认值(一阶段):
                dense  答对直答 2 | 答对思考 1 | 答错直答  0 | 答错思考 −1
                sparse 答对思考 2 | 答对直答 1 | 答错思考  0 | 答错直答 −1
              二阶段用环境变量改成与视角无关: 答对直答 1.1 | 答对思考 1.0 | 答错 0。

一阶段表两侧镜像对称, 模式压力与对错解耦: 组内(同一 prompt 的 8 条 rollout)
    dense : 答对 2−1 = +1 偏直答, 答错 0−(−1) = +1 偏直答
    sparse: 答对 2−1 = +1 偏思考, 答错 0−(−1) = +1 偏思考
即使答不对, 模式信号也在。需配合配对采样(paired_sampler.py): 同一题的 dense / sparse 两行进同一次更新,
两个方向相反的梯度构成对同一道题的对比, 迫使模型依据视频里的证据分布选模式。

格式罚 −1 而不是 0: 答案奖励最低是 −1, 格式坏记 0 会出现"格式坏优于答错"的倒挂。

evid_alpha(= direct_acc)只随返回值带出供监控, 不参与打分。

可调环境变量
    REWARD_FORMAT_OK / REWARD_FORMAT_BAD          格式合格/不合格(默认 0 / −1)
    REWARD_DENSE_OK_DIRECT / _OK_THINK            dense 答对(默认 +2 / +1)
    REWARD_DENSE_NG_DIRECT / _NG_THINK            dense 答错(默认  0 / −1)
    REWARD_SPARSE_OK_THINK / _OK_DIRECT           sparse 答对(默认 +2 / +1)
    REWARD_SPARSE_NG_THINK / _NG_DIRECT           sparse 答错(默认  0 / −1)
    REWARD_FORMAT_W / REWARD_ANSWER_W             两项权重(默认 1.0)
    REWARD_CHOICE_LETTERS                         合法选项字母域(默认 ABCDEFGH)
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _envf(name: str, default: float) -> float:
    try:
        v = os.environ.get(name, "")
        return float(v) if v != "" else float(default)
    except Exception:  # noqa: BLE001
        return float(default)


# ============================ 格式判据 ============================
_ANS_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)

# 允许的两种开头, 大小写严格匹配
_OPENERS = ("<answer>", "Alright")


def _opener_ok(out: str) -> bool:
    """开头(去前导空白)必须是 <answer> 或 Alright, 其它一律违规。"""
    return (out or "").lstrip().startswith(_OPENERS)


def _classify(gt: Any) -> str:
    s = str(gt).strip()
    if s.startswith("{"):
        try:
            d = json.loads(s)
        except Exception:  # noqa: BLE001
            d = None
        if isinstance(d, dict):
            if any(k in d for k in ("boxes", "BOXES")):
                return "sta"
            if any(k in d for k in ("time", "TIME")):
                return "tg"
    if re.fullmatch(r"(\([A-Ha-h]\)){2,}", s):
        return "choice_multi"
    if re.fullmatch(r"[\(\[]?\s*[A-Ha-h]\s*[\)\]\.]?", s):
        return "choice"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return "numerical"
    return "open"


def _valid_letters(extra_info: Optional[Dict]) -> set:
    opts = (extra_info or {}).get("options")
    if isinstance(opts, (list, tuple)) and opts:
        return {chr(ord("A") + i) for i in range(len(opts))}
    if isinstance(opts, str) and opts.strip():
        block = re.findall(r"(?m)^\s*[\(\[]?([A-Za-z])[\.\)\]]", opts)
        if block:
            return {c.upper() for c in block}
        return {c for c in opts.upper() if "A" <= c <= "Z"}
    return set(os.environ.get("REWARD_CHOICE_LETTERS", "ABCDEFGH").upper())


def _format_ok(out: str, ground_truth: Any = None, extra_info: Optional[Dict] = None) -> bool:
    if not out:
        return False
    if not _opener_ok(out):                      # ④ 开头只能是 <answer> 或 Alright
        return False
    ms = list(_ANS_RE.finditer(out))
    if len(ms) != 1 or len(re.findall(r"</?answer>", out, re.I)) != 2:
        return False
    if out[ms[0].end():].strip():
        return False
    body = ms[0].group(1).strip()
    if _classify(ground_truth) != "choice":
        return bool(body)
    return body.upper() in _valid_letters(extra_info)


# ============================ 答案 / 思考模式 ============================
def _gold_letter(gt: Any) -> Optional[str]:
    m = re.search(r"\b([A-Ha-h])\b", str(gt).strip())
    return m.group(1).upper() if m else None


def _answer_letter(out: str) -> Optional[str]:
    m = None
    for m in _ANS_RE.finditer(out or ""):
        pass
    if m is None:
        return None
    body = m.group(1).strip()
    return body.upper() if re.fullmatch(r"[A-Ha-h]", body) else None


def _is_thinking(out: str) -> bool:
    """回复【以 <answer> 开头】→ 不思考(直答);否则 → 思考。"""
    return not (out or "").lstrip().lower().startswith("<answer>")


def answer_reward(correct: bool, thinking: bool, dense: bool) -> float:
    """分档表(纯函数, 便于离线单测)。默认值:

        dense :  答对直答 +2 | 答对思考 +1 | 答错直答  0 | 答错思考 −1
        sparse:  答对思考 +2 | 答对直答 +1 | 答错思考  0 | 答错直答 −1
    """
    if dense:
        if correct:
            return _envf("REWARD_DENSE_OK_THINK", 1.0) if thinking else _envf("REWARD_DENSE_OK_DIRECT", 2.0)
        return _envf("REWARD_DENSE_NG_THINK", -1.0) if thinking else _envf("REWARD_DENSE_NG_DIRECT", 0.0)
    if correct:
        return _envf("REWARD_SPARSE_OK_THINK", 2.0) if thinking else _envf("REWARD_SPARSE_OK_DIRECT", 1.0)
    return _envf("REWARD_SPARSE_NG_THINK", 0.0) if thinking else _envf("REWARD_SPARSE_NG_DIRECT", -1.0)


_VIEW_WARNED = set()


def _is_dense(extra_info: Dict) -> bool:
    """view 缺失/异常时按 sparse 处理,并对每种异常值只告警一次。"""
    v = str(extra_info.get("view", "")).strip().lower()
    if v in ("dense", "sparse"):
        return v == "dense"
    if v not in _VIEW_WARNED:
        _VIEW_WARNED.add(v)
        logger.warning(f"[reward_stage2_v12] 未知 view={v!r},按 sparse 处理")
    return False


# ============================ 主函数 ============================
def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    """verl 自定义奖励入口。

    所有分支返回完全相同的 key 集合: verl 用第 0 条样本的 keys 索引其余样本。
    """
    extra_info = extra_info or {}
    out = solution_str or ""
    try:
        dense = _is_dense(extra_info)
        fmt_ok = _format_ok(out, ground_truth, extra_info)
        thinking = _is_thinking(out)
        gold = _gold_letter(ground_truth)
        pred = _answer_letter(out) if fmt_ok else None      # 格式不合格则跳过答案奖励
        has_answer = pred is not None and gold is not None
        correct = bool(has_answer and pred == gold)

        ans = answer_reward(correct, thinking, dense) if has_answer else 0.0
        fmt = _envf("REWARD_FORMAT_OK", 0.0) if fmt_ok else _envf("REWARD_FORMAT_BAD", -1.0)
        score = _envf("REWARD_FORMAT_W", 1.0) * fmt + _envf("REWARD_ANSWER_W", 1.0) * ans

        try:
            alpha = float(extra_info.get("direct_acc"))
        except (TypeError, ValueError):
            alpha = -1.0                                     # 仅监控,不参与打分

        return {
            "score": round(float(score), 6),
            "acc": float(correct),
            "answer_reward": float(ans),
            "format_reward": float(fmt),
            "evid_thinking": float(thinking),      # 1=思考 0=直答
            "evid_correct": float(correct),
            "evid_has_answer": float(has_answer),
            "evid_dense": float(dense),            # 1=dense 0=sparse
            "evid_alpha": float(alpha),            # direct_acc,只记录不打分
        }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[reward_stage2_v12] error ds={data_source}: {type(e).__name__}: {e}")
        return {"score": -1.0, "acc": 0.0, "answer_reward": 0.0, "format_reward": -1.0,
                "evid_thinking": 0.0, "evid_correct": 0.0, "evid_has_answer": 0.0,
                "evid_dense": 0.0, "evid_alpha": -1.0}


# ============================ 兼容 ============================
def install_trainer_hook():
    """no-op: 没有跨样本统计, 不需要 driver 侧 hook(保留以兼容会调用它的运行环境)。"""
    print("[evid_stage2_v12] 单段式奖励,无需 driver hook", file=sys.stderr, flush=True)


# ============================ 自测 ============================
if __name__ == "__main__":
    ok = []
    def t(out, gt, view, want, note=""):
        r = compute_score(solution_str=out, ground_truth=gt, extra_info={"view": view})
        good = abs(r["score"] - want) < 1e-9
        ok.append(good)
        print(f"  {'✓' if good else '✗'} {view:6s} score={r['score']:>5} 期望={want:>5}  "
              f"fmt={r['format_reward']:+.0f} ans={r['answer_reward']:+.0f} "
              f"直答={1-r['evid_thinking']:.0f} 对={r['evid_correct']:.0f}  {note}")
    print("总分 = 格式(0/−1) + 答案   ★v12: 开头必须是 <answer> 或 Alright")
    print(" dense 四象限(思考一律用 Alright 开头):")
    t("<answer>A</answer>", "A", "dense",  2.0, "答对+直答")
    t("Alright, the frames show X. <answer>A</answer>", "A", "dense",  1.0, "答对+思考")
    t("<answer>B</answer>", "A", "dense",  0.0, "答错+直答")
    t("Alright, the frames show X. <answer>B</answer>", "A", "dense", -1.0, "答错+思考")
    print(" sparse 四象限(与 dense 镜像):")
    t("Alright, the frames show X. <answer>A</answer>", "A", "sparse",  2.0, "答对+思考")
    t("<answer>A</answer>", "A", "sparse",  1.0, "答对+直答")
    t("Alright, the frames show X. <answer>B</answer>", "A", "sparse",  0.0, "答错+思考")
    t("<answer>B</answer>", "A", "sparse", -1.0, "答错+直答")
    print(" 格式违规 -> −1(答案跳过):")
    t("<answer>A</answer> trailing", "A", "dense", -1.0, "尾部有内容")
    t("<answer>A</answer><answer>B</answer>", "A", "dense", -1.0, "出现两次")
    t("<answer>A. 选项原文</answer>", "A", "dense", -1.0, "非单字母")
    t("<answer></answer>", "A", "sparse", -1.0, "空块")
    t("no tag at all", "A", "sparse", -1.0, "无标签")
    t("<answer>A", "A", "dense", -1.0, "半个标签")
    print(" ★v12 新增判据④:开头不是 <answer>/Alright 一律违规(这些在 v11 下都能拿分):")
    t("Think. <answer>A</answer>", "A", "dense", -1.0, "开头是 Think.")
    t("The video shows X. <answer>A</answer>", "A", "sparse", -1.0, "直接裸写描述")
    t("Well, <answer>A</answer>", "A", "dense", -1.0, "其它开场白")
    t("alright, lowercase. <answer>A</answer>", "A", "sparse", -1.0, "小写 alright(大小写严格)")
    t("Leaning into direct-answer mode. <answer>A</answer>", "A", "dense", -1.0, "自我宣告模式")
    t("![](https://x.png)\n\nAlright, <answer>A</answer>", "A", "sparse", -1.0, "Alright 前有杂物")
    print(" ★合法开头 + 前导空白应当通过:")
    t("  \n<answer>A</answer>", "A", "dense", 2.0, "前导空白+直答")
    t("\n Alright, X. <answer>A</answer>", "A", "sparse", 2.0, "前导空白+Alright思考")

    print("\n 组内模式净压力(对称性检查):")
    for v, dn in (("dense", True), ("sparse", False)):
        ok_gap = answer_reward(True, False, dn) - answer_reward(True, True, dn)
        ng_gap = answer_reward(False, False, dn) - answer_reward(False, True, dn)
        want = 1.0 if dn else -1.0
        good = abs(ok_gap - want) < 1e-9 and abs(ng_gap - want) < 1e-9
        ok.append(good)
        d = "偏直答" if dn else "偏思考"
        print(f"  {'✓' if good else '✗'} {v:6s} 直答−思考: 答对 {ok_gap:+.0f} / 答错 {ng_gap:+.0f}  (恒 {'+1 '+d if dn else '−1 即 +1 '+d})")

    a = set(compute_score(solution_str="<answer>A</answer>", ground_truth="A", extra_info={"view":"dense"}).keys())
    b = set(compute_score(solution_str=None, ground_truth=None, extra_info=None).keys())
    ok.append(a == b)
    print(f"\n  {'✓' if a==b else '✗'} 正常分支与异常分支 key 集合一致")
    print("自测", "全部通过 ✓" if all(ok) else f"失败 ✗ ({ok.count(False)} 项)")
    sys.exit(0 if all(ok) else 1)
