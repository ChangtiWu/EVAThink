"""首 token 置信度上限正则 —— 防止塌向单一模式的吸收态。

回答的第 1 个 token 决定模式(`<answer>` 直答 / `Alright` 思考)。某输入上 8 条 rollout 一旦全是同一模式,
组内没有对比样本, 奖励给不出模式梯度, 策略就停在那里。

对每条样本取第 1 个 token 的采样概率 p = exp(old_log_probs[:, 0]), p > c 时在第 1 个 token 的优势上扣
    A[i, 0] -= β · max(0, log p − log c)
  · 只动第 1 个 token, 不碰后续 token 的优势; 该用哪个模式仍由奖励决定;
  · 在全同模式的组里也有效(用被采到的样本自己的 p);
  · 平衡点: 若某模式在组内的平均优势为 a, p 稳定在约 c·exp(a/β)。

纯 torch, 不依赖 verl; 由 paired_task_runner 包在 ray_trainer.compute_advantage 外面调用。
"""
from __future__ import annotations

import math


def firsttok_cap_penalty(advantages, old_log_probs, response_mask, cap: float, beta: float):
    """原地修改 advantages[:, 0], 返回 (advantages, penalty[bs], p0[bs])。

    advantages / old_log_probs / response_mask: [bs, response_len], 回答右补齐(第 0 列 = 第 1 个生成 token)。
    """
    import torch

    if not (0.0 < cap < 1.0) or beta <= 0:
        raise ValueError(f"首 token 正则参数非法: cap={cap} beta={beta}")
    lp0 = old_log_probs[:, 0].float()
    valid = response_mask[:, 0].float() if response_mask is not None else torch.ones_like(lp0)
    pen = beta * torch.clamp(lp0 - math.log(cap), min=0.0) * valid
    advantages[:, 0] = advantages[:, 0] - pen.to(advantages.dtype)
    return advantages, pen, lp0.exp()


def firsttok_wrongmode_penalty(advantages, old_log_probs, response_mask, is_dense, is_think, fmt_ok,
                               cap_wrong: float, beta: float, cap_right: float | None = None):
    """单边版: 只压"模式不对"的过度笃定; cap_right 给"模式对的"一个宽松上限(不设则不管)。

按一阶段奖励表(同对错下 dense 直答比思考高, sparse 相反): dense 上思考 / sparse 上直答 = 不对的模式。
    A[i, 0] -= β · max(0, log p − log c_wrong) · [样本 i 是不对的模式] · [格式合法]
  模式对的样本在 cap_right 给定时按同一公式以 cap_right 为上限扣。
  · 不对的模式 p 稳定在约 c_wrong·exp(a/β);
  · 不分视角的单一策略会被夹在直答率 ~0.5 附近, 罚只落在违规视角上 → 梯度方向是"按视角分化"。
is_dense / is_think / fmt_ok: 长度 bs 的 bool(取自奖励输出的 evid_dense / evid_thinking / format_reward);
格式违规样本的视角/模式不可信, 不参与。
返回 (advantages, penalty[bs], p0[bs], wrong[bs])。
    """
    import torch

    if not (0.0 < cap_wrong < 1.0) or beta <= 0 or (cap_right is not None and not (0.0 < cap_right < 1.0)):
        raise ValueError(f"首 token 单边正则参数非法: cap_wrong={cap_wrong} cap_right={cap_right} beta={beta}")
    dev = advantages.device
    dense = torch.as_tensor(is_dense, dtype=torch.bool, device=dev).reshape(-1)
    think = torch.as_tensor(is_think, dtype=torch.bool, device=dev).reshape(-1)
    ok = torch.as_tensor(fmt_ok, dtype=torch.bool, device=dev).reshape(-1)
    if not (dense.numel() == think.numel() == ok.numel() == advantages.shape[0]):
        raise ValueError(f"视角/模式标签长度 {dense.numel()}/{think.numel()}/{ok.numel()} ≠ batch {advantages.shape[0]}")
    is_wrong = (dense & think) | (~dense & ~think)
    wrong, right = is_wrong & ok, ~is_wrong & ok
    lp0 = old_log_probs[:, 0].float()
    valid = response_mask[:, 0].float() if response_mask is not None else torch.ones_like(lp0)
    over = torch.clamp(lp0 - math.log(cap_wrong), min=0.0) * wrong.float()
    if cap_right is not None:
        over = over + torch.clamp(lp0 - math.log(cap_right), min=0.0) * right.float()
    pen = beta * over * valid
    advantages[:, 0] = advantages[:, 0] - pen.to(advantages.dtype)
    return advantages, pen, lp0.exp(), wrong


def summarize_views(pen, p0, is_dense, is_think, cap_wrong: float, cap_right: float | None = None) -> str:
    """按 dense/sparse × 思考/直答 精确计数: 被扣条数/该格条数, 以及该格首 token p 均值。"""
    import torch

    dense = torch.as_tensor(is_dense, dtype=torch.bool, device=pen.device).reshape(-1)
    think = torch.as_tensor(is_think, dtype=torch.bool, device=pen.device).reshape(-1)
    hit = pen > 0
    parts = []
    for vn, vm in (("dense", dense), ("sparse", ~dense)):
        cells = []
        for mn, mm in (("思考", think), ("直答", ~think)):
            m = vm & mm
            n = int(m.sum())
            pm = f"{float(p0[m].mean()):.2f}" if n else "-"
            cells.append(f"{mn} 扣 {int((hit & m).sum())}/{n}(p̄ {pm})")
        parts.append(f"{vn}: " + ", ".join(cells))
    n = int(hit.sum())
    rule = f"不对模式 p>{cap_wrong:g}" + (f" / 对的模式 p>{cap_right:g}" if cap_right is not None else "")
    tail = f"共 {n}/{pen.numel()} 条被扣({rule})"
    if n:
        tail += f", 平均扣 {float(pen[hit].mean()):.3f}, 最大 {float(pen.max()):.3f}"
    return " | ".join(parts) + " | " + tail


def summarize(pen, p0, cap: float) -> str:
    hit = pen > 0
    n = int(hit.sum())
    msg = f"{n}/{pen.numel()} 条首 token p>{cap:g} 被扣"
    if n:
        msg += f", 平均扣 {float(pen[hit].mean()):.3f}, 最大 {float(pen.max()):.3f}"
    msg += f"; 首 token p 均值 {float(p0.mean()):.3f}, p>0.99 的 {int((p0 > 0.99).sum())} 条"
    return msg
