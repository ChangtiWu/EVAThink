"""配对采样器 —— 保证同一 problem_id 的 dense / sparse 两行落进同一个 mini-batch。

同一道题有 dense 和 sparse 两行, 该用的模式相反(dense 直答、sparse 思考)。两行进入同一次参数更新时,
梯度里同时有这两个相反方向, 模型必须依据视频里的证据分布选模式, 而不能靠记住"这道题该直答"拿满奖励。
verl 默认的 RandomSampler 会把配对的两行打散到不同 batch。

注意: 本采样器不把两行合成一个 GRPO 组。组仍按 prompt 划分, 各自 8 条 rollout, 优势在各自组内计算;
它保证的是两个组落在同一次 optimizer step(本配置 mini_batch == train_batch, 每批一次更新)。

本文件只提供采样器与分组逻辑, 不改 verl; 由 paired_task_runner.py 挂载。

环境变量(都有默认值)
    EVID_PAIR_KEY               分组键(在 extra_info 里找), 默认 problem_id
    EVID_PAIR_GROUP_SIZE        每组期望行数, 默认 2
    EVID_PAIR_SHUFFLE_IN_GROUP  组内两行是否随机先后, 默认 1
    EVID_PAIR_STRICT            1 = 遇到不完整的组就报错; 默认 0 = 丢弃并告警

离线自测(只需 pandas, 不需要 verl):
    python3 paired_sampler.py --selftest <train.parquet> [batch_size]
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Sampler


def log(msg: str) -> None:
    print(f"[pair_sampler] {msg}", file=sys.stderr, flush=True)


def envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:  # noqa: BLE001
        return default


class _PairedIterator:
    """状态放在迭代器上。

    torchdata 的 StatefulDataLoader 是对【sampler 的迭代器】做快照的
    (官方 RandomSampler 类本身没有 state_dict, 状态在 _StatefulRandomSamplerIterator 上),
    所以 state_dict/load_state_dict 必须实现在这层。

    这里不搬运 torch.Generator 的内部状态,而是存 (epoch, yielded):排列完全由
    (seed, epoch) 决定,恢复时重算排列再跳过 yielded 条。比搬 generator 状态更稳,
    也不依赖 torch 版本。
    """

    def __init__(self, sampler: "PairedGroupSampler", epoch: int):
        self.sampler = sampler
        self.epoch = int(epoch)
        self.yielded = 0
        self.flat: List[int] = sampler.flat_order(self.epoch)

    def __iter__(self) -> "_PairedIterator":
        return self

    def __next__(self) -> int:
        if self.yielded >= len(self.flat):
            raise StopIteration
        v = self.flat[self.yielded]
        self.yielded += 1
        return v

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "yielded": self.yielded}

    def load_state_dict(self, state_dict: dict) -> None:
        self.epoch = int(state_dict.get("epoch", 0))
        self.yielded = int(state_dict.get("yielded", 0))
        self.flat = self.sampler.flat_order(self.epoch)
        if self.yielded > len(self.flat):
            log(f"⚠ 恢复的 yielded={self.yielded} 超过本 epoch 长度 {len(self.flat)},截断")
            self.yielded = len(self.flat)


class PairedGroupSampler(Sampler[int]):
    """按组发号:同组的行永远连续发出,组的先后每个 epoch 重新洗牌。

    只要 batch_size % group_size == 0 且 DataLoader 顺序切批(它就是这么做的),
    每个 batch 就恰好由 batch_size/group_size 个【完整的组】构成。
    """

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        shuffle_in_group: bool = True,
    ):
        self.groups: List[List[int]] = [list(g) for g in groups]
        if not self.groups:
            raise ValueError("[pair_sampler] 分组为空")
        self.group_size = len(self.groups[0])
        bad = [i for i, g in enumerate(self.groups) if len(g) != self.group_size]
        if bad:
            raise ValueError(f"[pair_sampler] 组大小不一致,首个异常组下标 {bad[0]}")
        self.batch_size = int(batch_size)
        if self.batch_size % self.group_size != 0:
            raise ValueError(
                f"[pair_sampler] batch_size={self.batch_size} 不能被 group_size="
                f"{self.group_size} 整除,无法保证整组同批。请把 train_batch_size"
                f"(或 gen_batch_size)设成 {self.group_size} 的倍数。"
            )
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.shuffle_in_group = bool(shuffle_in_group)
        self._epoch = 0
        self._last_iter: Optional[_PairedIterator] = None

    def flat_order(self, epoch: int) -> List[int]:
        n = len(self.groups)
        g: Optional[torch.Generator] = None
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed * 1_000_003 + int(epoch))
            order = torch.randperm(n, generator=g).tolist()
        else:
            order = list(range(n))
        flat: List[int] = []
        for gi in order:
            grp = self.groups[gi]
            if self.shuffle_in_group and g is not None and len(grp) > 1:
                perm = torch.randperm(len(grp), generator=g).tolist()
                grp = [grp[i] for i in perm]
            flat.extend(grp)
        return flat

    def __iter__(self) -> Iterator[int]:
        it = _PairedIterator(self, self._epoch)
        self._epoch += 1
        self._last_iter = it
        return it

    def __len__(self) -> int:
        return len(self.groups) * self.group_size

    # 兜底:某些版本可能在 sampler 层面找状态
    def state_dict(self) -> dict:
        return self._last_iter.state_dict() if self._last_iter else {"epoch": self._epoch, "yielded": 0}

    def load_state_dict(self, state_dict: dict) -> None:
        self._epoch = int(state_dict.get("epoch", 0))
        if self._last_iter is not None:
            self._last_iter.load_state_dict(state_dict)

    def check_batches(self, drop_last: bool = True) -> dict:
        """模拟一个 epoch 的切批,确认每个整批都由完整的组构成。"""
        flat = self.flat_order(0)
        nb = len(flat) // self.batch_size
        pos = {idx: gi for gi, g in enumerate(self.groups) for idx in g}
        broken = 0
        for b in range(nb):
            chunk = flat[b * self.batch_size : (b + 1) * self.batch_size]
            cnt: dict = {}
            for idx in chunk:
                cnt[pos[idx]] = cnt.get(pos[idx], 0) + 1
            if any(c != self.group_size for c in cnt.values()):
                broken += 1
        tail = len(flat) - nb * self.batch_size
        return {
            "batches": nb,
            "broken_batches": broken,
            "dropped_rows": tail if drop_last else 0,
            "dropped_groups": (tail // self.group_size) if drop_last else 0,
        }


def build_groups_from_extra_info(
    extra_infos: Sequence[dict], key: str, group_size: int, strict: bool = False
) -> List[List[int]]:
    buckets: "OrderedDict[str, List[int]]" = OrderedDict()
    missing = 0
    for i, e in enumerate(extra_infos):
        k = e.get(key) if isinstance(e, dict) else None
        if k is None:
            missing += 1
            continue
        buckets.setdefault(str(k), []).append(i)
    if missing:
        raise KeyError(f"[pair_sampler] 有 {missing} 行的 extra_info 里没有 '{key}',无法分组")
    complete = [v for v in buckets.values() if len(v) == group_size]
    incomplete = {k: len(v) for k, v in buckets.items() if len(v) != group_size}
    if incomplete:
        msg = (
            f"[pair_sampler] {len(incomplete)} 个 '{key}' 的行数不是 {group_size} "
            f"(样例 {list(incomplete.items())[:3]})"
        )
        if strict:
            raise ValueError(msg + " —— EVID_PAIR_STRICT=1,中止")
        log("⚠ " + msg + f" —— 已丢弃,保留 {len(complete)} 组")
    return complete


def build_paired_sampler(data_config, dataset) -> Optional[PairedGroupSampler]:
    """给 verl 的 create_rl_sampler 用的替身。装不上就返回 None,由调用方回退默认实现。"""
    key = os.environ.get("EVID_PAIR_KEY", "problem_id")
    gsz = envi("EVID_PAIR_GROUP_SIZE", 2)
    strict = envi("EVID_PAIR_STRICT", 0) == 1

    df = getattr(dataset, "dataframe", None)
    if df is None:
        log("⚠ dataset 没有 .dataframe,放弃接管")
        return None
    try:
        extra = list(df["extra_info"])
    except Exception as e:  # noqa: BLE001
        log(f"⚠ 取不到 extra_info({type(e).__name__}: {e}),放弃接管")
        return None

    groups = build_groups_from_extra_info(extra, key, gsz, strict=strict)
    if not groups:
        raise ValueError(f"[pair_sampler] 按 '{key}' 没有凑出任何完整的 {gsz} 行组")

    bs = int(data_config.get("gen_batch_size", None) or data_config.train_batch_size)
    shuffle = bool(data_config.get("shuffle", True))
    seed = int(data_config.get("seed") or 0)
    sampler = PairedGroupSampler(
        groups=groups,
        batch_size=bs,
        shuffle=shuffle,
        seed=seed,
        shuffle_in_group=bool(envi("EVID_PAIR_SHUFFLE_IN_GROUP", 1)),
    )
    rep = sampler.check_batches(drop_last=True)
    log(
        f"★ 配对采样已启用: {len(df)} 行 -> {len(groups)} 组 × {sampler.group_size} "
        f"= {len(sampler)} 行, batch_size={bs} -> 每批 {bs // sampler.group_size} 组"
    )
    log(
        f"  自检: {rep['batches']} 个整批, 破组批次 {rep['broken_batches']}, "
        f"尾部丢弃 {rep['dropped_rows']} 行({rep['dropped_groups']} 组), "
        f"shuffle={shuffle} seed={seed}"
    )
    if rep["broken_batches"]:
        raise RuntimeError("[pair_sampler] 自检失败:存在被拆开的组")
    return sampler


# --------------------------------------------------------------------------- #
# 离线自测
# --------------------------------------------------------------------------- #
def _selftest(parquet: str, batch_size: int = 32) -> int:
    import pandas as pd

    d = pd.read_parquet(parquet)
    extra = list(d["extra_info"])
    key = os.environ.get("EVID_PAIR_KEY", "problem_id")
    gsz = envi("EVID_PAIR_GROUP_SIZE", 2)
    groups = build_groups_from_extra_info(extra, key, gsz)
    print(f"行数 {len(d)}  完整组 {len(groups)}  组大小 {gsz}")

    s = PairedGroupSampler(groups, batch_size=batch_size, shuffle=True, seed=0)
    rep = s.check_batches(drop_last=True)
    print("切批自检:", rep)

    view = [e.get("view") for e in extra]
    flat = s.flat_order(0)
    nb = len(flat) // batch_size
    bad = sum(
        1
        for b in range(nb)
        if [view[i] for i in flat[b * batch_size : (b + 1) * batch_size]].count("dense") != batch_size // 2
    )
    print(f"每批 dense/sparse 等量: 不等量批次 {bad} / {nb}")

    it1 = iter(s)
    for _ in range(70):
        next(it1)
    sd = it1.state_dict()
    it2 = _PairedIterator(s, sd["epoch"])
    it2.load_state_dict(sd)
    resumed = [next(it2) for _ in range(10)]
    expect = s.flat_order(sd["epoch"])[70:80]
    print(f"断点续采一致: {resumed == expect}  (state={sd})")

    e0, e1 = s.flat_order(0), s.flat_order(1)
    print(f"不同 epoch 顺序不同: {e0 != e1}   两 epoch 元素集合相同: {sorted(e0) == sorted(e1)}")

    pos = {idx: gi for gi, g in enumerate(groups) for idx in g}
    adj = all(pos[flat[i]] == pos[flat[i + 1]] for i in range(0, len(flat) - 1, 2))
    print(f"同组两行相邻: {adj}")

    ok = rep["broken_batches"] == 0 and bad == 0 and resumed == expect and e0 != e1 and adj
    print("\n自测", "通过 ✓" if ok else "失败 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(_selftest(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32))
    print(__doc__)
