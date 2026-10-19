"""RealTimestampDataset —— 用真实视频时间信息覆盖 video_metadata。

训练数据的每个视角是一个预抽 32 张 JPEG 的目录, qwen_vl_utils 拿不到源视频信息, 只能现造
fps=2.0, frames_indices=[0..31]。Qwen3-VL 的 processor 会把 timestamps = frames_indices / fps 写成文本放进
prompt, 于是每个样本的时间戳都是 <0.2 seconds> ... <15.2 seconds>, 模型读到的视频时长恒为 15.2 秒。
而 dense / sparse 两个视角的真实时长差别很大(这正是证据疏密的一部分), 也是 curation 测信号时模型看到的。
在 messages 的 video 字典里传 fps 对预抽帧无效, 所以在数据集类里覆盖。

覆盖 _process_multi_modal_info: AgentLoop 调用 process_multi_modal_info / process_vision_info,
两者都汇到这个类方法, 它也是 verl 留给用户 override 的扩展点。它是类方法, 只拿得到 messages、拿不到
extra_info, 所以用帧目录名(每个视频唯一)做 key 查 ts_map.json。RLHFDataset.__getitem__ 不做视频处理,
不能挂在那里。

挂载(不改 verl 源码)
    data.custom_cls.path=<EVAThink>/verl_ext/ts_dataset.py
    data.custom_cls.name=RealTimestampDataset
    EVID_TS_MAP=<数据集目录>/ts_map.json     # 不设则按 data.train_files 同目录找

环境变量
    EVID_TS_MAP        查找表路径
    EVID_TS_DISABLE=1  不覆盖
    EVID_TS_REQUIRED=1 查不到就报错(默认 0: 保留原值并计数)
"""

from __future__ import annotations

import json
import os
import sys
import threading

from verl.utils.dataset.rl_dataset import RLHFDataset

_MAP = None
_MAP_LOCK = threading.Lock()
_STATS = {"ok": 0, "miss": 0, "nokey": 0}
_REPORTED = set()


def _log(msg: str) -> None:
    print(f"[ts_dataset] {msg}", file=sys.stderr, flush=True)


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:  # noqa: BLE001
        return default


def _load_map():
    """进程内只加载一次(AgentLoopWorker 是独立进程,各自加载)。"""
    global _MAP
    if _MAP is not None:
        return _MAP
    with _MAP_LOCK:
        if _MAP is not None:
            return _MAP
        p = os.environ.get("EVID_TS_MAP", "")
        if not p or not os.path.exists(p):
            _log(f"✗ 查找表不存在: {p!r} —— 不覆盖时间戳(设 EVID_TS_MAP 指向 ts_map.json)")
            _MAP = {}
            return _MAP
        with open(p, encoding="utf-8") as f:
            _MAP = json.load(f)
        _log(f"★ 已加载真实时间戳查找表: {len(_MAP)} 条  <- {p}  pid={os.getpid()}")
        return _MAP


def _video_keys(messages):
    """按出现顺序取出每个 video 条目的帧目录名,与 process_vision_info 返回的 videos 顺序一致。"""
    keys = []
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, (list, tuple)):
            continue
        for it in c:
            if isinstance(it, dict) and it.get("type") == "video":
                v = it.get("video")
                try:
                    first = str(list(v)[0]) if not isinstance(v, str) else v
                    keys.append(os.path.basename(os.path.dirname(first)))
                except Exception:  # noqa: BLE001
                    keys.append(None)
    return keys


def _apply(meta, rec, nframes):
    """把真实 fps/frames_indices 写进 metadata(dict 与对象两种形态都支持)。"""
    idx = list(rec["frames_indices"])
    if nframes:
        idx = idx[:nframes]
        while len(idx) < nframes:      # 帧数不足 32 的极短视频补过末帧
            idx.append(idx[-1])
    vals = {"fps": float(rec["fps"]), "frames_indices": idx,
            "total_num_frames": float(rec.get("total_num_frames", len(idx)))}
    if isinstance(meta, dict):
        meta.update(vals)
        return True
    for k, v in vals.items():
        try:
            setattr(meta, k, v)
        except Exception:  # noqa: BLE001
            return False
    return True


def override_video_metadata(messages, videos):
    if not videos:
        return videos
    m = _load_map()
    if not m:
        return videos
    keys = _video_keys(messages)
    for i, item in enumerate(videos):
        meta = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else None
        if meta is None:
            continue
        key = keys[i] if i < len(keys) else None
        if not key:
            _STATS["nokey"] += 1
            continue
        rec = m.get(key)
        if rec is None:
            _STATS["miss"] += 1
            if _envi("EVID_TS_REQUIRED", 0) == 1:
                raise KeyError(f"[ts_dataset] 查找表里没有 {key}")
            if key not in _REPORTED and len(_REPORTED) < 5:
                _REPORTED.add(key)
                _log(f"⚠ 查不到 {key},该条退回伪造时间戳")
            continue
        n = None
        try:
            n = int(item[0].shape[0])
        except Exception:  # noqa: BLE001
            pass
        if _apply(meta, rec, n):
            _STATS["ok"] += 1
            t = _STATS["ok"]
            if t in (1, 50, 500, 5000):
                dur = rec.get("duration", -1)
                _log(f"覆盖 #{t}: {key} fps={rec['fps']:.3f} 真实时长={dur:.1f}s "
                     f"(累计 ok={_STATS['ok']} miss={_STATS['miss']} nokey={_STATS['nokey']})")
    return videos


class RealTimestampDataset(RLHFDataset):
    """行为与 RLHFDataset 完全一致,只是把视频 metadata 换成真实值。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _envi("EVID_TS_DISABLE", 0) == 1:
            _log("EVID_TS_DISABLE=1 —— 不覆盖时间戳(对照组)")
            return
        # 未显式指定查找表时,按数据集目录找
        if not os.environ.get("EVID_TS_MAP"):
            try:
                df = self.data_files[0] if isinstance(self.data_files, (list, tuple)) else self.data_files
                cand = os.path.join(os.path.dirname(str(df)), "ts_map.json")
                if os.path.exists(cand):
                    os.environ["EVID_TS_MAP"] = cand
                    _log(f"自动定位查找表 -> {cand}")
            except Exception:  # noqa: BLE001
                pass
        _load_map()

    @classmethod
    def _process_multi_modal_info(cls, messages, image_patch_size, config):
        images, videos, audios = super()._process_multi_modal_info(
            messages, image_patch_size=image_patch_size, config=config)
        if _envi("EVID_TS_DISABLE", 0) != 1:
            videos = override_video_metadata(messages, videos)
        return images, videos, audios
