#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构造 sparse2dense 对比样本: 对 view == "sparse" 且证据集中在半段以内的题, 裁出只覆盖证据片段 + 少量冗余的
短视频, 在更短的片段上重新均匀抽 32 帧, 证据随之变稠密; 文本指令沿用直答 prompt(只把视频换成裁剪后的版本)。

数据流:
  valid jsonl(rollout_direct_valid.jsonl, 每行 {problem_id, data_source, video, prompt, acc, view, duration})
    └─ 取 view == "sparse" 的题
  证据定位标注(anno_gt.jsonl, response = 证据帧区间 JSON, Qwen3.5-27B 标注)
    └─ 解析 {status, start_frame, end_frame}, 帧号在 32 帧空间(0-31, 闭区间)
    └─ 只保留 status ∈ {localized, partial}, 且证据段长 (end-start+1) <= 半段(默认 16 帧)
  帧号 -> 时间: 32 帧是均匀抽取含首尾(np.linspace(0, N-1, 32)), frame i 对应归一化位置 i/(TOTAL-1)。
  裁剪: 证据段两侧各扩 pad 帧作冗余, pad 每条在 --pad-frames 的候选里随机取, 避免证据总在片段正中间;
        映射成时间区间后用 PyAV(libx264) 重编码。
        时间敏感题(题干/选项含绝对时间戳): 左端固定到 t=0 只裁右侧, 保住从视频起点度量的绝对时间。
  产物:
    1) 对比视频: 数据集目录名加 suffix, 文件名加 _{problem_id}
       (.../<数据集>/.../x.mp4 -> .../<数据集><suffix>/.../x_{pid}.mp4); 同一视频可能被多个题共享且证据段不同,
       所以文件名带 pid。
    2) 对比样本 prompt jsonl, 追加写: {problem_id, data_source, video, prompt, view="dense"},
       字段与 build_sample_dense2sparse.py 一致(那边出 view="sparse")。

用法:
  python build_sample_sparse2dense.py --valid rollout_direct_valid.jsonl --gt-anno anno_gt.jsonl --out contra_s2d.jsonl
  python build_sample_sparse2dense.py --limit 5 --dry-run       # 只看筛选/路径, 不裁剪

断点续跑: 目标视频已存在且大小达标即跳过。
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evathink_paths import WORK, VIDEO_ROOT, QWEN3VL  # noqa: E402
ROOT = WORK
DEFAULT_VALID = f"{ROOT}/outputs/qwen3vl-8b/rollout_direct_valid.jsonl"
DEFAULT_GT_ANNO = f"{ROOT}/outputs/qwen3vl-8b/anno_gt.jsonl"
DEFAULT_ROLLOUT_PROMPT = f"{ROOT}/outputs/pangu-30ba2/prompt4rollout_direct_mc.jsonl"
DEFAULT_OUT = f"{ROOT}/outputs/qwen3vl-8b/prompt4rollout_direct_contra.jsonl"

# 临时目录要在本地盘: mp4 mux 需要随机写, 网络挂载盘上可能失败或极慢; 写好再顺序拷到目标
SCRATCH = "/tmp/s2d_tmp"

TOTAL_FRAMES = 32          # 抽帧数(frame id 0..TOTAL_FRAMES-1)
MIN_VALID_BYTES = 2048     # 断点续跑: 目标视频小于此字节数视为不完整, 重裁


# ----------------------------- 解析 GT 定位 -----------------------------
def parse_localization(resp):
    """从 GT 标注 response 中抽出 dict, 容忍 ```json``` 包裹。"""
    if not resp:
        return None
    s = str(resp).strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s.strip())
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)
    try:
        d = json.loads(s)
    except Exception:
        return None
    return d if isinstance(d, dict) else None


# 绝对时间戳 H:MM:SS / M:SS(秒/分为两位), 用于识别"时间敏感"题
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def is_time_sensitive(prompt):
    """题干/选项里出现任何绝对时间戳(如 0:01:20 / 1:30) 即视为时间敏感。
    这类题(答案是时间段, 或引用绝对时刻) 若两侧裁剪会平移时间轴导致失真;
    对其只裁右侧、左侧从 t=0 全保留即可保住绝对时间映射。
    保守取向: 宁可误判(仅少裁一点, 无害) 也不漏判(会污染 GT)。"""
    return bool(TIMESTAMP_RE.search(prompt or ""))


def select_candidates(valid, gt, view_value, statuses, max_seg_frames, total_frames):
    """联表筛选出待裁剪的候选, 返回 (candidates, stats)。只处理 view==view_value(默认 sparse)。"""
    stats = Counter()
    cands = []
    for v in valid:
        if v.get("view") != view_value:
            continue
        stats["view_matched"] += 1
        pid = v["problem_id"]
        g = gt.get(pid)
        if g is None:
            stats["gt_missing"] += 1
            continue
        loc = parse_localization(g.get("response"))
        if not loc:
            stats["gt_unparseable"] += 1
            continue
        st = str(loc.get("status", "")).strip().lower()
        try:
            s = int(loc["start_frame"])
            e = int(loc["end_frame"])
        except (KeyError, TypeError, ValueError):
            stats["frame_unparseable"] += 1
            continue
        if statuses and st not in statuses:
            stats[f"status_skip:{st or 'empty'}"] += 1
            continue
        if s > e:
            s, e = e, s
        if not (0 <= s <= e <= total_frames - 1):
            stats["range_invalid"] += 1
            continue
        seg_len = e - s + 1
        if seg_len > max_seg_frames:
            stats["seg_too_long"] += 1
            continue
        stats["selected"] += 1
        keep_left = is_time_sensitive(v.get("prompt"))
        stats["temporal_keep_left" if keep_left else "normal_both_sides"] += 1
        cands.append({
            "problem_id": pid,
            "data_source": v["data_source"],
            "video": v["video"],
            "acc": v["acc"],
            "start_frame": s,
            "end_frame": e,
            "seg_len": seg_len,
            "status": st,
            "keep_left": keep_left,
        })
    return cands, stats


# ----------------------------- 路径改写 -----------------------------
def _rewrite_dataset_dir(video_path, anchor, suffix):
    """把 anchor(默认 QA) 之后的数据集目录名加 suffix。
    .../decompressed/QA/NeXT-QA/NextQA/x.mp4 -> .../decompressed/QA/NeXT-QA_contra/NextQA/x.mp4
    找不到 anchor 时退化为给视频的父目录加 suffix。"""
    parts = video_path.split("/")
    if anchor in parts:
        i = parts.index(anchor)
        # anchor 后须还有"数据集目录 + 文件名"(至少 2 段), 才对数据集目录加后缀
        if i + 1 < len(parts) - 1:
            parts = parts[:]
            parts[i + 1] = parts[i + 1] + suffix
            return "/".join(parts)
    d, f = os.path.split(video_path)
    return os.path.join(d + suffix, f)


def make_contra_path(video_path, problem_id, anchor="QA", suffix="_qwen_contra"):
    """对比样本视频路径 = 数据集目录加 suffix + 文件名后缀 _{problem_id}。
    加 problem_id 是因为同一视频可能被多个 problem_id 共享, 各自证据段不同 -> 裁出不同 clip,
    用 pid 区分文件名, 避免互相覆盖。
    .../QA/NeXT-QA/.../x.mp4 -> .../QA/NeXT-QA_contra/.../x_{pid}.mp4"""
    p = _rewrite_dataset_dir(video_path, anchor, suffix)
    d, f = os.path.split(p)
    stem, ext = os.path.splitext(f)
    return os.path.join(d, f"{stem}_{problem_id}{ext}")


# ----------------------------- 视频裁剪(PyAV) -----------------------------
def _probe_duration(container, vstream):
    if vstream.duration is not None and vstream.time_base is not None:
        d = float(vstream.duration * vstream.time_base)
        if d > 0:
            return d
    if container.duration is not None:
        d = container.duration / 1_000_000.0   # AV_TIME_BASE
        if d > 0:
            return d
    return None


def crop_by_frames(src, dst, start_f, end_f, pad_f, total_frames,
                   crf=23, min_clip_sec=1.0, tmp_dir=SCRATCH, keep_left=False):
    """把 src 中证据帧段(32 帧空间闭区间 [start_f, end_f], 两侧各扩 pad_f)对应的时间区间
    用 PyAV(libx264) 重编码成 mp4 写到 dst。
    keep_left=True(时间敏感题): 左端固定到视频起点(t=0), 只裁右侧。
    先写本地临时文件(mp4 mux 需随机写), 再顺序拷到 dst。返回 dict(ok, ...)。
    """
    import av

    container = av.open(src)
    try:
        vstream = container.streams.video[0]
        vstream.thread_type = "AUTO"
        duration = _probe_duration(container, vstream)
        if duration is None or duration <= 0:
            return {"ok": False, "err": "no_duration"}

        # 帧号 -> 时间: frame id i 对应归一化位置 i/(TOTAL-1)
        ps = 0 if keep_left else max(0, start_f - pad_f)
        pe = min(total_frames - 1, end_f + pad_f)
        denom = float(total_frames - 1)
        t0 = ps / denom * duration
        t1 = pe / denom * duration

        # 保证最小时长, 越界回夹
        if t1 - t0 < min_clip_sec:
            mid = (t0 + t1) / 2.0
            t0, t1 = mid - min_clip_sec / 2.0, mid + min_clip_sec / 2.0
        t0 = max(0.0, t0)
        t1 = min(duration, max(t1, t0 + 1e-3))

        rate = vstream.average_rate or vstream.guessed_rate or Fraction(25, 1)
        if float(rate) <= 0:
            rate = Fraction(25, 1)
        # 输出尺寸取偶数(libx264 + yuv420p 要求宽高为偶)
        W = (vstream.codec_context.width or 0) & ~1
        H = (vstream.codec_context.height or 0) & ~1

        os.makedirs(tmp_dir, exist_ok=True)
        safe = re.sub(r"[^0-9A-Za-z_.-]", "_", os.path.basename(dst))
        tmp = os.path.join(tmp_dir, f"{os.getpid()}_{threadsafe_id()}_{safe}")

        try:
            container.seek(int(t0 / vstream.time_base), stream=vstream, backward=True)
        except Exception:
            pass  # 不可 seek 的容器: 从头解码

        # 用细粒度 time_base(1/90000) + 强制 CFR 定时: 避免相邻帧 rescale 后 dts 冲突
        # (直接给 pts=n 会因 mp4 timescale 太粗导致 "non-monotonic dts" mux 报 EINVAL)。
        TB = Fraction(1, 90000)
        out = av.open(tmp, "w")
        ostream = out.add_stream("libx264", rate=rate)
        ostream.pix_fmt = "yuv420p"
        ostream.options = {"crf": str(crf)}
        ostream.codec_context.time_base = TB
        if W and H:
            ostream.width, ostream.height = W, H
        n = 0
        eps = 1e-4
        try:
            for frame in container.decode(vstream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * vstream.time_base)
                if t < t0 - eps:
                    continue
                if t > t1 + eps:
                    break
                if not ostream.width or not ostream.height:
                    ostream.width, ostream.height = frame.width & ~1, frame.height & ~1
                nf = frame.reformat(width=ostream.width, height=ostream.height,
                                    format="yuv420p")
                nf.pts = round(n / float(rate) / TB)   # CFR, 细粒度, 严格递增
                nf.time_base = TB
                n += 1
                for pkt in ostream.encode(nf):
                    out.mux(pkt)
            for pkt in ostream.encode():   # flush
                out.mux(pkt)
        finally:
            out.close()

        if n == 0:
            if os.path.exists(tmp):
                os.remove(tmp)
            return {"ok": False, "err": "no_frames_in_range"}

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(tmp, dst)   # 顺序写
        os.remove(tmp)
        return {"ok": True, "n_frames": n, "t0": round(t0, 3), "t1": round(t1, 3),
                "clip_sec": round(t1 - t0, 3), "duration": round(duration, 3),
                "pad_start": ps, "pad_end": pe,
                "mode": "keep_left" if keep_left else "both"}
    finally:
        container.close()


_id_counter = [0]
_id_lock = None


def threadsafe_id():
    global _id_lock
    if _id_lock is None:
        import threading
        _id_lock = threading.Lock()
    with _id_lock:
        _id_counter[0] += 1
        return _id_counter[0]


# ----------------------------- 主流程 -----------------------------
def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def process_one(c, args):
    """裁一条候选。返回 (candidate 或 None, info)。"""
    src = c["video"]
    dst = make_contra_path(src, c["problem_id"], anchor=args.contra_anchor, suffix=args.suffix)
    c["contra_video"] = dst
    info = {"problem_id": c["problem_id"], "dst": dst}

    if not os.path.exists(src):
        info["status"] = "src_missing"
        return None, info

    if (not args.overwrite) and os.path.exists(dst) and os.path.getsize(dst) >= MIN_VALID_BYTES:
        info["status"] = "skip_exist"
        return c, info

    if args.dry_run:
        info["status"] = "dry_run"
        return c, info

    try:
        keep_left = c.get("keep_left", False) and not args.no_temporal_keep_left
        # 冗余帧数每条随机取一个(默认在 2/3/4 里挑), 让裁出来的片段松紧不一, 别让模型
        # 学到"证据总是正好卡在片段正中间"这种伪特征。
        # 用 (seed, problem_id) 定种子而不是全局 random: 多线程下完成顺序不定, 这样才可复现,
        # 且同一条重跑拿到的 pad 一样(断点续跑/重裁不会前后不一致)。
        pad = random.Random(f"{args.pad_seed}-{c['problem_id']}").choice(args.pad_frames)
        c["pad_frames"] = pad
        info["pad"] = pad
        r = crop_by_frames(src, dst, c["start_frame"], c["end_frame"], pad,
                           args.total_frames, crf=args.crf, min_clip_sec=args.min_clip_sec,
                           tmp_dir=args.tmp_dir, keep_left=keep_left)
        if not r.get("ok"):
            info["status"] = "crop_fail:" + r.get("err", "?")
            return None, info
        info["status"] = "cropped"
        info.update(r)
        return c, info
    except Exception as e:
        info["status"] = "exc:" + type(e).__name__
        info["trace"] = traceback.format_exc().splitlines()[-1]
        return None, info


def main():
    ap = argparse.ArgumentParser(
        description="构造 sparse2dense 对比样本(裁证据段短视频 + rollout 格式 prompt)")
    ap.add_argument("--valid", default=DEFAULT_VALID, help="有效样本 jsonl(含 acc)")
    ap.add_argument("--gt-anno", default=DEFAULT_GT_ANNO,
                    help="GT 证据定位结果 jsonl(含 response=帧区间 JSON), 默认 anno_gt_5k.jsonl")
    ap.add_argument("--rollout-prompt", default=DEFAULT_ROLLOUT_PROMPT,
                    help="rollout 直接答题 prompt jsonl(按 problem_id 取 prompt); 缺失则用 valid 自带 prompt")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出对比样本 prompt jsonl")
    ap.add_argument("--view", default="sparse",
                    help="只处理 view 字段等于此值的样本(默认 sparse; view 由 "
                         "eval_rollout_acc_rule.py 按 --view-threshold 打好, 不用再跑 add_view4jsonl.py)")
    ap.add_argument("--total-frames", type=int, default=TOTAL_FRAMES, help="抽帧数(frame id 空间大小)")
    ap.add_argument("--max-seg-frames", type=int, default=TOTAL_FRAMES // 2,
                    help="证据段长(end-start+1)<= 此值才裁, 默认半段=16")
    ap.add_argument("--pad-frames", default="2,3",
                    help="证据段两侧各扩的冗余帧数(32 帧空间), 逗号分隔多个则每条样本随机取一个, "
                         "默认 2,3,4; 只给一个值(如 2)即固定")
    ap.add_argument("--pad-seed", type=int, default=42,
                    help="随机取 pad 的种子(按 problem_id 定种, 可复现)")
    ap.add_argument("--no-temporal-keep-left", action="store_true",
                    help="关闭时间敏感题的'只裁右侧'保护(默认开启: 含绝对时间戳的题左侧从 t=0 全保留)")
    ap.add_argument("--status", default="localized,partial",
                    help="接受的定位 status(逗号分隔), 空串=不限制")
    ap.add_argument("--min-clip-sec", type=float, default=1.0, help="裁剪片段最短时长(秒)")
    ap.add_argument("--crf", type=int, default=23, help="x264 crf(越大越糊/越小)")
    ap.add_argument("--contra-anchor", default="QA", help="在此路径段之后的目录名加后缀")
    ap.add_argument("--suffix", default="_qwen_contra", help="对比样本目录后缀")
    ap.add_argument("--shard", default="0/1",
                    help="i/n: 候选全量筛完(顺序固定)后只处理 cands[i::n], 多机并行各跑一片(默认 0/1 = 全部)")
    ap.add_argument("--tmp-dir", default=SCRATCH, help="本地临时目录(mp4 mux 用, 再拷到 bucket)")
    ap.add_argument("--workers", type=int, default=32, help="并发裁剪线程数")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 个候选(调试)")
    ap.add_argument("--overwrite", action="store_true", help="已存在也重裁")
    ap.add_argument("--dry-run", action="store_true", help="只筛选/算路径, 不真正裁剪视频")
    args = ap.parse_args()

    statuses = {s.strip().lower() for s in args.status.split(",") if s.strip()}
    args.pad_frames = [int(x) for x in str(args.pad_frames).split(",") if x.strip() != ""]
    if not args.pad_frames:
        print("[fatal] --pad-frames 解析为空", file=sys.stderr)
        return
    print(f"[cfg] pad 候选={args.pad_frames} (每条按 problem_id 随机取一个, seed={args.pad_seed})",
          file=sys.stderr)

    print(f"[load] valid={args.valid}", file=sys.stderr)
    valid = load_jsonl(args.valid)
    print(f"[load] gt-anno={args.gt_anno}", file=sys.stderr)
    gt = {d["problem_id"]: d for d in load_jsonl(args.gt_anno)}
    prompt_map = {}
    if args.rollout_prompt and os.path.exists(args.rollout_prompt):
        for d in load_jsonl(args.rollout_prompt):
            prompt_map[d["problem_id"]] = d.get("prompt")
    # valid 自带 prompt 兜底
    valid_prompt = {v["problem_id"]: v.get("prompt") for v in valid}
    print(f"[load] valid={len(valid)} gt={len(gt)} rollout_prompt={len(prompt_map)}", file=sys.stderr)

    cands, stats = select_candidates(
        valid, gt, args.view, statuses, args.max_seg_frames, args.total_frames)
    print(f"[select] {dict(stats)}", file=sys.stderr)
    src_cnt = Counter(c["data_source"].split("/")[0] for c in cands)
    print(f"[select] candidates={len(cands)} by_src={dict(src_cnt)}", file=sys.stderr)
    seg_dist = Counter(c["seg_len"] for c in cands)
    print(f"[select] seg_len_dist={dict(sorted(seg_dist.items()))}", file=sys.stderr)

    _si, _sn = (int(x) for x in str(args.shard).split("/"))
    if _sn > 1:
        cands = sorted(cands, key=lambda c: c["problem_id"])[_si::_sn]
        print(f"[select] 分片 {_si}/{_sn}: 本片 {len(cands)} 条", file=sys.stderr)
    if args.limit > 0:
        cands = cands[:args.limit]
        print(f"[select] limited to {len(cands)}", file=sys.stderr)

    results = []
    info_stat = Counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, c, args): c for c in cands}
        for fut in as_completed(futs):
            rec, info = fut.result()
            done += 1
            info_stat[info["status"].split(":")[0]] += 1
            if rec is not None:
                results.append(rec)
            else:
                print(f"[skip] pid={info['problem_id']} {info['status']} "
                      f"{info.get('trace', '')}", file=sys.stderr)
            if done % 20 == 0 or done == len(cands):
                print(f"[crop] {done}/{len(cands)} ok={len(results)} {dict(info_stat)}",
                      file=sys.stderr)

    # 写出对比样本 prompt jsonl(rollout 格式, 视频换成 _contra)
    # 追加写; 已在输出里的 problem_id 不重复追加(可反复重跑)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    existing_pids = set()
    if os.path.exists(args.out):
        for r in load_jsonl(args.out):
            existing_pids.add(r.get("problem_id"))
    print(f"[write] 输出已存在 {len(existing_pids)} 个 problem_id, 命中则跳过追加", file=sys.stderr)
    n_written = n_dup = 0
    with open(args.out, "a") as fo:
        for c in results:
            if c["problem_id"] in existing_pids:
                n_dup += 1
                continue
            prompt = prompt_map.get(c["problem_id"]) or valid_prompt.get(c["problem_id"])
            if prompt is None:
                print(f"[warn] pid={c['problem_id']} 无 rollout prompt, 跳过", file=sys.stderr)
                continue
            rec = {
                "problem_id": c["problem_id"],
                "data_source": c["data_source"],
                "video": c["contra_video"],
                "prompt": prompt,
                "view": "dense",   # 裁到证据段 -> 稠密视图
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing_pids.add(c["problem_id"])
            n_written += 1

    print(f"[done] cropped_or_exist={len(results)} 追加={n_written} 已存在跳过={n_dup} -> {args.out}",
          file=sys.stderr)
    print(f"[done] status_breakdown={dict(info_stat)}", file=sys.stderr)


if __name__ == "__main__":
    main()
