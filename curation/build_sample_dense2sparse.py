#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构造 dense2sparse 对比样本(与 build_sample_sparse2dense.py 对应):
把 view == "dense" 的样本, 在其视频前面或后面随机拼接一段时长相近的别的视频, 稀释证据,
变成含两个 event segment 的 sparse 视角。

输入: rollout_direct_valid.jsonl(eval_rollout_acc_rule.py 的产出, 带 view), 还需要 duration 字段
    (add_duration_from_meta.py 打), 没有 duration 的条目进不了拼接候选池。只处理 view == "dense" 的样本。

对每条 dense 样本:
  1) 视频: 优先从同一 data_source 里挑时长与目标相近(默认相差 <= 30%)的视频(规格更一致; 该源无其它视频时
     回退全池), 随机拼在目标视频前面或后面; 没有满足条件的就选时长最接近的。用 PyAV 重编码成一段。
  2) 文本: 在原 prompt 第一句之后插入一句, 说明视频包含两个 event segment、应根据前/后哪一段作答。

视频命名与 build_sample_sparse2dense.py 一致(复用其 make_contra_path):
  .../<数据集>/.../x.mp4 -> .../<数据集><suffix>/.../x_{problem_id}.mp4

输出: 追加写到 --out, 每行 {problem_id, data_source, video(拼接后的视频), prompt(改写后的指令)}。
已在输出里的 problem_id 不重复追加; 已存在的对比视频按 size 跳过(可断点续跑)。

用法:
  python build_sample_dense2sparse.py --input rollout_direct_valid.jsonl --out contra_d2s.jsonl --workers 8
"""

import argparse
import json
import os
import random
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction

# 复用 sparse2dense 的对比样本路径命名, 保证两边一致
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sample_sparse2dense import make_contra_path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evathink_paths import WORK, VIDEO_ROOT, QWEN3VL  # noqa: E402
ROOT = WORK
DEFAULT_INPUT = f"{ROOT}/outputs/qwen3vl-8b/rollout_direct_valid.jsonl"
DEFAULT_OUT = f"{ROOT}/outputs/qwen3vl-8b/prompt4rollout_direct_contra.jsonl"
# 临时目录要在本地盘: mp4 mux 需要随机写, 网络挂载盘上可能失败或极慢; 拼好后再顺序拷到 dst。
SCRATCH = "/tmp/d2s_tmp"
MIN_VALID_BYTES = 2048

# 直答 prompt 的首句(build_prompt4rollout_direct.py 的模板开头), 提示句插在它后面。
FIRST_SENTENCE = "You are given a video."
ADD_LATTER = "The video contains two separate event segments, please answer the question based only on the second event segment."   # 目标在后
ADD_FORMER = "The video contains two separate event segments, please answer the question based only on the first event segment."   # 目标在前


def modify_prompt(prompt, target_is_latter):
    """在第一句后插入"两段视频, 看前/后段"的提示。"""
    add = ADD_LATTER if target_is_latter else ADD_FORMER
    if FIRST_SENTENCE in prompt:
        return prompt.replace(FIRST_SENTENCE, FIRST_SENTENCE + "\n" + add, 1)
    return add + "\n" + prompt   # 兜底: 找不到首句就放最前


def choose_other(target_video, target_dur, pool, rng, tol):
    """从 pool[(video,dur)] 里挑一个拼接视频: 时长相差 <=tol 的随机选; 否则选最接近的。
    排除目标自身。返回 (video, dur) 或 None。"""
    cands = [(v, d) for v, d in pool if v != target_video and d]
    if not cands:
        return None
    ok = [(v, d) for v, d in cands if abs(d - target_dur) <= tol * target_dur]
    if ok:
        return rng.choice(ok)
    return min(cands, key=lambda vd: abs(vd[1] - target_dur))


def _probe_geom(container, vstream):
    """输出几何: 取目标视频的偶数宽高与帧率。"""
    rate = vstream.average_rate or vstream.guessed_rate or Fraction(25, 1)
    if float(rate) <= 0:
        rate = Fraction(25, 1)
    W = (vstream.codec_context.width or 0) & ~1
    H = (vstream.codec_context.height or 0) & ~1
    return W, H, rate


def concat_videos(sources, dst, geom_src, crf=23, tmp_dir=SCRATCH):
    """按 sources 顺序把多段视频重编码拼成一段 H.264 mp4 写到 dst。
    输出宽高/帧率取 geom_src(目标视频); 其余段的帧 reformat 到该尺寸。
    先写本地临时(mp4 mux 需随机写), 再顺序拷到 dst。"""
    import av

    with av.open(geom_src) as gc:
        W, H, rate = _probe_geom(gc, gc.streams.video[0])

    os.makedirs(tmp_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in os.path.basename(dst))
    tmp = os.path.join(tmp_dir, f"{os.getpid()}_{abs(hash(dst)) & 0xffffff}_{safe}")

    TB = Fraction(1, 90000)   # 细粒度 CFR 定时, 避免相邻帧 dts 冲突(见 sparse2dense)
    out = av.open(tmp, "w")
    ostream = out.add_stream("libx264", rate=rate)
    ostream.pix_fmt = "yuv420p"
    ostream.options = {"crf": str(crf)}
    ostream.codec_context.time_base = TB
    if W and H:
        ostream.width, ostream.height = W, H
    n = 0
    try:
        for src in sources:
            with av.open(src) as c:
                vs = c.streams.video[0]
                vs.thread_type = "AUTO"
                for frame in c.decode(vs):
                    if frame.pts is None:
                        continue
                    if not ostream.width or not ostream.height:
                        ostream.width, ostream.height = frame.width & ~1, frame.height & ~1
                    nf = frame.reformat(width=ostream.width, height=ostream.height,
                                        format="yuv420p")
                    nf.pts = round(n / float(rate) / TB)
                    nf.time_base = TB     # 必须设, 否则沿用源视频 time_base 导致时长错乱
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
        return {"ok": False, "err": "no_frames"}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(tmp, dst)
    os.remove(tmp)
    return {"ok": True, "n_frames": n}


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def process_one(plan, args):
    """按预先算好的 plan 拼一条。plan: {sample, other_video, target_is_latter, dst}。"""
    s = plan["sample"]
    dst = plan["dst"]
    info = {"problem_id": s["problem_id"], "dst": dst}

    target = s["video"]
    if not os.path.exists(target) or not os.path.exists(plan["other_video"]):
        info["status"] = "src_missing"
        return None, info

    if (not args.overwrite) and os.path.exists(dst) and os.path.getsize(dst) >= MIN_VALID_BYTES:
        info["status"] = "skip_exist"
    elif args.dry_run:
        info["status"] = "dry_run"
    else:
        # 目标在后 -> 别的视频拼前面; 目标在前 -> 拼后面
        sources = ([plan["other_video"], target] if plan["target_is_latter"]
                   else [target, plan["other_video"]])
        try:
            r = concat_videos(sources, dst, geom_src=target, crf=args.crf, tmp_dir=args.tmp_dir)
        except Exception as e:
            info["status"] = "exc:" + type(e).__name__
            info["trace"] = traceback.format_exc().splitlines()[-1]
            return None, info
        if not r.get("ok"):
            info["status"] = "concat_fail:" + r.get("err", "?")
            return None, info
        info["status"] = "concat"

    rec = {
        "problem_id": s["problem_id"],
        "data_source": s.get("data_source"),
        "video": dst,
        "prompt": modify_prompt(s["prompt"], plan["target_is_latter"]),
        "view": "sparse",   # 拼长稀释证据 -> 稀疏视图
    }
    return rec, info


def main():
    ap = argparse.ArgumentParser(description="构造 dense2sparse 对比样本(拼接等长视频稀释证据)")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="打过 view/duration 标的样本 jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="对比样本 prompt jsonl(追加写, 不覆盖)")
    ap.add_argument("--view", default="dense", help="只处理 view 等于此值的样本(默认 dense)")
    ap.add_argument("--dur-tol", type=float, default=0.30, help="拼接视频与目标的时长相差比例上限(默认0.3)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子(选拼接视频/前后位置), 便于复现")
    ap.add_argument("--crf", type=int, default=23, help="x264 crf")
    ap.add_argument("--tmp-dir", default=SCRATCH, help="本地临时目录(mp4 mux 用, 再拷到 bucket)")
    ap.add_argument("--workers", type=int, default=32, help="并发拼接线程数")
    ap.add_argument("--shard", default="0/1",
                    help="分片 'i/n': 只做第 i 份(0<=i<n)。★分片是在【方案全部算完之后】切的, "
                         "所以 n 份合起来和单机跑一模一样(拼接候选池、随机种子都用全量, 不受分片影响)")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 个(调试)")
    ap.add_argument("--overwrite", action="store_true", help="已存在也重拼")
    ap.add_argument("--suffix", default="_qwen_contra", help="对比样本目录后缀(与 build_sample_sparse2dense.py 的 --suffix 同义)")
    ap.add_argument("--dry-run", action="store_true", help="只算方案/路径, 不真正拼接")
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    dense = [r for r in rows if r.get("view") == args.view]
    # 拼接候选池: 本 jsonl 所有(视频, 时长), 按视频去重; 并按 data_source 分组
    #   优先从"同一 data_source"里选拼接视频(同数据集规格更一致, 减少分辨率/帧率差异);
    #   该源没有其它视频时(如某 source 只 1 个视频)回退到全池, 避免丢样本。
    uniq = {r["video"]: (r["video"], r.get("duration"), r.get("data_source")) for r in rows
            if r.get("video") and r.get("duration")}
    pool = [(v, d) for v, d, _ in uniq.values()]                      # 全池(回退用)
    pool_by_src = defaultdict(list)
    for v, d, ds in uniq.values():
        pool_by_src[ds].append((v, d))
    print(f"[load] 样本 {len(rows)}, view=={args.view} {len(dense)} 条, "
          f"拼接候选视频 {len(pool)} 个 / {len(pool_by_src)} 个 data_source", file=sys.stderr)

    # 已在输出文件里的 problem_id(避免重复追加)
    existing_pids = set()
    if os.path.exists(args.out):
        for r in load_jsonl(args.out):
            existing_pids.add(r.get("problem_id"))
    print(f"[load] 输出已存在 {len(existing_pids)} 个 problem_id, 命中则跳过追加", file=sys.stderr)

    # 串行定方案(带种子, 可复现且线程安全), 再并行拼接
    rng = random.Random(args.seed)
    plans = []
    n_no_other = n_fallback = 0
    for s in dense:
        # 优先同一 data_source; 该源无其它视频则回退全池
        other = choose_other(s["video"], s.get("duration"),
                             pool_by_src.get(s.get("data_source"), []), rng, args.dur_tol)
        if other is None:
            other = choose_other(s["video"], s.get("duration"), pool, rng, args.dur_tol)
            if other is not None:
                n_fallback += 1
        target_is_latter = rng.random() < 0.5     # True=别的视频拼前面, 目标是后段
        if other is None:
            n_no_other += 1
            continue
        plans.append({"sample": s, "other_video": other[0],
                      "target_is_latter": target_is_latter,
                      "dst": make_contra_path(s["video"], s["problem_id"], suffix=args.suffix)})
    if n_fallback:
        print(f"[plan] {n_fallback} 条同源无其它视频, 回退全池选取", file=sys.stderr)
    if n_no_other:
        print(f"[plan] {n_no_other} 条找不到可拼接视频, 跳过", file=sys.stderr)
    # 分片: 必须在 plans 算完之后切 —— 池子和 rng 序列都已按全量走完, 各分片拿到的
    #   方案与单机跑完全一致, 不会因为分片而选到不同的拼接视频。
    try:
        _si, _sn = (int(x) for x in str(args.shard).split("/"))
    except Exception:
        print(f"[fatal] --shard 格式应为 i/n, 收到 {args.shard!r}", file=sys.stderr)
        return
    if _sn > 1:
        plans = [p for k, p in enumerate(plans) if k % _sn == _si]
        print(f"[plan] 分片 {_si}/{_sn}: 本片 {len(plans)} 条", file=sys.stderr)
    if args.limit > 0:
        plans = plans[:args.limit]
    pos = Counter("latter" if p["target_is_latter"] else "former" for p in plans)
    print(f"[plan] 待处理 {len(plans)} 条  位置分布(目标段){dict(pos)}", file=sys.stderr)

    results = []
    info_stat = Counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, p, args): p for p in plans}
        for fut in as_completed(futs):
            rec, info = fut.result()
            done += 1
            info_stat[info["status"].split(":")[0]] += 1
            if rec is not None:
                results.append(rec)
            else:
                print(f"[skip] pid={info['problem_id']} {info['status']} {info.get('trace','')}",
                      file=sys.stderr)
            if done % 20 == 0 or done == len(plans):
                print(f"[concat] {done}/{len(plans)} ok={len(results)} {dict(info_stat)}",
                      file=sys.stderr)

    # 追加写(不覆盖); 跳过已在输出里的 problem_id
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_written = n_dup = 0
    with open(args.out, "a") as fo:
        for rec in results:
            if rec["problem_id"] in existing_pids:
                n_dup += 1
                continue
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing_pids.add(rec["problem_id"])
            n_written += 1

    print(f"[done] 成功 {len(results)}  追加 {n_written}  已存在跳过 {n_dup}  -> {args.out}",
          file=sys.stderr)
    print(f"[done] status={dict(info_stat)}", file=sys.stderr)


if __name__ == "__main__":
    main()
