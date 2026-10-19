#!/usr/bin/env python3
"""给每个视频精确抽 32 帧: 全程不 seek, 从头逐帧顺序解码, 数到目标帧号才取。

按帧号 seek(如 cv2 CAP_PROP_POS_FRAMES)依赖 pts -> 帧号换算, 可变帧率或起始时间戳有偏移的视频可能差几帧;
seek 到关键帧直接取帧则更不准。这里用 cv2 顺序 grab(), 命中目标帧才 retrieve(); cv2 读不出帧的(如 AV1)
走 assemble_trainset4verl._frames_pyav(PyAV 顺序解码计数), 同样精确。
帧号 np.linspace(0, total-1, min(32, total)).round(), 不足补末帧; total / fps 取 cv2 的 CAP_PROP_FRAME_COUNT /
CAP_PROP_FPS; jpg 质量 90; 帧目录名 vid_key(视频路径) + "__32f"; 时间元数据格式同训练端(A._mk_meta)。
若视频实际能解出的帧少于 total, 按实际帧数重算帧号再抽一遍, meta 里记 recount。
所有模型调用(直答、标注、慢思考)和训练都读这同一份帧。

输出: <out>/frames/<vid_key>__32f/000.jpg..031.jpg + meta.json(完成标记; 断点续跑时已完成的跳过)

用法:
  python recurate_extract_exact.py --out <work dir> --valid <prompt jsonl> --shard 0 --nshard 4 --procs 96
  python recurate_extract_exact.py --verify 20 --out <临时目录>
"""
import os
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")   # ffmpeg 静默, 免得坏包/AV1 刷屏
import argparse, json, random, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble_trainset4verl as A                 # noqa: E402  vid_key / _frames_pyav / _mk_meta 与训练端同源

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evathink_paths import WORK, VIDEO_ROOT, QWEN3VL  # noqa: E402
EFS = ROOT = ""
VALID = os.path.join(WORK, "full", "prompt4rollout_direct.jsonl")
NF, JPG_Q = 32, 90

local = lambda v: v


def target_indices(total):
    idx = np.linspace(0, total - 1, min(NF, total)).round().astype(int).tolist()
    while len(idx) < NF:
        idx.append(idx[-1])
    return idx


def _grab_seq(v, idx):
    """cv2 从头顺序 grab 到最后一个目标帧, 命中才 retrieve; 返回 ({帧号: BGR}, 成功 grab 的帧数)。"""
    want, last, got = set(idx), max(idx), {}
    cap = cv2.VideoCapture(v)
    i = 0
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, fr = cap.retrieve()
            if ok and fr is not None:
                got[i] = fr
        i += 1
    cap.release()
    return got, i


def _pyav(v):
    """AV1 等 cv2 读不出的: 训练端同一个 A._frames_pyav(PyAV 顺序解码计数) + 同样的 fps/total 取法。"""
    cap = cv2.VideoCapture(v)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    picked, used, pfps, ptotal = A._frames_pyav(v, NF)
    if not picked:
        raise RuntimeError("0 frames (cv2 与 PyAV 都解不出)")
    fps, total = (fps or pfps), (ptotal or total or len(picked))
    while len(picked) < NF:
        picked.append(picked[-1]); used.append(used[-1])
    return picked[:NF], used[:NF], fps, total, None, "pyav_seq"


def extract_exact(v):
    """返回 (BGR 帧列表, 帧号列表, fps, total, recount, backend)。"""
    cap = cv2.VideoCapture(v)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or total <= 0:
        return _pyav(v)
    idx = target_indices(total)
    got, ndec = _grab_seq(v, idx)
    if not got:                                     # cv2 一帧都读不出(AV1 等)
        return _pyav(v)
    recount = None
    if any(i not in got for i in idx):              # 元数据帧数偏大: 按实际能解出的帧数重算帧号再抽
        recount = {"meta_total": total, "decodable": ndec}
        total = ndec
        idx = target_indices(total)
        got, _ = _grab_seq(v, idx)
        if any(i not in got for i in idx):
            raise RuntimeError(f"重算后仍缺帧 decodable={ndec}")
    return [got[i] for i in idx], idx, fps, total, recount, "cv2_seq"


def write_frames(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for k, fr in enumerate(frames):
        p = os.path.join(out_dir, f"{k:03d}.jpg")
        cv2.imwrite(p, fr, [cv2.IMWRITE_JPEG_QUALITY, JPG_Q]); paths.append(p)
    return paths


def meta_ok(mf):
    """meta.json 能解析且帧号齐全 = 该视频已完成; 写到一半断掉的解析不了, 续跑时会重抽。"""
    try:
        return len(json.load(open(mf))["frames_indices"]) == NF
    except Exception:                               # noqa: BLE001
        return False


def work(job):
    v, out_dir = job                                # v = 数据文件里的视频绝对路径
    mf = os.path.join(out_dir, "meta.json")
    if meta_ok(mf):
        return v, "cached", None
    t = time.time()
    try:
        frames, idx, fps, total, recount, backend = extract_exact(local(v))
        write_frames(frames, out_dir)
        meta = {"video": v, **A._mk_meta(fps, total, idx, NF), "backend": backend, "recount": recount,
                "sec": round(time.time() - t, 1)}
        with open(mf, "w") as f:                    # 最后写 meta.json = 完成标记
            json.dump(meta, f, ensure_ascii=False)
        return v, backend + ("" if recount is None else "+recount"), None
    except Exception as e:                          # noqa: BLE001
        return v, "err", f"{type(e).__name__}:{str(e)[:120]}"



def load_videos(valid):
    dur = {}
    for l in open(valid):
        d = json.loads(l)
        dur[d["video"]] = max(dur.get(d["video"], 0), float(d.get("duration") or 0))
    return dur


def verify(n, tmp_root, seed=0):
    """自检: ① 本脚本抽的 jpg 与 vLLM opencv 后端顺序读取的同帧号原始帧应为同一帧(jpg 有损, 平均差 ~1-3);
       ② 对比按帧号 seek 抽到的帧, 看是否取偏。
    """
    from vllm.multimodal.video import VideoBackend as VB
    dur = load_videos(VALID)
    vs = sorted(dur, key=lambda v: -dur[v])
    pick = vs[:3] + random.Random(seed).sample(vs, n - 3)       # 最长的 3 个 + 随机
    mean_diff = lambda a, b: float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
    bad1 = bad2 = 0
    for v in pick:
        t = time.time(); lv = local(v)
        frames, idx, fps, total, recount, backend = extract_exact(lv)
        pa = write_frames(frames, os.path.join(tmp_root, "seq", A.vid_key(v)))
        mine = [cv2.imread(p) for p in pa]
        if backend != "cv2_seq":                     # AV1: cv2 读不了, 既没有 opencv 参考也不能跑训练端 seek, 只报抽帧结果
            print(f"  {os.path.basename(v)[:34]:34s} {dur[v]:6.0f}s backend={backend} total={total} 抽帧 {time.time()-t:.0f}s (无 opencv 参考, 跳过比对)", flush=True)
            continue
        ref_msg = "opencv 参考读不全"
        cap = VB.open_video_capture(open(lv, "rb").read())
        uniq = sorted(set(idx))
        ref, ok_idx = VB.read_frames(cap, uniq, total_frames_num=total)
        if len(ok_idx) == len(uniq):
            rmap = dict(zip(ok_idx, ref))
            d1 = max(mean_diff(cv2.cvtColor(m, cv2.COLOR_BGR2RGB), rmap[i]) for m, i in zip(mine, idx))
            bad1 += d1 > 5; ref_msg = f"vs vLLM顺序参考 最大 {d1:.2f}{' ★' if d1 > 5 else ''}"
        tp, tmeta = A.extract_frames(lv, NF, os.path.join(tmp_root, "train", A.vid_key(v)), JPG_Q)
        same_idx = tmeta["frames_indices"] == idx
        d2 = max(mean_diff(cv2.imread(p), m) for p, m in zip(tp, mine))
        bad2 += d2 > 5
        print(f"  {os.path.basename(v)[:34]:34s} {dur[v]:6.0f}s total={total:6d} recount={recount} | {ref_msg} | "
              f"训练端seek 帧号相同={same_idx} 最大差 {d2:.2f}{' ★取偏' if d2 > 5 else ''} ({time.time()-t:.0f}s)", flush=True)
    print(f"[verify] {len(pick)} 个视频: 顺序解码与 vLLM 参考不一致 {bad1} 个; 训练端 seek 取偏 {bad2} 个", flush=True)


def main():
    try:                                            # 兜底: 被绑在少数核上时放开到全部核(子进程继承)
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
    except Exception:                               # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--valid", default=VALID)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--procs", type=int, default=max(1, len(os.sched_getaffinity(0)) // 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", type=int, default=0)
    a = ap.parse_args()
    if a.verify:
        return verify(a.verify, a.out)

    dur = load_videos(a.valid)
    vs = sorted(dur)                                    # 固定顺序后按下标分片
    mine = [v for i, v in enumerate(vs) if i % a.nshard == a.shard]
    mine.sort(key=lambda v: -dur[v])                    # 长视频先提交, 负载更均衡
    if a.limit:
        mine = mine[:a.limit]
    froot = os.path.join(a.out, "frames")
    os.makedirs(froot, exist_ok=True)
    jobs = [(v, os.path.join(froot, f"{A.vid_key(v)}__{NF}f")) for v in mine]
    print(f"[extract] shard {a.shard}/{a.nshard}: 视频 {len(jobs)} 个, 进程 {a.procs}", flush=True)
    t0 = time.time(); stat = {}; errs = []
    with ProcessPoolExecutor(a.procs) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            v, st, err = f.result()
            stat[st] = stat.get(st, 0) + 1
            if err:
                errs.append({"video": v, "err": err})
            if k % 300 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"[extract] {k}/{len(jobs)}  {el/60:.1f}分  预计总 {el/k*len(jobs)/60:.1f}分  {stat}", flush=True)
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    with open(os.path.join(a.out, "logs", f"extract_errors.shard{a.shard}.jsonl"), "w") as f:
        for e in errs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[extract] ★ shard{a.shard} 完成 {stat}  失败 {len(errs)}  用时 {(time.time()-t0)/60:.1f} 分", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
