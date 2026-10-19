#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把最终筛选的样本对(sample_final.jsonl, 每题两行: dense 一行 + sparse 一行)组装成 verl 训练 parquet。

每行一条训练样本, 格式:
    {
      "data_source": str,
      "prompt":  [{"role":"user","content":"<video>\n<question>\n<options>\n\n<POSTAMBLE>"}],
      "ability": "video_reasoning",
      "reward_model": {"ground_truth": "<选项字母>", "style": "rule"},
      "extra_info": {"data_type":"video","problem_type":"choice",
                     "problem_id":..., "evid_case":..., "data_source":...,
                     "view": "dense"|"sparse", "direct_acc":.., "dense_acc":.., "sparse_acc":.., "gap":..,
                     "sparse_slow_acc":..,                                           # curation 信号
                     "src_fps":.., "src_total_frames":.., "src_duration":.., "frames_indices":[..]},  # 真实时间
      "videos": [{"type":"video","video": ["/abs/000.jpg", ... 共 num_frames 帧]}],
      "images": [], "media": [],
    }

帧: 每个视频取 num_frames 帧(np.linspace(0, N-1, num_frames) 取整; N < num_frames 时补末帧), videos 列存帧路径。

时间戳: 预抽帧目录本身不带时间信息, qwen_vl_utils 会现造 fps=2.0 / frames_indices=[0..31], 模型读到的视频时长
恒为 15.2 s。所以把 fps / 总帧数 / 帧号写进 extra_info, 并另写 ts_map.json(key = 帧目录名), 由 verl 端的
verl_ext/ts_dataset.py 读取还原真实时间戳(EVID_TS_MAP=<out 同目录>/ts_map.json)。

题干/选项从每行自己的 direct prompt 里切出; GT 从 datapool_mc.jsonl 按 problem_id 回取。

用法:
  python assemble_trainset4verl.py --input sample_final.jsonl --source datapool_mc.jsonl --out train.parquet
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evathink_paths import WORK, VIDEO_ROOT, QWEN3VL  # noqa: E402
ROOT = WORK
DEFAULT_INPUT = f"{ROOT}/outputs/qwen3vl-8b/sample_final.jsonl"
DEFAULT_SOURCE = f"{ROOT}/datasets/origin_dataset/datapool_mc.jsonl"   # 只为取 GT answer
DEFAULT_OUT = f"{ROOT}/datasets/qwen3vl-8b/trainset_verl_8k/train.parquet"
DEFAULT_FRAMES_ROOT = f"{ROOT}/datasets/qwen3vl-8b/trainset_verl_8k/frames_train"

# 追加在题干+选项之后的作答模式指令, 所有样本逐字相同。
POSTAMBLE = """Answer the question based on the video. First decide whether explicit written reasoning is necessary. Choose exactly one of the following two response modes. Your response must begin with either `<answer>` or `Alright` with no other opening.

1. Direct-answer mode
Use this mode when the visual evidence clearly supports the final answer and no written reasoning is needed. Begin directly with <answer> and output only the final answer block.

Example:
<answer>A</answer>

2. Evidence-grounded reasoning mode
Use this mode only when the answer is not immediately clear, or when the relevant evidence is temporally sparse, discontinuous, or scattered across multiple frames. Begin with `Alright`, briefly connect the relevant observed clues, and then provide the final answer block. Include only as much reasoning as necessary to determine the answer.

Example:
Alright, the earlier frames show the person picking up the cup, and the later frames show them drinking from it... <answer>A</answer>

For every response:
- The response must begin with either `<answer>` for direct answering or `Alright` for evidence-grounded reasoning. Do not use any other opening.
- Put the final answer only inside <answer></answer> at the very end of the response."""


# curation 信号字段(写进 extra_info)
SIGNAL_FIELDS = ("dense_acc", "sparse_acc", "gap", "sparse_slow_acc")


# ------------------------- 题面/答案清洗(与 rollout 构建一致) -------------------------
def clean_question(problem):
    """去掉开头 <video>/<image> 标签与结尾 'Options:'。"""
    q = problem or ""
    q = re.sub(r"^\s*<(?:video|image)>\s*", "", q)
    q = re.sub(r"\s*Options?\s*:\s*$", "", q, flags=re.IGNORECASE)
    return q.strip()


def clean_answer(answer):
    """剥去 <answer>...</answer>, 返回内部文本(通常是单个选项字母)。"""
    a = answer or ""
    m = re.search(r"<answer>(.*?)</answer>", a, flags=re.DOTALL)
    return (m.group(1) if m else a).strip()


# 从 direct prompt 里切题干与选项(模板见 build_prompt4rollout_direct.py)
QA_RE = re.compile(r"Question:\n(.*?)\n\n(.*?)\n\nYour final answer must be enclosed",
                   re.DOTALL)
# 拼接视频的两段提示句(ADD_LATTER / ADD_FORMER, 见 build_sample_dense2sparse.py)
HINT_RE = re.compile(r"The video contains two separate event segments,[^\n]*?event segment\.")


def split_direct_prompt(prompt):
    """从该行的 direct prompt 里切出 (题干, 选项块, 两段提示句或 None)。切不出来返回 None。"""
    p = prompt or ""
    m = QA_RE.search(p)
    if not m:
        return None
    question, options = m.group(1).strip(), m.group(2).strip()
    if not question or not options:
        return None
    h = HINT_RE.search(p)
    return question, options, (h.group(0) if h else None)


def build_content(question, options_block):
    """统一装配:

        <video>
        题干
        选项

        POSTAMBLE

    不插入 direct prompt 里 "The video contains two separate event segments, ... only on the {first|second}
    event segment." 这句: 同一题的 dense / sparse 两条只应该差在视频上。dense2sparse 的 sparse 样本因此不被
    告知看哪一段, 需要模型自己从证据里找。
    """
    body = "<video>\n" + question
    if options_block:
        body += "\n" + options_block
    return body + "\n\n" + POSTAMBLE


def map_problem_type(pt):
    """源 problem_type -> verl otype。目前全是多选。"""
    pt = (pt or "").lower()
    if "multiple" in pt or pt == "choice":
        return "choice"
    if "num" in pt:
        return "numerical"
    return "choice"


# ------------------------------- 抽帧(cv2, 与参考脚本一致) -------------------------------
def vid_key(mp4):
    """视频 -> 唯一目录名: basename(去后缀) 非字母数字转 _, 再拼 8 位路径 md5 防撞。"""
    base = os.path.splitext(os.path.basename(mp4))[0]
    base = re.sub(r"[^A-Za-z0-9]", "_", base)[:120]
    h = hashlib.md5(mp4.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{h}"


def _mk_meta(fps, total, idxs, n):
    """按抽帧时【真正用到的帧号】组装时间元数据, 供 verl 端还原真实时间戳。
    idxs 不足 n 时补末帧号(与 extract_frames 的补末帧一致)。信息不全返回 None。"""
    idxs = [int(i) for i in idxs][:n]
    while idxs and len(idxs) < n:
        idxs.append(idxs[-1])
    if not idxs or not fps or float(fps) <= 0 or int(total) <= 0:
        return None
    return {"src_fps": float(fps), "src_total_frames": int(total),
            "src_duration": (int(total) - 1) / float(fps), "frames_indices": idxs}


def probe_video_meta(mp4, n):
    """只读头信息(不解码)拿 fps/总帧数, 按与 extract_frames 相同的公式复原帧号。
    给「帧已抽好、直接复用」的情况用。"""
    cap = cv2.VideoCapture(mp4)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or total <= 0:                # AV1 之类 cv2 读不了 -> PyAV 读头
        try:
            import av
            with av.open(mp4) as c:
                st = c.streams.video[0]
                if fps <= 0 and st.average_rate:
                    fps = float(st.average_rate)
                if total <= 0:
                    total = int(st.frames or 0)
                    if total <= 0 and st.duration and st.time_base and fps > 0:
                        total = int(float(st.duration * st.time_base) * fps)
        except Exception:                     # noqa: BLE001
            return None
    if fps <= 0 or total <= 0:
        return None
    idxs = np.linspace(0, total - 1, min(n, total)).round().astype(int).tolist()
    return _mk_meta(fps, total, idxs, n)


def _frames_pyav(mp4, n):
    """cv2 解不了时的退路: 用 PyAV 顺序解码抽 n 帧。
    返回 (BGR ndarray 列表, 对应的源帧号列表, fps, 源总帧数); 解不出时列表为空。

    AV1 编码的视频 cv2 自带的 ffmpeg 打得开但读不出帧, PyAV(libdav1d)能解。
    """
    import av
    fps = 0.0
    with av.open(mp4) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        fps = float(st.average_rate) if st.average_rate else 0.0
        total = int(st.frames or 0)
        if total > 0:
            want = sorted(set(np.linspace(0, total - 1, min(n, total))
                              .round().astype(int).tolist()))
            wantset, mx = set(want), want[-1]
            out, got, i = [], [], 0
            for fr in c.decode(video=0):
                if i in wantset:
                    out.append(fr.to_ndarray(format="bgr24"))
                    got.append(i)
                if i >= mx:
                    break
                i += 1
            if out:
                return out, got, fps, total
        # 帧数读不到(或按索引没抓到) -> 整段解出来再均匀抽
        frames = [fr.to_ndarray(format="bgr24") for fr in c.decode(video=0)]
    if not frames:
        return [], [], fps, total
    total = len(frames)                       # 以实际解出来的帧数为准
    idxs = np.linspace(0, total - 1, min(n, total)).round().astype(int).tolist()
    return [frames[i] for i in idxs], idxs, fps, total


def extract_frames(mp4, n, out_dir, jpg_q):
    """均匀抽 n 帧(np.linspace(0,total-1,min(n,total)) 取整), 补末帧凑够 n, 存 jpg。
    已抽满则直接复用。返回 (帧路径列表, 时间元数据 dict|None)。

    元数据记的是真正存进 jpg 的那些帧的源帧号(读帧失败、补末帧、走 PyAV 退路时会偏离理论值)。
    """
    os.makedirs(out_dir, exist_ok=True)
    done = sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg"))
    if len(done) == n:
        return [os.path.join(out_dir, f) for f in done], probe_video_meta(mp4, n)

    cap = cv2.VideoCapture(mp4)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    used = []                                # 与 picked 一一对应的源帧号
    if total <= 0:
        # 帧数读不到 -> 顺序解码全部再抽
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        total = len(frames)
        if total == 0:
            cap.release()
            picked, used, pfps, ptotal = _frames_pyav(mp4, n)   # 同上: 交给 PyAV
            if not picked:
                raise RuntimeError("0 frames (cv2 与 PyAV 都解不出)")
            fps, total = (fps or pfps), (ptotal or len(picked))
            while len(picked) < n:
                picked.append(picked[-1])
                used.append(used[-1])
            paths = []
            for k, fr in enumerate(picked[:n]):
                q = os.path.join(out_dir, f"{k:03d}.jpg")
                cv2.imwrite(q, fr, [cv2.IMWRITE_JPEG_QUALITY, jpg_q])
                paths.append(q)
            return paths, _mk_meta(fps, total, used, n)
        idxs = np.linspace(0, total - 1, min(n, total)).round().astype(int).tolist()
        picked = [frames[i] for i in idxs]
        used = list(idxs)
    else:
        idxs = np.linspace(0, total - 1, min(n, total)).round().astype(int).tolist()
        picked = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:                       # seek 失败 -> 从头顺序解码到该帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cur, last = 0, None
                while cur <= i:
                    ok2, f2 = cap.read()
                    if not ok2:
                        break
                    last, cur = f2, cur + 1
                fr = last
            if fr is not None:
                picked.append(fr)
                used.append(int(i))
    cap.release()
    if not picked:
        # cv2 一帧都没读出来(AV1 之类) -> 换 PyAV 再试一次, 别直接判失败
        picked, used, pfps, ptotal = _frames_pyav(mp4, n)
        if not picked:
            raise RuntimeError("0 frames (cv2 与 PyAV 都解不出)")
        fps, total = (fps or pfps), (ptotal or total or len(picked))
    while len(picked) < n and picked:        # 不足补末帧
        picked.append(picked[-1])
        used.append(used[-1])
    paths = []
    for k, fr in enumerate(picked[:n]):
        p = os.path.join(out_dir, f"{k:03d}.jpg")
        cv2.imwrite(p, fr, [cv2.IMWRITE_JPEG_QUALITY, jpg_q])
        paths.append(p)
    return paths, _mk_meta(fps, total, used, n)


def _work(job):
    """并行抽帧 worker。job=(mp4, n, out_dir, jpg_q)。返回 (mp4, paths, meta, err)。"""
    mp4, n, out_dir, jpg_q = job
    try:
        paths, meta = extract_frames(mp4, n, out_dir, jpg_q)
        return mp4, paths, meta, None
    except Exception as e:                    # noqa: BLE001
        return mp4, None, None, f"{type(e).__name__}:{str(e)[:60]}"


# ----------------------------------------- main -----------------------------------------
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
        description="把 sample_final.jsonl 的对比样本对组装成 verl 视频 RL parquet(dense/sparse 各一条)。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="最终样本对 jsonl(select_sample_final.py 产物)")
    p.add_argument("--source", default=DEFAULT_SOURCE, help="源数据集大 JSON(提供题干/选项/答案)")
    p.add_argument("--out", default=DEFAULT_OUT, help="输出 parquet 路径")
    p.add_argument("--frames-root", default=DEFAULT_FRAMES_ROOT, help="抽帧 jpg 根目录")
    p.add_argument("--num-frames", type=int, default=32, help="每个视频抽帧数(np.linspace 均匀采样)")
    p.add_argument("--nproc", type=int, default=16, help="并行抽帧进程数")
    p.add_argument("--jpg-quality", type=int, default=90, help="抽帧 jpg 质量")
    p.add_argument("--preview-jsonl", default=None,
                   help="同时写一份可读 jsonl 预览(默认 out 同名 .preview.jsonl; 传 '' 关闭)")
    p.add_argument("--ts-map", default=None,
                   help="真实时间戳查找表输出路径(默认 out 同目录 ts_map.json; 传 '' 关闭)。"
                        "verl 端 ts_dataset.py 用它把 qwen_vl_utils 现造的 fps=2.0 换回真实值")
    p.add_argument("--limit", type=int, default=0, help="仅处理前 N 对(调试用, 0=全部)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    nf = args.num_frames

    rows_in = load_jsonl(args.input)
    if args.limit:
        rows_in = rows_in[:args.limit]
    print(f"[load] 样本 {len(rows_in)} 行(每题两行: dense + sparse) <- {args.input}", flush=True)
    # 源数据集这里【只用来取 GT answer】: sample_final 里没有答案, 题干选项则从各行自己的 prompt 切
    src_idx = {}
    with open(args.source) as f:
        head = f.read(2048).lstrip(); f.seek(0)
        src_rows = json.load(f) if head.startswith("[") else [json.loads(l) for l in f if l.strip()]
    for e in src_rows:
        src_idx[str(e.get("problem_id"))] = e
    print(f"[load] GT 源 {len(src_idx)} 条 <- {args.source}", flush=True)

    # 1) 逐行转成待生成条目(一行 = 一条训练样本), 并收集需抽帧的唯一视频
    entries = []
    skip_no_src = skip_no_ans = skip_no_video = skip_unparsed = 0
    n_hint = 0
    for r in rows_in:
        pid = r.get("problem_id")
        src = src_idx.get(str(pid))
        if src is None:
            skip_no_src += 1
            continue
        gt = clean_answer(src.get("answer", ""))
        if not gt:
            skip_no_ans += 1
            continue
        video = r.get("video")
        if not video:
            skip_no_video += 1
            continue
        parsed = split_direct_prompt(r.get("prompt"))
        if parsed is None:
            skip_unparsed += 1
            continue
        question, options_block, hint = parsed
        n_hint += hint is not None       # 只统计: 这句提示不写进训练 prompt
        entries.append({
            "problem_id": pid,
            "data_source": r.get("data_source") or src.get("data_source"),
            "content": build_content(question, options_block),
            "gt": gt,
            "otype": map_problem_type(src.get("problem_type")),
            "view": r.get("view"),
            "video": video,
            "signals": {k: r.get(k) for k in SIGNAL_FIELDS},
            "evid_case": r.get("evid_case"),
            "direct_acc": r.get("acc"),          # 该视图自己的 direct 正确率
        })
    print(f"[build] 待生成 {len(entries)} 条; 其中 {n_hint} 条的源 prompt 里有"
          f"【两段视频提示句】—— 按新口径统一丢弃(dense/sparse 只差视频, prompt 必须一致)",
          flush=True)

    # 2) 并行抽帧(按唯一视频)
    uniq_videos = sorted({e["video"] for e in entries})
    print(f"[frames] 待生成样本 {len(entries)} 条, 唯一视频 {len(uniq_videos)} 个, 抽 {nf} 帧/视频, "
          f"{args.nproc} 进程...", flush=True)
    jobs = [(v, nf, os.path.join(args.frames_root, f"{vid_key(v)}__{nf}f"), args.jpg_quality)
            for v in uniq_videos]
    frame_map, meta_map = {}, {}
    ok = fail = no_meta = 0
    os.makedirs(args.frames_root, exist_ok=True)
    with Pool(args.nproc) as pool:
        for k, (mp4, paths, meta, err) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            frame_map[mp4] = paths
            if meta:
                meta_map[mp4] = meta
            elif paths:
                no_meta += 1                  # 帧抽出来了但读不到 fps/总帧数
            if paths:
                ok += 1
            else:
                fail += 1
                print(f"  ✗ 抽帧失败 {os.path.basename(mp4)}: {err}", flush=True)
            if (k + 1) % 50 == 0:
                print(f"  ...{k+1}/{len(jobs)} ok={ok} fail={fail} {(time.time()-t0)/60:.1f}min", flush=True)
    print(f"[frames] 抽帧完成 ok={ok} fail={fail}  时间元数据 {len(meta_map)}/{ok}"
          f"{f' (★{no_meta} 个拿不到 fps/总帧数)' if no_meta else ''}", flush=True)

    # 2b) 抽帧失败的视图连同它的配对一起丢, 否则该题只剩单边视图
    failed_pids = {e["problem_id"] for e in entries if not frame_map.get(e["video"])}
    if failed_pids:
        print(f"[frames] ⚠ {len(failed_pids)} 题有视图抽帧失败, 连同其配对视图一并丢弃"
              f"(保证每题两条视图齐全)", flush=True)

    # 3) 组装 verl 行
    rows = []
    skip_frame_fail = skip_broken_pair = rows_no_meta = 0
    for e in entries:
        paths = frame_map.get(e["video"])
        if not paths:
            skip_frame_fail += 1
            continue
        if e["problem_id"] in failed_pids:
            skip_broken_pair += 1
            continue
        extra = {
            "data_type": "video",
            "problem_type": e["otype"],
            "problem_id": e["problem_id"],
            "evid_case": e["evid_case"],
            "data_source": e["data_source"],
            "view": e["view"],
            "direct_acc": e["direct_acc"],
        }
        for k in SIGNAL_FIELDS:
            extra[k] = e["signals"].get(k)
        meta = meta_map.get(e["video"])       # src_fps / src_total_frames / src_duration / frames_indices
        if meta:
            extra.update(meta)
        else:
            rows_no_meta += 1
        rows.append({
            "data_source": e["data_source"],
            "prompt": [{"role": "user", "content": e["content"]}],
            "ability": "video_reasoning",
            "reward_model": {"ground_truth": e["gt"], "style": "rule"},
            "extra_info": extra,
            "videos": [{"type": "video", "video": paths}],
            "images": [],
            "media": [],
        })

    df = pd.DataFrame(rows, columns=["data_source", "prompt", "ability", "reward_model",
                                     "extra_info", "images", "videos", "media"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)

    # 3a) 真实时间戳查找表: verl 的 AgentLoop 调的是【类方法】process_multi_modal_info,
    #     只拿得到 messages、拿不到 extra_info, 所以 ts_dataset.py 得靠帧目录名查表。
    ts_map_path = args.ts_map
    if ts_map_path is None:
        ts_map_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "ts_map.json")
    n_ts = 0
    if ts_map_path:
        tsm = {}
        for r in rows:
            frames0 = r["videos"][0]["video"]
            ei = r["extra_info"]
            if not frames0 or "src_fps" not in ei:
                continue
            tsm[os.path.basename(os.path.dirname(frames0[0]))] = {
                "fps": float(ei["src_fps"]),
                "frames_indices": [int(x) for x in ei["frames_indices"]],
                "total_num_frames": float(ei["src_total_frames"]),
                "duration": float(ei["src_duration"]),
            }
        with open(ts_map_path, "w") as f:
            json.dump(tsm, f)
        n_ts = len(tsm)

    # 3b) 可读预览 jsonl(帧路径只留数量, 避免超长)
    preview = args.preview_jsonl
    if preview is None:
        preview = os.path.splitext(args.out)[0] + ".preview.jsonl"
    if preview:
        with open(preview, "w") as f:
            for r in rows:
                rp = dict(r)
                v0 = rp["videos"][0]
                rp["videos"] = [{"type": v0["type"], "n_frames": len(v0["video"]),
                                 "frame0": v0["video"][0] if v0["video"] else None}]
                f.write(json.dumps(rp, ensure_ascii=False) + "\n")

    # 4) 统计
    print("=" * 64)
    print(f"[done] 写出 {len(df)} 行 -> {args.out}")
    if preview:
        print(f"       预览 -> {preview}")
    if ts_map_path:
        print(f"       ★时间戳查找表 {n_ts} 条 -> {ts_map_path}")
        print(f"        训练侧要挂: data.custom_cls.path=<repo>/verl_ext/ts_dataset.py "
              f"data.custom_cls.name=RealTimestampDataset, 且 ray start 之前 "
              f"export EVID_TS_MAP={ts_map_path}")
    if rows_no_meta:
        print(f"  ★⚠ {rows_no_meta} 行没拿到真实时间信息(extra_info 缺 src_fps), "
              f"这些行在训练时会退回伪造时间戳")
    print(f"  view 分布: {dict(Counter(r['extra_info']['view'] for r in rows))}")
    print(f"  题型分布: {dict(Counter(r['extra_info']['problem_type'] for r in rows))}")
    print(f"  data_source 分布: {dict(Counter(str(r['data_source']).split('/')[0] for r in rows))}")
    print(f"  抽帧数/视频: {nf}   (每行 videos[0].video 帧数应=={nf})")
    _per = Counter(r["extra_info"]["problem_id"] for r in rows)
    _dist = dict(sorted(Counter(_per.values()).items()))
    print(f"  ★每题条数分布(必须只有 2): {_dist}")
    print(f"  跳过: GT源无pid={skip_no_src}  无答案={skip_no_ans}  无视频路径={skip_no_video}  "
          f"prompt切不出题面={skip_unparsed}  抽帧失败={skip_frame_fail}  "
          f"因配对视图失败被连带丢弃={skip_broken_pair}")
    print(f"  总耗时 {(time.time()-t0)/60:.1f}min")
    if rows:
        s = rows[0]
        print("-" * 64)
        print("样例行[0]:")
        print(f"  data_source : {s['data_source']}")
        print(f"  view        : {s['extra_info']['view']}  gt={s['reward_model']['ground_truth']}")
        print(f"  signals     : " + ", ".join(f"{k}={s['extra_info'][k]}" for k in SIGNAL_FIELDS))
        print(f"  videos[0]   : type={s['videos'][0]['type']} n_frames={len(s['videos'][0]['video'])}")
        print(f"  prompt      : {s['prompt'][0]['content'][:160]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
