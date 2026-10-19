#!/usr/bin/env python3
"""第一步直答: 在预抽的精确帧上, 每题用 Qwen3-VL-8B 直答 n 次(默认 8)。

prompt = 输入 jsonl 里的 direct prompt(build_prompt4rollout_direct.py 产出)
输入   = 帧 jpg 列表 + 真实时间戳(用 meta.json 的 fps / frames_indices / total 覆盖 qwen_vl_utils 现造的元数据,
         与训练端 RealTimestampDataset 同一套构造)
推理   = 离线 vLLM LLM.generate; 采样 T=1.0 top_p=1.0 top_k=20 max_tokens=2048
输出   = <out>/rollout/rollout_direct.shard<k>.jsonl, 每行一个 rollout
         (problem_id / data_source / video / prompt / rollout_idx / response / prompt_tokens), 交给 eval_rollout_acc_rule.py 打分
分块   = 每 CHUNK 题 预处理 -> 生成 -> 落盘; 断点续跑: 已写满 n 条的题跳过
"""
import argparse, collections, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recurate_extract_exact import EFS, ROOT, VALID, NF, local, QWEN3VL# noqa: E402
import assemble_trainset4verl as A                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--valid", default=VALID)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="只做预处理并打印, 不起模型")
    a = ap.parse_args()

    qs = [json.loads(l) for l in open(a.valid)]
    qs.sort(key=lambda q: int(q["problem_id"]))
    qs = [q for i, q in enumerate(qs) if i % a.nshard == a.shard]
    fp = os.path.join(a.out, "rollout", f"rollout_direct.shard{a.shard}.jsonl")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    done = collections.Counter()
    if os.path.exists(fp):
        for l in open(fp):
            try: done[json.loads(l)["problem_id"]] += 1
            except Exception: pass                  # noqa: E722  截断的最后一行
    todo = [q for q in qs if done[q["problem_id"]] < a.n_samples]
    if a.limit:
        todo = todo[:a.limit]
    froot = os.path.join(a.out, "frames")
    print(f"[direct] shard {a.shard}/{a.nshard}: 本分片 {len(qs)} 题, 已完成 {len(qs) - len(todo)}, 待跑 {len(todo)}", flush=True)

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    MODEL = os.environ.get("PROBE_MODEL", QWEN3VL)
    if not os.path.isfile(os.path.join(MODEL, "config.json")):
        MODEL = QWEN3VL
    proc = AutoProcessor.from_pretrained(MODEL)
    ps = proc.image_processor.patch_size

    def prep(q):
        d = os.path.join(froot, f"{A.vid_key(q['video'])}__{NF}f")
        mf = os.path.join(d, "meta.json")
        try:
            m = json.load(open(mf))
        except Exception:                           # noqa: BLE001  不存在或写到一半
            return q, None, "无帧(meta.json 缺失或损坏)"
        frames = [os.path.join(d, f"{k:03d}.jpg") for k in range(NF)]
        msgs = [{"role": "user", "content": [{"type": "video", "video": frames},
                                             {"type": "text", "text": q["prompt"]}]}]
        _, vid, kw = process_vision_info(msgs, image_patch_size=ps, return_video_kwargs=True,
                                         return_video_metadata=True)
        vt, md = vid[0]
        n = int(vt.shape[0])
        idx = [int(x) for x in m["frames_indices"]][:n]
        while len(idx) < n:
            idx.append(idx[-1])
        vals = {"fps": float(m["src_fps"]), "frames_indices": idx, "total_num_frames": float(m["src_total_frames"])}
        if isinstance(md, dict):
            md.update(vals)
        else:
            for k, v in vals.items():
                setattr(md, k, v)
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return q, {"prompt": text, "multi_modal_data": {"video": vid}, "mm_processor_kwargs": kw}, None

    if a.dry_run:
        with ThreadPoolExecutor(8) as ex:
            for q, inp, err in ex.map(prep, todo[:4]):
                v = inp["multi_modal_data"]["video"][0] if inp else None
                print(f"  pid={q['problem_id']} 旧acc={q['acc']} 旧view={q['view']} err={err} "
                      f"帧={tuple(v[0].shape) if v else None} fps={v[1]['fps'] if v else None} "
                      f"idx[:3]={v[1]['frames_indices'][:3] if v else None}", flush=True)
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, tensor_parallel_size=int(os.environ.get("TP", "1")),
              gpu_memory_utilization=float(os.environ.get("MEM_UTIL", "0.85")),
              trust_remote_code=True, max_model_len=24576, limit_mm_per_prompt={"video": 1},
              mm_processor_cache_gb=0, seed=0)
    sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=20, max_tokens=2048, n=a.n_samples)
    print(f"[direct] vLLM ready model={MODEL}", flush=True)

    CH = int(os.environ.get("CHUNK", "64"))
    t0 = time.time(); n_done = 0; miss = collections.Counter()
    with open(fp, "a") as f, ThreadPoolExecutor(int(os.environ.get("PREP_WORKERS", "16"))) as ex:
        for c0 in range(0, len(todo), CH):
            batch, meta = [], []
            for q, inp, err in ex.map(prep, todo[c0:c0 + CH]):
                if err:
                    miss[err] += 1; continue
                batch.append(inp); meta.append(q)
            for q, o in zip(meta, llm.generate(batch, sp, use_tqdm=False)):
                for k, c in enumerate(o.outputs):
                    f.write(json.dumps({"problem_id": q["problem_id"], "data_source": q["data_source"],
                                        "video": q["video"], "prompt": q["prompt"], "rollout_idx": k,
                                        "response": c.text, "prompt_tokens": len(o.prompt_token_ids)},
                                       ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(meta)
            el = time.time() - t0
            print(f"[direct] 生成 {min(c0 + CH, len(todo))}/{len(todo)} 题  {el/60:.1f}分  "
                  f"预计总 {el/max(n_done,1)*len(todo)/60:.1f}分  缺帧 {dict(miss)}", flush=True)
    print(f"[direct] ★ shard{a.shard} 完成 {n_done} 题  缺帧 {dict(miss)}  用时 {(time.time()-t0)/60:.1f} 分 -> {fp}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
