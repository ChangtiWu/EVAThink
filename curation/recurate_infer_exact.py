#!/usr/bin/env python3
"""通用离线推理: 读 {problem_id, video, prompt, ...} jsonl, 在预抽的精确帧上跑 vLLM, 每题 n 条。

用于: 证据定位标注(Qwen3.5-27B, T=0, n=1, max_tokens=256) / 对比视频直答(n=8, max_tokens=512)
      / sparse 视角慢思考(n=8, max_tokens=1024)。
输入帧 = recurate_extract_exact.py 抽的 <out>/frames/<vid_key(video)>__32f/{000..031.jpg, meta.json},
      用 meta.json 的真实 fps / frames_indices / total 覆盖 qwen_vl_utils 现造的元数据 = 真实时间戳。
输出 = <out>/rollout/<name>.shard<k>.jsonl, 每行一个 rollout: 输入的全部字段原样带上 + rollout_idx / response / prompt_tokens。
分块 = 每 CHUNK 题 预处理 -> 生成 -> 落盘; 断点续跑按 problem_id 计数, 已满 n 条的跳过。
"""
import argparse, collections, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recurate_extract_exact import ROOT, NF, QWEN3VL# noqa: E402
import assemble_trainset4verl as A                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl, 每行至少 problem_id / video / prompt")
    ap.add_argument("--out", required=True, help="recurate 输出根目录(下有 frames/ rollout/)")
    ap.add_argument("--name", required=True, help="输出文件名前缀, 如 anno_gt / rollout_direct_contra / rollout_slow")
    ap.add_argument("--model", default=os.environ.get("PROBE_MODEL", QWEN3VL))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-thinking", action="store_true",
                    help="套模板时传 enable_thinking=False(Qwen3.5 不显式关就默认先思考; 旧标注客户端也是显式关的)。"
                         "Qwen3-VL-8B-Instruct 模板里没有这个分支, 不受影响")
    ap.add_argument("--dry-run", action="store_true", help="只做预处理并打印, 不起模型")
    a = ap.parse_args()
    tpl_kw = {"enable_thinking": False} if a.no_thinking else {}

    qs = [json.loads(l) for l in open(a.input) if l.strip()]
    qs.sort(key=lambda q: int(q["problem_id"]))
    qs = [q for i, q in enumerate(qs) if i % a.nshard == a.shard]
    fp = os.path.join(a.out, "rollout", f"{a.name}.shard{a.shard}.jsonl")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    done = collections.Counter()
    if os.path.exists(fp):
        for l in open(fp):
            try: done[json.loads(l)["problem_id"]] += 1
            except Exception: pass                   # noqa: E722  截断的最后一行
    todo = [q for q in qs if done[q["problem_id"]] < a.n]
    if a.limit:
        todo = todo[:a.limit]
    froot = os.path.join(a.out, "frames")
    print(f"[infer:{a.name}] shard {a.shard}/{a.nshard}: 本分片 {len(qs)} 题, 已完成 {len(qs) - len(todo)}, 待跑 {len(todo)}  "
          f"n={a.n} T={a.temperature} top_k={a.top_k} max_tokens={a.max_tokens}", flush=True)

    model = a.model
    if not os.path.isfile(os.path.join(model, "config.json")):
        model = QWEN3VL
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(model)
    ps = proc.image_processor.patch_size

    def prep(q):
        d = os.path.join(froot, f"{A.vid_key(q['video'])}__{NF}f")
        try:
            m = json.load(open(os.path.join(d, "meta.json")))
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
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tpl_kw)
        return q, {"prompt": text, "multi_modal_data": {"video": vid}, "mm_processor_kwargs": kw}, None

    if a.dry_run:
        with ThreadPoolExecutor(8) as ex:
            for q, inp, err in ex.map(prep, todo[:4]):
                v = inp["multi_modal_data"]["video"][0] if inp else None
                print(f"  pid={q['problem_id']} err={err} 帧={tuple(v[0].shape) if v else None} "
                      f"fps={v[1]['fps'] if v else None} idx[:3]={v[1]['frames_indices'][:3] if v else None}", flush=True)
                if inp: print("     prompt 结尾:", repr(inp["prompt"][-60:]), flush=True)
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=model, tensor_parallel_size=int(os.environ.get("TP", "1")),
              gpu_memory_utilization=float(os.environ.get("MEM_UTIL", "0.85")),
              trust_remote_code=True, max_model_len=a.max_model_len, limit_mm_per_prompt={"video": 1},
              mm_processor_cache_gb=0, seed=0)
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k if a.temperature > 0 else -1,
                        max_tokens=a.max_tokens, n=a.n)
    print(f"[infer:{a.name}] vLLM ready model={model}", flush=True)

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
                    f.write(json.dumps({**q, "rollout_idx": k, "response": c.text,
                                        "prompt_tokens": len(o.prompt_token_ids)}, ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(meta)
            el = time.time() - t0
            print(f"[infer:{a.name}] 生成 {min(c0 + CH, len(todo))}/{len(todo)} 题  {el/60:.1f}分  "
                  f"预计总 {el/max(n_done,1)*len(todo)/60:.1f}分  缺帧 {dict(miss)}", flush=True)
    print(f"[infer:{a.name}] ★ shard{a.shard} 完成 {n_done} 题  缺帧 {dict(miss)}  用时 {(time.time()-t0)/60:.1f} 分 -> {fp}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
