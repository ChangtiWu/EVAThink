"""Two runtime patches loaded automatically when verl_ext/ is the first entry on PYTHONPATH.

1) qwen_vl_utils.fetch_video ndarray -> list shim (always on).
   The training parquet stores each view as a list of 32 frame paths. After the parquet round trip the list comes
   back as a numpy array, and vanilla qwen_vl_utils asserts isinstance(list/tuple) and crashes.

2) Chunked entropy (only with EVID_ENTROPY_CHUNK=1; size EVID_ENTROPY_CHUNK_SIZE, default 2048).
   With use_remove_padding=False, verl v0.8.0 computes the entropy on the padded [rows, longest length, vocab]
   logits in one fp32 softmax. Dynamic batching packs rows by real tokens but pads to the longest one, which can
   allocate ~16 GiB at once and OOM in compute_log_prob on 64 GB cards. verl's own
   entropy_from_logits_with_chunking only applies to the rmpad branch. This replaces
   verl.utils.torch_functional.entropy_from_logits with a row-chunked version (same math, same dtype; falls back to
   the original when gradients are needed or the input is not contiguous).

If your environment already has a sitecustomize.py, import this module from it instead.
"""

import builtins
import os
import sys

# ---------- 1) ndarray -> list shim ----------
try:
    import numpy as np
    import qwen_vl_utils.vision_process as _vp

    if not getattr(_vp.fetch_video, "_ndfix", False):
        _orig_fv = _vp.fetch_video

        def fetch_video(ele, *a, **k):
            v = ele.get("video")
            if isinstance(v, np.ndarray):
                ele = dict(ele)
                ele["video"] = [str(x) for x in v.tolist()]
            return _orig_fv(ele, *a, **k)

        fetch_video._ndfix = True
        _vp.fetch_video = fetch_video
except Exception as e:  # noqa: BLE001
    print(f"[evathink_site] ndarray shim skipped: {type(e).__name__}: {e}", file=sys.stderr)


# ---------- 2) chunked entropy ----------
_EC_ON = os.environ.get("EVID_ENTROPY_CHUNK", "0").strip() == "1"


def _install_entropy_chunk(tf):
    try:
        import torch

        orig = tf.entropy_from_logits
        if getattr(orig, "_evid_orig", None) is not None:
            return
        chunk = int(os.environ.get("EVID_ENTROPY_CHUNK_SIZE", "2048"))

        def entropy_from_logits(logits):
            if (logits.dim() < 2 or not logits.is_contiguous()
                    or (torch.is_grad_enabled() and logits.requires_grad)):
                return orig(logits)
            flat = logits.view(-1, logits.shape[-1])
            out = torch.empty(flat.shape[0], dtype=logits.dtype, device=logits.device)
            for i in range(0, flat.shape[0], chunk):
                c = flat[i:i + chunk]
                pd = torch.nn.functional.softmax(c, dim=-1)
                out[i:i + chunk] = torch.logsumexp(c, dim=-1) - torch.sum(pd * c, dim=-1)
            return out.view(logits.shape[:-1])

        entropy_from_logits._evid_orig = orig
        tf.entropy_from_logits = entropy_from_logits
        print(f"[evathink_site] chunked entropy installed (chunk={chunk}, pid={os.getpid()})", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[evathink_site] chunked entropy not installed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


if _EC_ON:
    _real_import = builtins.__import__

    def _imp(name, *a, **k):
        m = _real_import(name, *a, **k)
        if not getattr(builtins, "_evathink_ec_done", False):
            # wait until verl.utils.torch_functional is fully initialized (its own `import torch` also lands here)
            _tf = sys.modules.get("verl.utils.torch_functional")
            if (_tf is not None and hasattr(_tf, "entropy_from_logits")
                    and not getattr(getattr(_tf, "__spec__", None), "_initializing", False)):
                builtins._evathink_ec_done = True
                _install_entropy_chunk(_tf)
        return m

    builtins.__import__ = _imp
