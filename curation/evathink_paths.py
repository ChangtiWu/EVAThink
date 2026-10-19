"""Shared locations for the curation scripts. Override with environment variables.

EVATHINK_WORK        working directory for every intermediate file (default: <repo>/work)
EVATHINK_VIDEO_ROOT  root that the relative video paths in datapool_mc.jsonl are resolved against
EVATHINK_QWEN3VL     Qwen3-VL-8B-Instruct checkpoint used for the direct / slow-thinking rollouts
"""
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORK = os.environ.get("EVATHINK_WORK", os.path.join(_REPO, "work"))
VIDEO_ROOT = os.environ.get("EVATHINK_VIDEO_ROOT", "")
QWEN3VL = os.environ.get("EVATHINK_QWEN3VL", "Qwen/Qwen3-VL-8B-Instruct")
