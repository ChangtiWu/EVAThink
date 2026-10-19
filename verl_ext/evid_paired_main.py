"""训练入口 —— 与 verl.trainer.main_ppo 等价, 只把 train sampler 换成配对采样。

用法
    export PYTHONPATH="<EVAThink>/verl_ext:${PYTHONPATH:-}"     # 追加, 不要覆盖(Ascend 上 CANN 路径在里面)
    python -m evid_paired_main <与 main_ppo 相同的 Hydra 参数>

设计
    · 复用 verl 自己的 @hydra.main(config_path="config", config_name="ppo_trainer")。
    · 只替换 mp.run_ppo 的默认 task_runner_class, 不动 mp.TaskRunner(原因见 paired_task_runner.py 文件头)。
    · PairedTaskRunner 定义在独立可导入模块里, 按引用序列化。

EVID_PAIR_DISABLE=1 时退回 verl 默认 RandomSampler。
"""

from __future__ import annotations

import os
import sys

import ray

import verl.trainer.main_ppo as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_sampler import log  # noqa: E402
from paired_task_runner import PairedTaskRunner  # noqa: E402


def main():
    original_run_ppo = mp.run_ppo

    def run_ppo_paired(config, task_runner_class=None):
        if task_runner_class is None:
            task_runner_class = ray.remote(num_cpus=1)(PairedTaskRunner)
            log(f"★ task_runner_class = {PairedTaskRunner.__module__}.{PairedTaskRunner.__name__}")
        return original_run_ppo(config, task_runner_class=task_runner_class)

    mp.run_ppo = run_ppo_paired          # mp.main 内部按模块全局查 run_ppo,改这里即可
    mp.main()


if __name__ == "__main__":
    main()
