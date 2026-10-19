"""PairedTaskRunner —— 覆写 TaskRunner.run, 把 train sampler 换成配对采样。

放在独立的可导入模块里: 入口以 `python -m evid_paired_main` 启动时模块名是 `__main__`, Ray worker 里不可导入,
cloudpickle 只能按值序列化定义在其中的类(方法的 globals 变成快照)。本模块名固定为 paired_task_runner,
类按引用序列化。

覆写整个 run() 并把 sampler 直接传给 RayPPOTrainer(train_sampler=...), 而不是替换 main_ppo 的模块全局
(替换 mp.TaskRunner 会让基类也被按值序列化, 之后对模块属性的修改在 worker 里不可见)。

下面 run() 的函数体复制自 verl v0.8.0 main_ppo.py 的 TaskRunner.run(做法同官方 recipe/dapo/main_dapo.py),
只改了 train_sampler 一处。verl 升级后需要重新对齐; 导入时会检查原版 run() 是否仍有复制时依赖的调用点, 不符则告警。

环境变量
    EVID_PAIR_DISABLE=1   不接管, 退回 verl 默认 sampler
    EVID_PAIR_REQUIRED=0  配对建不起来时允许回退到默认 sampler; 默认 1 = 报错中止
"""

from __future__ import annotations

import inspect
import os
import socket
import sys

from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_sampler import build_paired_sampler, envi, log  # noqa: E402

# 复制自 verl v0.8.0 的 TaskRunner.run。升级后若原版变了,这里必须跟着改。
_EXPECTED_MARKERS = [
    "create_rl_sampler(config.data, train_dataset)",   # 我们替换掉的那一行
    "self.add_teacher_model_resource_pool(config)",    # 复制体里逐行照抄的步骤
    "self.add_ref_policy_worker(config, actor_rollout_cls)",
    "trainer = RayPPOTrainer(",
    "train_sampler=train_sampler,",
    "trainer.init_workers()",
    "trainer.fit()",
]


def _check_base_drift() -> None:
    """原版 run() 若少了我们照抄时依赖的调用点,说明 verl 升级过,复制体可能已过期。"""
    try:
        src = inspect.getsource(TaskRunner.run)
    except Exception:  # noqa: BLE001
        log("⚠ 取不到 TaskRunner.run 源码,跳过漂移检查")
        return
    missing = [m for m in _EXPECTED_MARKERS if m not in src]
    if missing:
        log(f"⚠ verl 的 TaskRunner.run 变了,缺少标记 {missing} —— "
            f"本文件复制的函数体可能已过期,请人工对齐 verl/trainer/main_ppo.py")
    else:
        log("基类 run() 结构标记齐全,复制体与原版一致")


class PairedTaskRunner(TaskRunner):
    """与官方 TaskRunner 行为一致,唯一差别是 train_sampler 用配对采样。"""

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        self.add_reward_model_resource_pool(config)

        self.add_teacher_model_resource_pool(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )

        # ---- 与官方 TaskRunner.run 的唯一差别 ----
        _check_base_drift()
        if envi("EVID_PAIR_DISABLE", 0) == 1:
            log("EVID_PAIR_DISABLE=1 —— 不接管,使用 verl 默认 sampler")
            train_sampler = create_rl_sampler(config.data, train_dataset)
        else:
            train_sampler = build_paired_sampler(config.data, train_dataset)
            if train_sampler is None:
                if envi("EVID_PAIR_REQUIRED", 1) == 1:
                    raise RuntimeError(
                        "[pair_sampler] 配对采样没能建起来,而 EVID_PAIR_REQUIRED=1(默认)。"
                        "宁可中止也不静默跑成随机采样 —— 上一次就是这么白跑的。"
                        "确认数据集每个 problem_id 恰好 2 行;要强行回退请传 EVID_PAIR_REQUIRED=0。"
                    )
                log("⚠ 未能接管,回退 verl 默认 sampler(EVID_PAIR_REQUIRED=0)")
                train_sampler = create_rl_sampler(config.data, train_dataset)
        log(f"最终 train_sampler = {type(train_sampler).__module__}.{type(train_sampler).__name__}")
        # ---- 差别到此为止 ----

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        _install_firsttok_cap()      # EVID_FIRSTTOK_CAP 未设时什么都不做
        trainer.fit()


def _install_firsttok_cap() -> None:
    """首 token 置信度上限正则(见 firsttok_reg.py): 包一层 ray_trainer.compute_advantage。

    compute_advantage 在 driver(= 本 TaskRunner 进程)里执行, fit() 按模块全局名查找它, 所以在 fit() 之前
    替换 verl.trainer.ppo.ray_trainer 的模块属性即可生效(trainer 就是在本进程里建的, 不涉及序列化快照)。
    环境变量(多节点时须在 ray start 之前 export):
        EVID_FIRSTTOK_CAP        对称版上限 c, 如 0.85(两种模式都压); 不设 = 不开启对称版
        EVID_FIRSTTOK_CAP_WRONG  单边版上限 c_wrong, 如 0.5: 只压"模式不对"(dense 思考 / sparse 直答), 模式对的不管;
                                 设了它就用单边版(优先于 EVID_FIRSTTOK_CAP)
        EVID_FIRSTTOK_CAP_RIGHT  单边版里给"模式对的"加的宽松上限, 如 0.85; 不设 = 对的模式不管
        EVID_FIRSTTOK_BETA       强度 β, 默认 10
    单边版的视角/模式取自奖励输出 evid_dense / evid_thinking / format_reward —— ray_trainer.fit() 在调
    compute_advantage 之前已把 reward_extra_infos 写进 batch.non_tensor_batch。
    """
    cap_s = os.environ.get("EVID_FIRSTTOK_CAP", "").strip()
    wrong_s = os.environ.get("EVID_FIRSTTOK_CAP_WRONG", "").strip()
    if not cap_s and not wrong_s:
        log("首 token 正则: 未开启(EVID_FIRSTTOK_CAP / EVID_FIRSTTOK_CAP_WRONG 均未设)")
        return
    import numpy as np
    import verl.trainer.ppo.ray_trainer as rt
    from firsttok_reg import firsttok_cap_penalty, firsttok_wrongmode_penalty, summarize, summarize_views

    beta = float(os.environ.get("EVID_FIRSTTOK_BETA", "10"))
    oneside = bool(wrong_s)
    cap = float(wrong_s if oneside else cap_s)
    right_s = os.environ.get("EVID_FIRSTTOK_CAP_RIGHT", "").strip()
    cap_right = float(right_s) if (oneside and right_s) else None
    orig = rt.compute_advantage
    if getattr(orig, "_evid_firsttok", False):
        return
    state = {"step": 0}

    def _flag(nt, key, thr):
        if key not in nt:
            raise KeyError(f"首 token 单边正则需要奖励输出 {key}, non_tensor_batch 里没有(现有: {sorted(nt.keys())})")
        return np.asarray(nt[key], dtype=np.float32) > thr

    def compute_advantage(data, *args, **kwargs):
        data = orig(data, *args, **kwargs)
        keys = data.batch.keys()
        mask = data.batch["response_mask"] if "response_mask" in keys else None
        state["step"] += 1
        if oneside:
            nt = data.non_tensor_batch
            dense, think = _flag(nt, "evid_dense", 0.5), _flag(nt, "evid_thinking", 0.5)
            fmt_ok = _flag(nt, "format_reward", -0.5)
            adv, pen, p0, _ = firsttok_wrongmode_penalty(
                data.batch["advantages"], data.batch["old_log_probs"], mask, dense, think, fmt_ok, cap, beta,
                cap_right=cap_right)
            msg = summarize_views(pen, p0, dense, think, cap, cap_right)
        else:
            adv, pen, p0 = firsttok_cap_penalty(data.batch["advantages"], data.batch["old_log_probs"], mask, cap, beta)
            msg = summarize(pen, p0, cap)
        data.batch["advantages"] = adv
        log(f"首 token 正则 step{state['step']}: {msg}")
        return data

    compute_advantage._evid_firsttok = True
    compute_advantage._evid_orig = orig
    rt.compute_advantage = compute_advantage
    if oneside:
        right_txt = f"对的模式上限 c_right={cap_right}" if cap_right is not None else "对的模式不设上限"
        log(f"★ 首 token 单边正则已开启: 不对的模式(dense 思考 / sparse 直答) c_wrong={cap} β={beta}, {right_txt}")
    else:
        log(f"★ 首 token 正则已开启: c={cap} β={beta} (A[:,0] -= β·max(0, log p0 − log c))")
