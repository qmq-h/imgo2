"""Offline kinematics checks plus an optional CPU Torch AMP update regression.

Run: python -m unittest discover -s tests -p test_amp_alignment.py -v
"""

import ast
import importlib.util
import math
import re
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/tools"))
from audit_amp_dataset import audit, axis_rotation, forward_kinematics, read_chain, rotate


class KinematicsTests(unittest.TestCase):
    def test_axis_rotation_known_right_angle(self):
        result = rotate(axis_rotation([0, 0, 1], math.pi / 2), [1, 0, 0])
        for actual, expected in zip(result, [0, 1, 0]):
            self.assertAlmostEqual(actual, expected)

    def test_zero_pose_leg_length(self):
        urdf = ET.parse(ROOT.parent / "imgo2_description/urdf/imgo2.urdf").getroot()
        result = forward_kinematics(read_chain(urdf, "FL_FOOT"), [0, 0, 0])
        for actual, expected in zip(result, [0.2205, 0.16185, -0.42575]):
            self.assertAlmostEqual(actual, expected)

    def test_dataset_matches_urdf_without_middle_leg_swap(self):
        result = audit(ROOT / "datasets/imgo2_motion", ROOT.parent / "imgo2_description/urdf/imgo2.urdf")
        for row in result["motions"]:
            with self.subTest(motion=row["file"]):
                error = row["fk_coordinate_rmse_m_by_order"]
                self.assertLess(error["FL_FR_RL_RR"], 0.005)
                self.assertGreater(error["FL_RL_FR_RR"], 0.20)


class RewardContractTests(unittest.TestCase):
    # These two guards exist because the AMP task was unconstructible without them:
    # (a) mdp.reset_amp_reference_state did not exist (amp_events was not imported anywhere),
    #     so importing amp_env_cfg raised AttributeError before any run could start;
    # (b) base_height_l2 kept RewardsCfg's body_names="" and Isaac Lab raises
    #     "Not all regular expressions are matched!" while resolving it at env construction.
    # Neither is reachable from a simulator-free test, so they are asserted statically here.

    CONFIG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py"
    AMP_EVENTS = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/amp_events.py"
    REWARDS = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/rewards.py"

    def test_runners_keep_checkpoint_iter_in_sync(self):
        """checkpoint 的 'iter' 必须来自「当前轮」，否则 resume 会从错误位置继续。

        AMP 与 PPO 的 learn() 曾在循环内从不更新 self.current_learning_iteration，
        于是它一直是该次运行的起点（通常 0）：save() 把它写进 checkpoint，
        load() 又用它作为续训起点 —— 结果是续训从 0 重跑，且第一次保存覆盖 model_0.pt。
        HIM runner 一直是对的（循环内 self.current_learning_iteration = it）。
        2026-09-17 实测 2026-09-17_17-19-04 的 20 个 checkpoint 全部 iter=0，即此缺陷。
        """
        runners = ROOT / "scripts/rl_lab/rl_lab/runners"
        for name in ("amp_on_policy_runner.py", "ppo_on_policy_runner.py", "him_on_policy_runner.py"):
            with self.subTest(runner=name):
                src = (runners / name).read_text(encoding="utf-8")
                self.assertIn("self.current_learning_iteration = it", src,
                              f"{name} 的 learn() 循环内未同步迭代计数，checkpoint 的 iter 会失真")
                # save() 必须写这个字段，否则 load() 无从恢复
                self.assertIn("'iter': self.current_learning_iteration", src)

    def test_runner_progress_line_and_final_save_use_absolute_totals(self):
        """进度条分母 / ETA / 收尾 checkpoint 名都必须用「总轮数」，不能用「当前轮 + 总轮数」。

        2026-09-18 用户看到控制台显示 `Learning iteration 4391/14391`：分母被写成
        `self.current_learning_iteration + locs['num_learning_iterations']`，而 AMP-08 的修复
        已经让循环内把 `current_learning_iteration` 更新成**当前轮** ⇒ 看起来像把轮数加了两遍。
        同一个错误还有两个连带后果：
          * ETA 用 `num_learning_iterations - it`，resume 时会少算 `start_iter` 那么多轮的时间；
          * 循环结束后 `self.current_learning_iteration += num_learning_iterations` 会再加一次，
            于是收尾保存写成 `model_19999.pt` / `iter=19999`（10000 轮 + 起点 9999），而
            `--resume` 默认取「最新 checkpoint」时正好会踩到它。
        HIM runner 一直是对的（用 `start_iter` / `locs['tot_iter']`），所以三个 runner 一起检查。
        """
        runners = ROOT / "scripts/rl_lab/rl_lab/runners"
        buggy_denominator = "self.current_learning_iteration + locs['num_learning_iterations']"
        buggy_eta = re.compile(r"\(\s*locs\['num_learning_iterations'\]\s*-\s*locs\['it'\]")
        for name in ("amp_on_policy_runner.py", "ppo_on_policy_runner.py", "him_on_policy_runner.py"):
            with self.subTest(runner=name):
                src = (runners / name).read_text(encoding="utf-8")
                self.assertIn("self.current_learning_iteration = it", src)
                # 分母 = 总轮数
                self.assertIn("{locs['it']}/{locs['tot_iter']}", src)
                self.assertNotIn(buggy_denominator, src)
                # ETA：平均每轮耗时按「本次已跑轮数」算；剩余轮数必须含起点
                self.assertIn("locs['it'] - locs['start_iter'] + 1", src)
                self.assertIsNone(buggy_eta.search(src), f"{name} 的 ETA 剩余轮数漏了起点")
                # 循环内同步过计数的两个 runner：收尾不能再加一次，必须落在 tot_iter
                if name != "him_on_policy_runner.py":
                    self.assertNotIn("self.current_learning_iteration += num_learning_iterations", src)
                    self.assertIn("self.current_learning_iteration = tot_iter", src)
                    self.assertIn("locs['tot_iter'] - locs['it']", src)

    def test_amp_runner_logs_feet_air_metrics(self):
        """训练日志必须能看到「滞空」物理量，否则每次确认步频都要占 GPU 回放。

        2026-09-18 加的 `feet_air_time` 奖励项只会体现在 Episode_Reward/feet_air_time（奖励值），
        看不到物理量；这里锁定 runner 额外记录两条与 eval_gait.py 同口径的标量，
        并要求取不到接触传感器时安全降级（返回 None 而不是抛错）。
        """
        src = (ROOT / "scripts/rl_lab/rl_lab/runners/amp_on_policy_runner.py").read_text(encoding="utf-8")
        self.assertIn("AMP/mean_last_air_time_s", src)
        self.assertIn("AMP/mean_air_time_fraction", src)
        self.assertIn("def _feet_air_metrics", src)
        # 口径：四足 last_air_time 均值 + 当前腾空足比例
        self.assertIn('find_bodies(".*_FOOT", preserve_order=True)', src)
        self.assertIn("last_air_time", src)
        self.assertIn("current_air_time", src)
        # 鲁棒性：没有接触传感器/没有 track_air_time 时不能崩
        self.assertIn("return None, None", src)
        self.assertIn("if mean_air_time is None:", src)
        # 控制台也要打印（终端里能直接看）
        self.assertIn("{air_line}", src)

    def test_amp_reference_reset_term_is_imported_and_defined(self):
        cfg_src = self.CONFIG.read_text(encoding="utf-8-sig")
        events_src = self.AMP_EVENTS.read_text(encoding="utf-8-sig")
        self.assertIn("def reset_amp_reference_state", events_src)
        # the config must import amp_events itself; the mdp package does not re-export it
        self.assertIn("from imgo2_rl.tasks.manager_based.locomotion.velocity.mdp import amp_events", cfg_src)
        # and must reference the function through that module, not through `mdp`
        self.assertIn("func=mdp_amp.reset_amp_reference_state", cfg_src)
        self.assertNotIn("func=mdp.reset_amp_reference_state", cfg_src)
        self.assertIn("# from .amp_events import *", (self.AMP_EVENTS.parent / "__init__.py").read_text(
            encoding="utf-8-sig"), "if mdp re-exports amp_events, this guard needs revisiting")

    def test_amp_task_terms_match_reference_per_step_scale(self):
        """AMP-06：任务项的每步系数必须精确，高度项必须仍是任务里最强的项。

        单位语义：AMP 风格奖励是**每步**（runner 里直接算），而 Isaac Lab 的 RewardManager
        计算 `term × weight × dt`，weight 的语义是「每秒」。因此 weight 一律写成
        `每步系数 / step_dt`，由代码换算。

        2026-09-17 第二次调参（当前值）：速度项 1.0→4.0、高度项 -10→-5，并补
        `lin_vel_z_l2` / `ang_vel_xy_l2` / `joint_pos_limits` 三个轻量姿态约束。
        依据：a1 论文代码的真实每步系数是 1.67（配置里的 50 还会被 legged_robot 再乘 dt），
        go2 用 4.0/2.0 + 辅助项。历史见 README AMP-06 与 docs/amp_experiments_2026-09-17.md。
        """
        path = self.CONFIG
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)  # 27 个上游文件带 UTF-8 BOM
        cfg_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Imgo2AmpMoveEnvCfg")
        method = next(n for n in cfg_class.body if isinstance(n, ast.FunctionDef) and n.name == "_keep_only_amp_task_rewards")
        # 方法里要用到模块级常量与两个 Isaac Lab 配置类。离线环境没有 isaaclab，所以
        # 常量从源码 AST 里取（保证测的是真实取值，而不是测试自己复制的一份），
        # `RewTerm` / `SceneEntityCfg` 用轻量桩代替（只关心它们携带的字段）。
        consts = {}
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                continue
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue  # 例如 RLAMP_TRACK_LIN_VEL_RAW = 1.5 * 1.0 / (0.005 * 6)
        self.assertIn("FEET_AIR_TIME_THRESHOLD_S", consts, "阈值常量必须留在模块级（供测试与标定复用）")

        def deficit_marker(*args, **kwargs):
            return 0.0

        namespace = {
            **consts,
            "RewTerm": lambda **kwargs: SimpleNamespace(**kwargs),
            "SceneEntityCfg": lambda name, **kwargs: SimpleNamespace(name=name, **kwargs),
            "mdp": SimpleNamespace(feet_air_time_deficit=deficit_marker),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

        expected_per_step = {
            "track_lin_vel_xy_exp": 4.0,
            "track_ang_vel_z_exp": 2.0,
            "base_height_l2": -5.0,
            "lin_vel_z_l2": -1.0,
            "ang_vel_xy_l2": -0.05,
            "joint_pos_limits": -2.0,
            "feet_air_time_deficit": -3.0,
        }
        for dt in (0.01, 0.02, 0.04):
            with self.subTest(step_dt=dt):
                terms = SimpleNamespace(**{name: SimpleNamespace(weight=0.0, params={}) for name in (
                    *expected_per_step, "feet_slide")})
                terms.base_height_l2.params["asset_cfg"] = SimpleNamespace(body_names="")
                cfg = SimpleNamespace(rewards=terms, sim=SimpleNamespace(dt=dt / 4), decimation=4,
                                      base_link_name="base",
                                      foot_body_names=["FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT"])
                namespace[method.name](cfg)

                # 其它奖励项仍被过滤掉
                self.assertIsNone(terms.feet_slide)

                # 高度项保留：目标 0.30 m、无高度扫描传感器、body 名已解析成具体 body
                self.assertAlmostEqual(terms.base_height_l2.params["target_height"], 0.30)
                self.assertIsNone(terms.base_height_l2.params["sensor_cfg"])
                self.assertEqual(terms.base_height_l2.params["asset_cfg"].body_names, ["base"])

                # 滞空缺口项（2026-10-10）：阈值取实测参考中位、四足 body 必须显式给
                # （给 "" 会在 Isaac Lab 里抛 "Not all regular expressions are matched!"）。
                air = terms.feet_air_time_deficit
                self.assertIsNotNone(air, "该 AMP 专用项必须活过 keep_rewards 过滤")
                self.assertIs(air.func, deficit_marker, "必须接线到 mdp.feet_air_time_deficit")
                self.assertAlmostEqual(air.params["threshold"], 0.20, places=6)
                self.assertEqual(air.params["command_name"], "base_velocity")
                self.assertEqual(air.params["sensor_cfg"].name, "contact_forces")
                self.assertEqual(air.params["sensor_cfg"].body_names,
                                 ["FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT"])
                self.assertLess(air.weight, 0.0, "函数返回非负缺口 ⇒ 权重必须为负")

                # weight × step_dt 必须精确还原每步系数（防止写回手算魔数或漏乘 dt）
                for name, per_step in expected_per_step.items():
                    self.assertAlmostEqual(getattr(terms, name).weight * dt, per_step, places=6)

                # 关键比例：高度项必须**强于**单个速度项（压住贴地爬行），
                # 但不能强过两个速度项之和（否则会像 Run1 那样压住移动）。
                lin_ps = terms.track_lin_vel_xy_exp.weight * dt
                ang_ps = terms.track_ang_vel_z_exp.weight * dt
                height_ps = abs(terms.base_height_l2.weight * dt)
                self.assertGreater(height_ps, lin_ps)
                self.assertLess(height_ps, lin_ps + ang_ps)

    def test_amp_reward_weights_are_derived_not_hardcoded(self):
        """防止回归成手写魔数：weight 必须由「每步系数 / step_dt」表达。"""
        src = self.CONFIG.read_text(encoding="utf-8-sig")
        for const in ("TRACK_LIN_VEL_PER_STEP", "TRACK_ANG_VEL_PER_STEP", "BASE_HEIGHT_PER_STEP",
                      "LIN_VEL_Z_PER_STEP", "ANG_VEL_XY_PER_STEP", "JOINT_POS_LIMITS_PER_STEP",
                      "FEET_AIR_TIME_DEFICIT_PER_STEP", "FEET_AIR_TIME_THRESHOLD_S"):
            self.assertIn(const, src)
        # 不应出现直接写死的 weight 数值
        for magic in ("weight = 50.0", "weight = 2500", "weight = -500", "1.0 / step_dt",
                      "0.3 / step_dt", "-10.0 / step_dt", "weight = -150.0", "threshold = 0.5"):
            self.assertNotIn(magic, src)

    def test_amp_feet_air_time_deficit_is_penalty_only(self):
        """2026-10-10（AMP-06 的 P2）：滞空项必须**只罚不奖**，且阈值来自实测参考。

        背景：策略的唯一实质缺陷是周期 0.198 s（参考 0.600 s）。原始 `feet_air_time`
        在 `air > θ` 之后**仍然线性给奖**，等于奖励「滞空越长越好」，可以走出
        高抬腿/原地蹦的捷径；缺口形式 `relu(θ − air)` 在 θ 处精确为 0，只对「迈太快」
        给梯度。阈值必须来自实测：21 份动作逐足腾空 0.19–0.28 s（中位 ≈0.21 s），
        库里默认的 0.5 s 比参考周期还长。
        """
        rewards_src = self.REWARDS.read_text(encoding="utf-8-sig")
        tree = ast.parse(rewards_src)
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("feet_air_time_deficit", funcs, "缺口函数必须定义在 mdp/rewards.py")
        body_src = ast.get_source_segment(rewards_src, funcs["feet_air_time_deficit"])
        # 只罚不奖：clamp(θ − air, min=0)；绝不能出现无下界的 (air − θ)
        self.assertIn("torch.clamp(threshold - last_air_time, min=0.0)", body_src)
        self.assertNotIn("last_air_time - threshold", body_src)
        # 沿用参考的两道门：站定不给信号 + 机身直立度门
        self.assertIn("> 0.1", body_src)
        self.assertIn("projected_gravity_b", body_src)
        # 原始 feet_air_time 必须保持原样（不改成缺口形式，避免同时动两处）
        stock = ast.get_source_segment(rewards_src, funcs["feet_air_time"])
        self.assertIn("(last_air_time - threshold) * first_contact", stock)

        # AMP 配置必须通过 mdp 引用新函数，且阈值/系数是模块级常量
        cfg_src = self.CONFIG.read_text(encoding="utf-8-sig")
        self.assertIn("func=mdp.feet_air_time_deficit", cfg_src)
        self.assertIn("FEET_AIR_TIME_THRESHOLD_S = 0.20", cfg_src)
        self.assertNotIn('"threshold": 0.5', cfg_src)


HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))


@unittest.skipUnless(HAS_TORCH, "CPU regression needs the training environment's torch and numpy")
class AMPUpdateTests(unittest.TestCase):
    def test_raw_statistics_and_normalized_gradient_penalty(self):
        import numpy as np
        import torch

        def load_cpu_module(relative, **dependencies):
            # The package __init__ imports Isaac Lab export utilities. Load the
            # real CPU implementations without those simulator-only imports.
            path = ROOT / "scripts/rl_lab/rl_lab" / relative
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))  # 27 个上游文件带 UTF-8 BOM
            tree.body = [node for node in tree.body if not (
                isinstance(node, ast.ImportFrom) and (node.module or "").startswith("rl_lab"))]
            scope = {"__name__": "amp_cpu_regression", **dependencies}
            exec(compile(tree, str(path), "exec"), scope)
            return scope

        utils = load_cpu_module("utils/utils.py")
        ActorCritic = load_cpu_module("modules/actor_critic.py")["ActorCritic"]
        storage = load_cpu_module("storage/rollout_storage.py",
                                  split_and_pad_trajectories=utils["split_and_pad_trajectories"])
        replay = load_cpu_module("storage/replay_buffer.py")
        AMPPPO = load_cpu_module("algorithms/amp_ppo.py", ActorCritic=ActorCritic,
                                 RolloutStorage=storage["RolloutStorage"],
                                 ReplayBuffer=replay["ReplayBuffer"])["AMPPPO"]
        AMPDiscriminator = load_cpu_module("algorithms/amp_discriminator.py")["AMPDiscriminator"]
        Normalizer = utils["Normalizer"]

        torch.manual_seed(0)
        normalizer = Normalizer(43)
        normalizer.mean[:] = 2.0
        normalizer.var[:] = 4.0
        expert = torch.full((4, 43), 0.3)
        next_expert = torch.full((4, 43), 0.4)

        class ExpertData:
            def feed_forward_generator(self, num_mini_batch, mini_batch_size):
                for _ in range(num_mini_batch):
                    yield expert, next_expert

        actor = ActorCritic(3, 3, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
        discriminator = AMPDiscriminator(86, 2.0, [8], "cpu", 0.1)
        algorithm = AMPPPO(actor, discriminator, ExpertData(), normalizer,
                           device="cpu", amp_replay_buffer_size=16)
        algorithm.init_storage(2, 2, [3], [3], [2])
        obs = torch.zeros(2, 3)
        for _ in range(2):
            algorithm.act(obs, obs, torch.full((2, 43), 0.1))
            algorithm.process_env_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), {},
                                       torch.full((2, 43), 0.2))
        algorithm.compute_returns(obs)
        expected_expert = normalizer.normalize_torch(expert, "cpu")
        expected_next_expert = normalizer.normalize_torch(next_expert, "cpu")
        with patch.object(normalizer, "update", wraps=normalizer.update) as update, \
             patch.object(discriminator, "compute_grad_pen", wraps=discriminator.compute_grad_pen) as penalty:
            losses = algorithm.update()
        self.assertTrue(all(math.isfinite(loss) for loss in losses))
        self.assertEqual(update.call_count, 2)
        np.testing.assert_allclose(update.call_args_list[0].args[0], 0.1)
        np.testing.assert_allclose(update.call_args_list[1].args[0], 0.3)
        torch.testing.assert_close(penalty.call_args.args[0], expected_expert)
        torch.testing.assert_close(penalty.call_args.args[1], expected_next_expert)


@unittest.skipUnless(HAS_TORCH, "CPU regression needs the training environment's torch and numpy")
class FeetAirTimeDeficitTests(unittest.TestCase):
    """滞空缺口项的数值行为（只加载那一个函数，不需要 Isaac Lab）。"""

    def _load(self):
        import torch

        path = RewardContractTests.REWARDS
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        func = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "feet_air_time_deficit")
        # 源码文件顶部有 `from __future__ import annotations`，单独 exec 一个函数时必须显式带上，
        # 否则形参上的 Isaac Lab 类型注解会在 def 时求值并 NameError。
        module = ast.parse("from __future__ import annotations\n")
        module.body.append(func)
        scope = {"torch": torch}
        exec(compile(module, str(path), "exec"), scope)
        return scope["feet_air_time_deficit"], torch

    @staticmethod
    def _env(torch, air_time, first_contact, command=0.5, gravity_z=-1.0):
        class _Sensor:
            def __init__(self):
                self.data = SimpleNamespace(last_air_time=torch.tensor([air_time]))
            def compute_first_contact(self, dt):
                return torch.tensor([first_contact])
        class _Scene:
            def __init__(self):
                self.sensors = {"contact_forces": _Sensor()}
            def __getitem__(self, key):
                assert key == "robot"
                return SimpleNamespace(data=SimpleNamespace(
                    projected_gravity_b=torch.tensor([[0.0, 0.0, gravity_z]])))
        return SimpleNamespace(
            step_dt=0.02,
            command_manager=SimpleNamespace(
                get_command=lambda name: torch.tensor([[command, 0.0, 0.0]])),
            scene=_Scene())

    def test_deficit_zero_at_or_above_threshold_and_never_a_reward(self):
        fn, torch = self._load()
        sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3])
        threshold = 0.20

        # 全部达到阈值 ⇒ 精确为 0；而原始 feet_air_time 形式在同一输入上是**正奖励**（滞空越长奖越多）
        air = [0.60, 0.60, 0.60, 0.60]
        env = self._env(torch, air, [1.0] * 4)
        self.assertEqual(float(fn(env, "base_velocity", sensor_cfg, threshold)), 0.0)
        self.assertGreater(sum(a - threshold for a in air), 0.0)

        # 低于阈值才给缺口：relu(θ − air) 逐足求和
        air = [0.05, 0.20, 0.35, 0.10]
        env = self._env(torch, air, [1.0] * 4)
        expected = sum(max(threshold - a, 0.0) for a in air)
        self.assertAlmostEqual(float(fn(env, "base_velocity", sensor_cfg, threshold)), expected, places=6)
        self.assertAlmostEqual(expected, 0.25, places=6)

        # 只看「刚落地」那一拍：没落地的足不贡献
        env = self._env(torch, air, [1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(fn(env, "base_velocity", sensor_cfg, threshold)), 0.15, places=6)

    def test_deficit_respects_command_and_upright_gates(self):
        fn, torch = self._load()
        sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3])
        air = [0.05, 0.20, 0.35, 0.10]
        threshold = 0.20
        raw = 0.25

        # 站定（指令范数 ≤ 0.1）不给信号
        env = self._env(torch, air, [1.0] * 4, command=0.0)
        self.assertEqual(float(fn(env, "base_velocity", sensor_cfg, threshold)), 0.0)

        # 机身直立度门：-g_z = 0.35 ⇒ clamp(0.35, 0, 0.7) / 0.7 = 0.5
        env = self._env(torch, air, [1.0] * 4, gravity_z=-0.35)
        self.assertAlmostEqual(float(fn(env, "base_velocity", sensor_cfg, threshold)), raw * 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
