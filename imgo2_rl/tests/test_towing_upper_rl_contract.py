"""Offline contract tests for the first upper-RL environment milestone."""

import ast
import importlib.util
from pathlib import Path
import re
import statistics
import sys
import unittest

try:
    import torch
except ImportError:  # The repository's offline contract checks also run outside the training env.
    torch = None


RL = Path(__file__).resolve().parents[1]
PKG = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


logic = load("towing_upper_logic_test", PKG / "upper_logic.py")


class UpperLogicTests(unittest.TestCase):
    def test_actor_contract_matches_paper_plan(self):
        spec = logic.UpperObservationSpec()
        self.assertEqual([name for name, _ in spec.terms],
                         ["cmd_vel", "reference_command", "last_action", "base_ang_vel", "projected_gravity",
                          "last_loco_action", "joint_pos", "joint_vel"])
        self.assertEqual(spec.frame_dim, 51)
        self.assertEqual(spec.decoder_dim, 5)
        self.assertEqual(spec.actor_dim, 56)
        self.assertEqual(dict(spec.terms)["cmd_vel"], 3)
        self.assertEqual(dict(spec.terms)["last_action"], 3)

    def test_decoder_targets_are_physical_units(self):
        """2026-09-23 改为物理量：target 不再归一化，head 也不再带 tanh。

        原先归一化到 [-1,1] 配合 tanh，但归一化尺度会按 s² 压低 loss 的物理权重
        （力 s=10 ⇒ 0.01、质量 s=5 ⇒ 0.04），使这两项几乎训不动；且 `v/(1.0,0.5)` 的
        clamp 会截断超速真值。现在直接回归 m/s、kg、N。
        """
        decoder = logic.DecoderSpec()
        self.assertEqual(decoder.dim, 5)
        self.assertEqual(
            decoder.terms,
            (("robot_velocity_xy", 2), ("cart_mass", 1), ("towing_force_xy", 2)))
        # normalize_decoder_targets 已废弃，现为恒等（仅保留维数检查）
        for values in ((0, 0, 5, 0, 0), (0.5, -0.25, 10, 10, -10), (2, -2, 15, 30, -30)):
            self.assertEqual(logic.normalize_decoder_targets(values), tuple(float(v) for v in values))
        self.assertAlmostEqual(logic.denormalize_force(0.5), 0.5)
        self.assertAlmostEqual(logic.denormalize_force(-3.5), -3.5)
        with self.assertRaises(ValueError):
            logic.normalize_decoder_targets((0, 0, 5, 0))  # 维数不符仍要报错

    def test_asymmetric_action_mapping_and_speed_limits(self):
        spec = logic.UpperActionSpec()
        self.assertEqual(logic.normalized_acceleration((-1, -1, -1), spec), (-1.0, -0.5, -1.0))
        self.assertEqual(logic.normalized_acceleration((1, 1, 1), spec), (0.5, 0.5, 1.0))
        self.assertEqual(logic.normalized_acceleration((0, 0, 0), spec), (0.0, 0.0, 0.0))
        self.assertEqual(logic.integrate_reference_speed((0, -0.3, -1), (-1, -1, -1), spec),
                         (0.0, -0.3, -1.0))
        self.assertEqual(logic.integrate_reference_speed((1, 0.3, 1), (1, 1, 1), spec),
                         (1.0, 0.3, 1.0))

    def test_command_schedule_contains_settle_tow_and_explicit_zero(self):
        self.assertEqual(logic.scheduled_command(0.5, 0.7, tow_start_s=1.0, stop_time_s=5.0), 0.0)
        self.assertEqual(logic.scheduled_command(2.0, 0.7, tow_start_s=1.0, stop_time_s=5.0), 0.7)
        self.assertEqual(logic.scheduled_command(5.0, 0.7, tow_start_s=1.0, stop_time_s=5.0), 0.0)

    def _registered_entry_points(self):
        """Parse the towing registry block and resolve the two config entry points."""
        source = (PKG / "__init__.py").read_text("utf-8")
        env_match = re.search(
            r'"env_cfg_entry_point"\s*:\s*f"\{__name__\}\.'
            r'([A-Za-z_][\w.]*):([A-Za-z_]\w*)"',
            source,
        )
        agent_match = re.search(
            r'"([A-Za-z_][\w]*)_cfg_entry_point"\s*:\s*f"\{agents\.__name__\}\.'
            r'([A-Za-z_][\w.]*):([A-Za-z_]\w*)"',
            source,
        )
        self.assertIsNotNone(env_match, "注册块缺少 env_cfg_entry_point")
        self.assertIsNotNone(agent_match, "注册块缺少 agent cfg entry point")
        env_module, env_class = env_match.group(1), env_match.group(2)
        agent_key, agent_module, agent_class = agent_match.groups()

        def resolves(relative_module, class_name):
            path = PKG / (relative_module.replace(".", "/") + ".py")
            if not path.exists():
                return False
            names = {node.name for node in ast.walk(ast.parse(path.read_text("utf-8")))
                     if isinstance(node, ast.ClassDef)}
            return class_name in names

        return env_module, env_class, agent_key, agent_module, agent_class, resolves

    def _train_agent_default(self):
        train = (RL / "scripts/rl_lab/towing/train.py").read_text("utf-8")
        default_agent = re.search(r'"--agent",\s*type=str,\s*default="([^"]+)"', train)
        self.assertIsNotNone(default_agent, "train.py 未声明 --agent 默认值")
        return default_agent.group(1)

    def test_registry_entry_points_resolve_and_match_the_train_agent_name(self):
        env_module, env_class, agent_key, agent_module, agent_class, resolves = \
            self._registered_entry_points()
        self.assertEqual((env_module, env_class), ("upper_env_cfg", "UpperTowingEnvCfg"))
        self.assertTrue(resolves(env_module, env_class))
        self.assertTrue(resolves(f"agents.{agent_module}", agent_class))
        # The task is driven by the repository-owned rl_lab stack, not an external RSL-RL runner.
        self.assertEqual(agent_key, "rl_lab")
        # load_cfg_from_registry 把 --agent 的值当**完整注册键**查表，不做后缀推导
        # （官方入口默认 "rsl_rl_cfg_entry_point"），所以默认值必须是完整键名。
        self.assertEqual(self._train_agent_default(), f"{agent_key}_cfg_entry_point")

    def test_train_agent_default_is_a_full_registry_key(self):
        """回归守卫：--agent 必须是完整注册键，不能是去掉后缀的 ``rl_lab``。

        2026-09-22 冒烟实遇
        ``ValueError: Could not find configuration ... entry point: 'rl_lab'``：
        默认值曾被写成去后缀的名字，而 ``load_cfg_from_registry`` 是拿该值**原样**在注册
        kwargs 里查，只有完整键名能解析。
        """
        default_agent = self._train_agent_default()
        self.assertTrue(
            default_agent.endswith("_cfg_entry_point"),
            f"--agent 必须是完整注册键（形如 'rl_lab_cfg_entry_point'），当前为 {default_agent!r}")
        self.assertNotEqual(default_agent, "rl_lab")
        _, _, agent_key, _, _, _ = self._registered_entry_points()
        self.assertEqual(default_agent, f"{agent_key}_cfg_entry_point")

    def test_towing_runner_log_gets_start_iter(self):
        """回归守卫：``_log`` 必须把 ``start_iter`` 作为形参，且调用处要传。

        2026-09-22 实跑崩在 ``_log`` 里的 ``NameError: name 'start_iter' is not defined``——
        ``start_iter`` 是 ``learn()`` 的局部变量，``_log()`` 取不到；ETA 计算依赖它。
        该错误只在训练跑到 ``_log`` 时才暴露，静态断言成本更低。
        """
        path = RL / "scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py"
        tree = ast.parse(path.read_text("utf-8"))
        log_fn = next((n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == "_log"), None)
        self.assertIsNotNone(log_fn, "未找到 _log")
        params = [a.arg for a in log_fn.args.args]
        self.assertIn("start_iter", params, "_log 形参缺少 start_iter")
        self.assertIn("iteration", params)
        self.assertIn("total_iter", params)
        call = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "_log"), None)
        self.assertIsNotNone(call, "未找到 _log 的调用处")
        self.assertEqual(len(call.args), len(params) - 1,
                         f"_log 调用实参 {len(call.args)} 个，形参 {len(params) - 1} 个")

    def test_episode_infos_are_aggregated_before_logging(self):
        """回归守卫：同名的 episode 指标只能输出一行。

        ``episode_infos`` 是 rollout 内**每一步**有环境 reset 就追加一份 ``infos["log"]``，
        48 步会累积几十份同键字典。2026-09-22 用户实测看到「每个环境都输出了一条」：
        旧的日志拼接对每份字典逐条打印，90 行同名指标刷屏。修复是按键聚合后只打印一份。
        """
        path = RL / "scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py"
        tree = ast.parse(path.read_text("utf-8"))
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn("_aggregate_episode_infos", names,
                      "缺少 episode 统计聚合函数；直接逐份遍历会刷屏")
        # 聚合函数必须是 staticmethod 且接受 episode_infos
        agg = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_aggregate_episode_infos")
        self.assertEqual([a.arg for a in agg.args.args], ["episode_infos"])
        self.assertTrue(any(isinstance(d, ast.Name) and d.id == "staticmethod"
                            for d in agg.decorator_list),
                        "_aggregate_episode_infos 应为 staticmethod")

    @unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")
    def test_aggregation_writes_each_episode_scalar_once(self):
        """端到端：用真实 SummaryWriter 确认同一 (tag, step) 只写一次。

        旧代码对 `episode_infos` 逐份遍历写 TensorBoard，而该列表在 48 步 rollout 里会累积
        ~48 份同键字典 ⇒ 每个指标在每个 step 被重复写 48 次（2026-09-22 实测 tfevents：
        `tracking_velocity` 单轮多出 940 条重复）。修复后应恰好 1 次。
        """
        import tempfile
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        path = RL / "scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py"
        source = path.read_text("utf-8")
        # 只取聚合函数与写入函数，避免触发 rl_lab / isaaclab 的导入链
        tree = ast.parse(source)
        wanted = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        cls = next(c for c in wanted if c.name == "TowingOnPolicyRunner")
        keep = [n for n in cls.body
                if isinstance(n, ast.FunctionDef)
                and n.name in ("_aggregate_episode_infos", "_write_scalars")]
        mini = ast.Module(body=[ast.ClassDef(
            name="MiniRunner", bases=[], keywords=[], decorator_list=[], type_params=[],
            body=keep)], type_ignores=[])
        ns = {"torch": torch, "statistics": statistics}
        exec(compile(ast.fix_missing_locations(mini), "<mini>", "exec"), ns)
        runner = ns["MiniRunner"]()

        with tempfile.TemporaryDirectory() as tmp:
            from torch.utils.tensorboard import SummaryWriter
            runner.writer = SummaryWriter(log_dir=tmp, flush_secs=1)
            keys = ["Episode_Reward/tracking_velocity", "Episode_Termination/time_out"]
            infos = [{k: torch.tensor(0.5) for k in keys} for _ in range(48)]
            stats = runner._aggregate_episode_infos(infos)
            self.assertEqual(sorted(stats), sorted(keys), "聚合后键集合应不变")
            runner._write_scalars(0, 1.0, 0.1, 1.0, 0.1, torch.tensor(0.1), 0.5, 100.0,
                                  [-1.0], [10.0], stats)
            runner.writer.flush()
            ea = EventAccumulator(tmp, size_guidance={"scalars": 0}); ea.Reload()
            for key in keys:
                events = ea.Scalars(f"Episode/{key}")
                self.assertEqual(len(events), 1,
                                 f"Episode/{key} 应只写 1 次，实际 {len(events)} 次")

    def test_collision_filters_resolve_one_prim_per_environment(self):
        """回归守卫：碰撞传感器的 filter 不得用 `Robot/.*` 通配。

        Isaac Lab 的 `ContactSensorCfg` 要求每个 filter 项在每个环境里只解析出一个 prim
        （`contact_sensor_cfg.py` attention 段）。2026-09-22 实测 `{ENV_REGEX_NS}/Robot/.*`
        每环境展开 19 个 prim × 256 环境 = 4864，抛
        `Filter pattern ... (expected 256, found 4864)`，使 `force_matrix_w` 失效 ⇒
        `cart_collision` 恒假 ⇒ 碰撞 reward 与终止全部失效。
        """
        src = (PKG / "upper_env_cfg.py").read_text("utf-8")
        # 只看可执行代码：注释与 docstring 里会引用这个错误写法作为反例。
        # 用 AST 剥掉所有常量字符串（含 docstring），再序列化回代码检查。
        tree_all = ast.parse(src)
        for node in ast.walk(tree_all):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = ""
        code = ast.unparse(tree_all)
        self.assertNotIn("Robot/.*", code, "可执行代码里不能对 Robot 用 .* 通配")
        self.assertNotIn("Robot/*", code, "可执行代码里不能对 Robot 用 * 通配")
        # filter 必须从 URDF link 名逐个构造（用 AST 确认函数与调用存在）
        tree = ast.parse(src)
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn("_robot_body_filters", funcs)
        self.assertIn("_robot_link_names_from_urdf", funcs)
        self.assertIn("_ROBOT_BODY_FILTERS = _robot_body_filters()", src)

        # URDF 与传感器数量：5 个车体传感器各自使用 shared filter 列表
        self.assertEqual(src.count("filter_prim_paths_expr=_ROBOT_BODY_FILTERS"), 5,
                         "5 个车体碰撞传感器都应使用显式 filter 列表")

        # 仓库根下标必须是 7（2026-09-22 修过 parents[6] 的写错）
        self.assertIn("_REPO_ROOT = Path(__file__).resolve().parents[7]", src,
                      "upper_env_cfg 的仓库根应是 parents[7]；写错会让 USD 缓存落到 imgo2_rl/")

    def test_collision_filter_matches_urdf_link_count(self):
        """filter 项数必须等于 URDF 的 link 数（每 link 一项，避免通配展开）。"""
        import re as _re
        urdf = RL.parent / "imgo2_description" / "urdf" / "imgo2.urdf"
        links = _re.findall(r'<link\s+name="([^"]+)"', urdf.read_text(encoding="utf-8"))
        self.assertEqual(len(links), 17, f"URDF link 数应为 17，实际 {len(links)}")
        src = (PKG / "upper_env_cfg.py").read_text("utf-8")
        # 构造式子必须是逐 link 一项
        self.assertIn('f"{{ENV_REGEX_NS}}/Robot/{name}" for name in _robot_link_names_from_urdf()', src)

    def test_per_step_physics_does_not_readback_physx(self):
        """性能守卫：每个物理步的拖曳物理不得回读 PhysX 的质量／惯量。

        `apply_actions` 每个 5 ms 物理步调用一次，4096 环境下每轮 `480 × 4096` 次。原来的
        `_apply_towing_physics` 每步都 `root_physx_view.get_masses()/get_inertias()`，即每步一次
        GPU→CPU→GPU 往返，实测把每轮固定开销推到约 4.6 s（与算力无关）。现在这些量只在
        reset 后重建一次缓存。
        """
        src = (PKG / "upper_mdp.py").read_text("utf-8")
        tree = ast.parse(src)
        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn("_refresh_mass_cache", funcs)
        self.assertIn("_invalidate_mass_cache", funcs)

        def calls(func_name, wanted):
            found = []
            for n in ast.walk(funcs[func_name]):
                if isinstance(n, ast.Call) and getattr(n.func, "attr", "") in wanted:
                    found.append((n.lineno, n.func.attr))
            return found

        hot = calls("_apply_towing_physics", {"get_masses", "get_inertias"})
        self.assertEqual(hot, [], f"每步热路径里仍有 PhysX 回读: {hot}")
        # 缓存必须由 _refresh_mass_cache 统一重建
        self.assertTrue(calls("_refresh_mass_cache", {"get_masses", "get_inertias"}),
                        "_refresh_mass_cache 应负责回读并缓存")

        # reset 改写质量后必须失效缓存，且顺序在 set_masses 之后
        reset = funcs["reset_towing_episode"]
        set_line = min(n.lineno for n in ast.walk(reset)
                       if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "set_masses")
        inv = [n.lineno for n in ast.walk(reset)
               if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "_invalidate_mass_cache"]
        self.assertTrue(inv, "reset_towing_episode 改质量后必须调用 _invalidate_mass_cache")
        self.assertGreater(min(inv), set_line, "失效缓存必须发生在 set_masses 之后")

    def test_policy_observation_excludes_privileged_load_signals(self):
        tree = ast.parse((PKG / "upper_env_cfg.py").read_text("utf-8"))
        policy = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg")
        names = {target.id for node in policy.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)}
        self.assertFalse(names & {"cart_mass", "cart_velocity", "rope_state", "ground_friction"})

    @unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")
    def test_decoder_output_is_detached_before_actor(self):
        decoder_module = load(
            "towing_dynamics_decoder_test",
            RL / "scripts/rl_lab/rl_lab/modules/towing_decoder.py")
        decoder = decoder_module.TowingDynamicsDecoder(
            frame_dim=51, feature_dim=8, hidden_dim=8)
        frames = torch.zeros(3, 51, requires_grad=True)
        prediction, hidden = decoder(frames)
        self.assertEqual(tuple(prediction.shape), (3, 5))
        self.assertEqual(tuple(hidden.shape), (1, 3, 8))
        actor_obs = decoder_module.augment_actor_observation(frames, prediction)
        self.assertEqual(tuple(actor_obs.shape), (3, 56))
        actor_obs.sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))

    @unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")
    def test_decoder_force_weighted_mass_supervision_and_update(self):
        decoder_module = load(
            "towing_dynamics_decoder_train_test",
            RL / "scripts/rl_lab/rl_lab/modules/towing_decoder.py")
        force = torch.tensor([[0.0, 0.0], [1.0, 0.0], [11.0, 0.0]])
        weight = decoder_module.mass_supervision_weight(force)
        self.assertTrue(torch.equal(weight, torch.tensor([0.0, 0.0, 0.5])))

        decoder = decoder_module.TowingDynamicsDecoder(
            frame_dim=51, feature_dim=8, hidden_dim=8)
        trainer = decoder_module.DynamicsDecoderTrainer(decoder)
        frames = torch.randn(4, 3, 51, requires_grad=True)
        targets = torch.zeros(4, 3, 5)
        weights = torch.zeros(4, 3)
        weights[:, 1] = 1.0
        loss = trainer.update(frames, targets, weights)
        self.assertEqual(loss.ndim, 0)
        self.assertIsNone(frames.grad)
        self.assertFalse(decoder.training)

    @unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")
    def test_decoder_loss_reports_three_weighted_components(self):
        """去掉 target 归一化与 head tanh 后，三项尺度不同（m/s、kg、N），
        1:1:1 的权重并不等权（实测质量项可占 85%、速度项仅 0.1%）。因此 loss()
        必须把三项加权后的贡献单独返回，供日志判断权重是否真的配平。
        """
        decoder_module = load(
            "towing_decoder_parts_test",
            RL / "scripts/rl_lab/rl_lab/modules/towing_decoder.py")
        decoder = decoder_module.TowingDynamicsDecoder(
            frame_dim=51, feature_dim=8, hidden_dim=8)
        prediction = torch.zeros(2, 5)
        targets = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0], [1.0, 0.0, 12.0, 2.0, 0.0]])
        weight = torch.ones(2)
        total, parts = decoder.loss(prediction, targets, weight)
        self.assertEqual(len(parts), 3)
        self.assertAlmostEqual(float(total), float(sum(parts)), places=5)
        # 质量项的绝对贡献应远大于速度项（尺度差异的直接体现）
        self.assertGreater(float(parts[2]), float(parts[0]))

    def test_min_clearance_reward_is_ratio_based_and_gated(self):
        """最小间距奖励：阈值按绳长比例给出，且对无小车环境屏蔽。

        2026-09-23 用户要求「维持小车与机器人距离不低于绳长的 0.6 倍」。注意实测几何：
        初始「后表面→车斗」间隙约 0.349 m < 0.6×0.8=0.48 m，故该项一开局即激活
        （docstring 已记录该事实与两种可选处理）。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("def min_clearance_violation(", mdp)
        # 阈值必须是 ratio × rope_length，而不是写死的绝对量
        self.assertIn("threshold = ratio * rope_length", mdp)
        # 无小车环境必须屏蔽（与 clearance/collision 同一约定）
        self.assertIn("return violation * term.cart_present[:, 0]", mdp)
        # 注册项存在且 ratio=0.6、权重为负
        self.assertIn("min_clearance = RewTerm(func=mdp.min_clearance_violation, weight=-2.0", cfg)
        self.assertIn('"ratio": 0.6', cfg)
        self.assertIn('"rope_length": 0.8', cfg)

    def test_min_clearance_matches_rope_length_config(self):
        """阈值参数必须与 action term 的 rope_length 一致，避免两处漂移。"""
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        import re as _re
        rope_in_term = _re.search(r"rope_length: float = ([0-9.]+)", mdp).group(1)
        rope_in_reward = _re.search(r'"rope_length": ([0-9.]+)', cfg).group(1)
        self.assertEqual(rope_in_term, rope_in_reward,
                         f"rope_length 不一致：term={rope_in_term} reward={rope_in_reward}")

    def test_reset_event_contract_has_all_v0_work_condition_axes(self):
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        for fragment in (
            '"speed_range": (0.2, 1.0)',
            '"stop_time_range": (4.0, 6.0)',
            '"mass_range": (5.0, 15.0)',
            '"friction_range": (0.4, 1.2)',
            '"wheel_damping_range": (0.008, 0.032)',
            '"no_cart_fraction": 0.125',
            '"no_cart_lateral_offset": 2.0',
        ):
            self.assertIn(fragment, cfg)
        self.assertIn("default_masses[ids_cpu] * scale[:, None]", mdp)
        self.assertIn("default_inertias[ids_cpu] * scale[:, None, None]", mdp)
        self.assertIn("torch.randint(0, 2, (count,)", mdp)
        self.assertIn("elapsed_s < self.stop_time_s", mdp)
        self.assertIn("self._apply_towing_physics()", mdp)
        self.assertIn("self._physics_step % low_level_decimation", mdp)
        self.assertIn("-self.wheel_damping * self._cart.data.joint_vel", mdp)
        self.assertIn("SplitRopeModel(", mdp)
        # target 改为物理量后不再有归一化；断言 decoder_targets 直接返回原始量
        self.assertIn("return torch.cat((robot_velocity_xy, mass, force), dim=1)", mdp)
        self.assertNotIn("force / (force.abs() + 10.0)", mdp)
        self.assertIn("(force - minimum_force).clamp_min(0.0)", mdp)
        self.assertIn('params={"minimum_force": 1.0, "force_scale": 10.0}', cfg)
        # 质量 target 也改为物理量（不再 (m-5)/5-1）
        self.assertIn("    mass = term.cart_mass", mdp)
        self.assertNotIn("(term.cart_mass - 5.0) / 5.0 - 1.0", mdp)

    def test_no_cart_environments_are_physically_and_semantically_masked(self):
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("self.cart_present = torch.ones", mdp)
        self.assertIn("force_on_robot), dim=-1) * present", mdp)
        self.assertIn("force_on_cart), dim=-1) * present", mdp)
        self.assertIn("cart_state[~cart_present, 1] += no_cart_lateral_offset", mdp)
        self.assertIn("* term.cart_present[:, 0]", mdp)
        self.assertIn("term.cart_present.float()", mdp)

    def test_safety_and_history_producers_are_live(self):
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn('prim_path="{ENV_REGEX_NS}/Cart/base_link"', cfg)
        for body in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
            self.assertIn(f'prim_path="{{ENV_REGEX_NS}}/Cart/{body}"', cfg)
        # 2026-09-22 改：filter 必须逐个列出机器人 body，不能用 `Robot/.*` 通配
        # （每 filter 项须每环境只解析出 1 个 prim，否则 force_matrix_w 失效）。
        self.assertEqual(cfg.count("filter_prim_paths_expr=_ROBOT_BODY_FILTERS"), 5,
                         "5 个车体传感器都应使用显式 filter 列表")
        self.assertIn("sensor.data.force_matrix_w", mdp)
        self.assertIn("maximum_force > self.cfg.collision_force_threshold", mdp)
        self.assertIn("self.rope_state[:, 0] =", mdp)
        self.assertIn("self.towing_force_b[:] = force_robot_b[:, 0, :2]", mdp)
        self.assertIn("frame = ObsTerm(func=mdp.policy_frame)", cfg)
        self.assertNotIn("cmd_vel = ObsTerm", cfg)
        ppo_cfg = (PKG / "agents/upper_ppo_cfg.py").read_text("utf-8")
        self.assertIn("TowingActorCriticCfg", ppo_cfg)
        self.assertNotIn("RslRlRNNModelCfg", ppo_cfg)
        self.assertNotIn("isaaclab_rl.rsl_rl", ppo_cfg)

    def test_upper_ppo_normalizes_only_privileged_critic_input(self):
        cfg = (PKG / "agents/upper_ppo_cfg.py").read_text("utf-8")
        self.assertIn("critic_empirical_normalization = True", cfg)
        self.assertEqual(cfg.count('rnn_type="gru"'), 1)
        self.assertIn("policy = TowingActorCriticCfg", cfg)

    def test_ppo_configs_use_isaac_lab_2_2_rsl_rl_interface(self):
        upper_cfg = (PKG / "agents/upper_ppo_cfg.py").read_text("utf-8")
        base_cfg = (PKG.parent / "locomotion/velocity/base_move/agents/rsl_rl_ppo_cfg.py").read_text("utf-8")
        for cfg in (base_cfg,):
            self.assertNotIn("RslRlMLPModelCfg", cfg)
            self.assertNotIn("actor_obs_normalization", cfg)
            self.assertNotIn("critic_obs_normalization", cfg)
        self.assertNotIn("isaaclab_rl.rsl_rl", upper_cfg)

    def test_towing_runner_owns_decoder_normalizer_and_checkpoint(self):
        runner = (RL / "scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py").read_text("utf-8")
        wrapper = (RL / "scripts/rl_lab/rl_lab/wrapper/towing_vec_env_wrapper.py").read_text("utf-8")
        self.assertIn("class TowingOnPolicyRunner", runner)
        self.assertIn("augment_actor_observation(raw_obs, estimate)", runner)
        self.assertIn("critic_normalizer(critic_obs, update=update)", runner)
        self.assertIn('"decoder_optimizer_state_dict"', runner)
        self.assertIn('"critic_normalizer_state_dict"', runner)
        self.assertIn("dones=torch.stack(done_rollout)", runner)
        self.assertIn("class TowingVecEnvWrapper", wrapper)
        self.assertIn('{"policy", "critic", "decoder"}', wrapper)
        train = (RL / "scripts/rl_lab/towing/train.py").read_text("utf-8")
        self.assertIn("TowingVecEnvWrapper", train)
        self.assertIn("TowingOnPolicyRunner", train)
        self.assertNotIn("rsl_rl.runners", train)

    def test_rl_lab_recurrent_reset_uses_boolean_done_mask(self):
        recurrent = (RL / "scripts/rl_lab/rl_lab/modules/actor_critic_recurrent.py").read_text("utf-8")
        self.assertIn("done_mask = dones.bool()", recurrent)
        self.assertIn("hidden_state[..., done_mask, :] = 0.0", recurrent)
        self.assertIn("if self.hidden_states is None or dones is None", recurrent)

    def test_hierarchical_frequency_contract_is_200_50_20_hz(self):
        env_cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        policy_cfg = (PKG / "utils/policy_cfg.py").read_text("utf-8")
        self.assertIn("self.sim.dt = 0.005", env_cfg)
        self.assertIn("self.decimation = 10", env_cfg)
        self.assertIn("low_level_control_dt=0.02", env_cfg)
        self.assertIn("control_dt=0.02", policy_cfg)
        self.assertIn("cfg.physics_dt - env.physics_dt", mdp)
        self.assertIn("cfg.upper_control_dt - env.step_dt", mdp)
        self.assertIn("cfg.low_level_control_dt - self._policy_cfg.control_dt", mdp)

    def test_post_stop_distance_excludes_initial_settle_phase(self):
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("elapsed_s >= term.stop_time_s", mdp)

    def test_post_stop_towing_force_is_gated_and_bounded(self):
        env_cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("stop_towing_force = RewTerm(func=mdp.post_stop_towing_force", env_cfg)
        self.assertIn('params={"force_scale": 10.0}', env_cfg)
        self.assertIn("post_stop = (elapsed_s >= term.stop_time_s).float()", mdp)
        self.assertIn("normalized_force = force / (force + force_scale)", mdp)
        self.assertIn("normalized_force * post_stop * term.cart_present[:, 0]", mdp)


if __name__ == "__main__":
    unittest.main()
