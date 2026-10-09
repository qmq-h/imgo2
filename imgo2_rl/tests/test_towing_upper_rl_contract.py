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
# 场景网格是纯算术模块（无 torch / Isaac Lab 依赖），可离线加载
grid = load("towing_connection_grid_test", PKG / "mdp/connection_grid.py")


class UpperLogicTests(unittest.TestCase):
    def test_actor_contract_matches_paper_plan(self):
        """57 维单帧 + 6 维 estimate = 79 维 actor 输入（2026-10-08 残差方案）。

        命令项只保留 `loco_command`（送冻结策略的脚本指令）；`last_action` 由 3 维
        归一化加速度变为 12 维关节残差；decoder 力改 3 维后 estimate 为 6 维。
        """
        spec = logic.UpperObservationSpec()
        self.assertEqual([name for name, _ in spec.terms],
                         ["loco_command", "last_action", "base_ang_vel", "projected_gravity",
                          "last_loco_action", "joint_pos", "joint_vel"])
        self.assertEqual(spec.frame_dim, 57)
        self.assertEqual(spec.decoder_dim, 6)
        self.assertEqual(spec.actor_dim, 79)
        self.assertEqual(dict(spec.terms)["loco_command"], 3)
        self.assertEqual(dict(spec.terms)["last_action"], 12)

    def test_decoder_targets_are_physical_units(self):
        """2026-09-23 改为物理量：target 不再归一化，head 也不再带 tanh。

        原先归一化到 [-1,1] 配合 tanh，但归一化尺度会按 s² 压低 loss 的物理权重
        （力 s=10 ⇒ 0.01、质量 s=5 ⇒ 0.04），使这两项几乎训不动；且 `v/(1.0,0.5)` 的
        clamp 会截断超速真值。现在直接回归 m/s、kg、N。
        2026-10-08：速度保持 2 维、牵引力改 3 维（机体系 x/y/z），故 dim 5 → 6。
        """
        decoder = logic.DecoderSpec()
        self.assertEqual(decoder.dim, 6)
        self.assertEqual(
            decoder.terms,
            (("robot_velocity_xy", 2), ("cart_mass", 1), ("towing_force_xyz", 3)))
        # normalize_decoder_targets 已废弃，现为恒等（仅保留维数检查）
        for values in ((0, 0, 5, 0, 0, 0), (0.5, -0.25, 10, 10, -10, 3), (2, -2, 15, 30, -30, 0)):
            self.assertEqual(logic.normalize_decoder_targets(values), tuple(float(v) for v in values))
        self.assertAlmostEqual(logic.denormalize_force(0.5), 0.5)
        self.assertAlmostEqual(logic.denormalize_force(-3.5), -3.5)
        with self.assertRaises(ValueError):
            logic.normalize_decoder_targets((0, 0, 5, 0, 0))  # 维数不符仍要报错

    def test_upper_action_is_a_scaled_clipped_joint_residual(self):
        """动作 = 12 维归一化关节残差，映射为 `residual_scale ⊙ clip(u,±1)`。

        残差尺度取冻结策略契约的 `action_scale`，因此幅值受底层动作范围界定；
        归一化动作必须先裁到 ±1，否则残差会超出设计幅值（网络输出不受物理约束）。
        旧的 3 维加速度映射／参考速度积分已删除，不能留下任何可用入口。
        """
        scale = (0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25)
        spec = logic.UpperActionSpec(residual_scale=scale, control_dt=0.05)
        spec.validate()
        self.assertEqual(spec.delta_joint_pos([0.0] * 12), (0.0,) * 12)
        self.assertEqual(spec.delta_joint_pos([1.0] * 12), (0.25,) * 12)
        self.assertEqual(spec.delta_joint_pos([-1.0] * 12), (-0.25,) * 12)
        # 超出 ±1 必须被裁掉（而不是线性外推）
        self.assertEqual(spec.delta_joint_pos([5.0] * 12), (0.25,) * 12)
        self.assertEqual(spec.delta_joint_pos([-5.0] * 12), (-0.25,) * 12)
        # 逐关节尺度不被统一化：不同关节可以有不同幅值
        mixed = logic.UpperActionSpec(residual_scale=(0.1,) * 6 + (0.5,) * 6)
        self.assertEqual(mixed.delta_joint_pos([1.0] * 12), (0.1,) * 6 + (0.5,) * 6)
        with self.assertRaises(ValueError):
            spec.delta_joint_pos([1.0] * 11)          # 维数必须等于 residual_scale
        with self.assertRaises(ValueError):
            logic.UpperActionSpec(residual_scale=()).validate()
        with self.assertRaises(ValueError):
            logic.UpperActionSpec(residual_scale=(0.25,) * 11 + (0.0,)).validate()
        # 已删除的旧入口不能复活
        self.assertFalse(hasattr(logic, "normalized_acceleration"))
        self.assertFalse(hasattr(logic, "integrate_reference_speed"))

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
        prediction, latent, hidden = decoder.forward_with_latent(frames)
        self.assertEqual(tuple(prediction.shape), (3, 6))
        self.assertEqual(decoder.output_dim, 6)
        self.assertEqual(tuple(hidden.shape), (1, 3, 8))
        actor_obs = decoder_module.augment_actor_observation(frames, prediction, latent)
        self.assertEqual(tuple(actor_obs.shape), (3, 73))
        actor_obs.sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))

    @unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")
    def test_towing_network_forward_contract_and_zero_init(self):
        """端到端数值契约：57 帧 → 6 维估计(GRU 128) → 79 维 actor(GRU 256) → 12 维残差；critic 72 → 1。

        这条用**真实网络类**跑一次前向，锁住三件事：
        1. 各层宽度与 `upper_logic` 契约一致（frame 57 / decoder 6 / actor 79 / action 12）；
        2. `augment_actor_observation` 的拼接结果能直接喂进 `ActorCriticRecurrent`（63 → GRU(63,256)）；
        3. towing runner 的**零初始化**语义：零初始化后首拍残差必须精确为 0（实测未初始化时
           为 |a|max ≈ 0.149 归一化，即约 ±0.04 rad 的系统性关节偏置），保证起点等于冻结策略自身的步态。

        `rl_lab.modules.__init__` 会经 `utils.export_deploy_cfg` 间接 import isaaclab（离线进程
        没有 `omni.log`），所以这里按文件路径加载三个模块，并给推理路径不使用的
        `unpad_trajectories` 打桩；结束后恢复 `sys.modules`，避免影响其它测试。
        """
        import importlib.util
        import types

        modules_dir = RL / "scripts/rl_lab/rl_lab/modules"
        stub_names = ("rl_lab", "rl_lab.modules", "rl_lab.utils",
                      "rl_lab.modules.actor_critic", "rl_lab.modules.actor_critic_recurrent",
                      "rl_lab.modules.towing_decoder")
        saved = {name: sys.modules.get(name) for name in stub_names}

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        try:
            for name in ("rl_lab", "rl_lab.modules", "rl_lab.utils"):
                module = types.ModuleType(name)
                module.__path__ = []
                sys.modules[name] = module
            sys.modules["rl_lab.utils"].unpad_trajectories = lambda out, masks: out
            load("rl_lab.modules.actor_critic", modules_dir / "actor_critic.py")
            recurrent = load("rl_lab.modules.actor_critic_recurrent",
                             modules_dir / "actor_critic_recurrent.py")
            decoder_module = load("rl_lab.modules.towing_decoder",
                                  modules_dir / "towing_decoder.py")

            obs = logic.UpperObservationSpec()
            dec_spec = logic.DecoderSpec()
            actor_dim = obs.actor_dim
            # critic 特权组：57 帧 + 机器人速度 2 + 小车速度 2 + 绳状态 4 + 机体系三维拉力 3
            #              + 质量/摩擦/轮阻/有无小车 4 = 72
            critic_dim = obs.frame_dim + 2 + 2 + 4 + 3 + 4
            self.assertEqual(critic_dim, 72)
            action_dim = 12

            decoder = decoder_module.TowingDynamicsDecoder(
                frame_dim=obs.frame_dim, feature_dim=128, hidden_dim=128, num_layers=1)
            actor_critic = recurrent.ActorCriticRecurrent(
                num_actor_obs=actor_dim, num_critic_obs=critic_dim, num_actions=action_dim,
                actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64],
                activation="elu", rnn_type="gru", rnn_hidden_size=256, rnn_num_layers=1,
                init_noise_std=0.5)
            # 与 `towing_on_policy_runner` 相同的零初始化（只动 actor 末层）
            last_linear = [layer for layer in actor_critic.actor
                           if isinstance(layer, torch.nn.Linear)][-1]
            torch.nn.init.zeros_(last_linear.weight)
            torch.nn.init.zeros_(last_linear.bias)

            self.assertEqual(decoder.output_dim, dec_spec.dim)
            self.assertEqual(last_linear.out_features, action_dim)
            self.assertEqual(actor_critic.memory_a.rnn.input_size, actor_dim)
            self.assertEqual(actor_critic.memory_c.rnn.input_size, critic_dim)
            self.assertEqual(actor_critic.std.numel(), action_dim)

            decoder.eval()
            actor_critic.eval()
            frames = torch.randn(4, obs.frame_dim)
            with torch.inference_mode():
                estimate, latent, hidden = decoder.forward_with_latent(frames)
                actions = actor_critic.act_inference(
                    decoder_module.augment_actor_observation(frames, estimate, latent))
                values = actor_critic.evaluate(torch.randn(4, critic_dim))
            self.assertEqual(tuple(estimate.shape), (4, dec_spec.dim))
            self.assertEqual(tuple(hidden.shape), (1, 4, 128))
            self.assertEqual(tuple(actions.shape), (4, action_dim))
            self.assertEqual(tuple(values.shape), (4, 1))
            # 残差必须从 0 起步：零初始化后首拍动作精确为 0（探索噪声由 std 另加）
            self.assertEqual(actions.abs().max().item(), 0.0)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    @unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")
    def test_decoder_head_widths_are_two_velocity_one_mass_three_force(self):
        """力 3 维、速度 2 维必须落在 head 宽度上，且索引常量与之一致。

        这条守的是「改了 head 忘了切片」：`force_newtons`／`loss()`／play 的列都从
        `FORCE_SLICE` 推导，一旦 head 宽度与常量脱钩就会静默错位。
        """
        decoder_module = load(
            "towing_decoder_dims_test",
            RL / "scripts/rl_lab/rl_lab/modules/towing_decoder.py")
        decoder = decoder_module.TowingDynamicsDecoder(frame_dim=51, feature_dim=8, hidden_dim=8)
        self.assertEqual(decoder.velocity_head.out_features, 2)
        self.assertEqual(decoder.mass_head.out_features, 1)
        self.assertEqual(decoder.force_head.out_features, 3)
        self.assertEqual(decoder_module.OUTPUT_DIM, 6)
        self.assertEqual(decoder_module.MASS_INDEX, 2)
        self.assertEqual(decoder_module.FORCE_SLICE, slice(3, 6))
        prediction = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6)
        self.assertTrue(torch.equal(decoder.force_newtons(prediction), prediction[:, 3:6]))
        # 三维力的质量监督权重按真实张力算；二维力必须直接报错（旧布局不能悄悄通过）
        three_d = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 11.0]])
        weight = decoder_module.mass_supervision_weight(three_d)
        self.assertTrue(torch.equal(weight, torch.tensor([0.0, 0.0, 0.5])))
        with self.assertRaises(ValueError):
            decoder_module.mass_supervision_weight(torch.zeros(3, 2))

    @unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")
    def test_decoder_force_weighted_mass_supervision_and_update(self):
        decoder_module = load(
            "towing_dynamics_decoder_train_test",
            RL / "scripts/rl_lab/rl_lab/modules/towing_decoder.py")
        force = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [11.0, 0.0, 0.0]])
        weight = decoder_module.mass_supervision_weight(force)
        self.assertTrue(torch.equal(weight, torch.tensor([0.0, 0.0, 0.5])))

        decoder = decoder_module.TowingDynamicsDecoder(
            frame_dim=51, feature_dim=8, hidden_dim=8)
        trainer = decoder_module.DynamicsDecoderTrainer(decoder)
        frames = torch.randn(4, 3, 51, requires_grad=True)
        targets = torch.zeros(4, 3, 6)
        weights = torch.zeros(4, 3)
        weights[:, 1] = 1.0
        loss = trainer.update(frames, targets, weights)
        self.assertEqual(loss.ndim, 0)
        self.assertIsNone(frames.grad)
        self.assertFalse(decoder.training)

    def test_low_level_position_error_uses_the_frozen_target_not_the_residual_command(self):
        """底层跟踪误差惩罚必须拿**冻结策略自己**的关节目标，不能用「目标 + 残差」。

        用户要求：惩罚「底层输出的 pos 和真实 pos 的差距」，且明确「是底层输出的 pos 而不是
        经过残差的 pos」。混淆两者会让该项退化成罚上层动作，与 action_magnitude/action_rate
        重复。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("def low_level_position_error_l2(", mdp)
        self.assertIn("actual = term._asset.data.joint_pos[:, term._policy_to_asset]", mdp)
        self.assertIn("return (actual - term.loco_joint_targets).square().sum(dim=1)", mdp)
        body = mdp[mdp.index("def low_level_position_error_l2("):]
        body = body[:body.index("\ndef ", 1)]
        # 只看代码，跳过 docstring（docstring 自己会把 `+ delta_joint_pos` 当反例引用）
        code = body[body.index('"""', body.index('"""') + 3) + 3:]
        self.assertNotIn("delta_joint_pos", code)
        self.assertNotIn("_held_joint_targets", code)
        # 缓存来自 output.joint_targets，且叠加残差之前先记录，顺序反了就不是"底层输出"了
        self.assertIn("self.loco_joint_targets.copy_(output.joint_targets)", mdp)
        self.assertLess(mdp.index("self.loco_joint_targets.copy_(output.joint_targets)"),
                        mdp.index("output.joint_targets + self.delta_joint_pos"))
        self.assertIn(
            "low_level_pos_error = RewTerm(func=mdp.low_level_position_error_l2, weight=-5.0)",
            cfg)

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
        prediction = torch.zeros(2, 6)
        targets = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 12.0, 2.0, 0.0, 0.0]])
        weight = torch.ones(2)
        total, parts = decoder.loss(prediction, targets, weight)
        self.assertEqual(len(parts), 3)
        self.assertAlmostEqual(float(total), float(sum(parts)), places=5)
        # 质量项的绝对贡献应远大于速度项（尺度差异的直接体现）
        self.assertGreater(float(parts[2]), float(parts[0]))

    def test_min_clearance_reward_is_ratio_based_and_gated(self):
        """最小间距奖励：阈值按**逐 env 连接长度**比例给出，且对无小车环境屏蔽。

        2026-09-23 用户要求「维持小车与机器人距离不低于连接长度的 ratio 倍」。2026-10-08
        场景改成 20 行长度 0.4–0.8 m 的网格后，阈值必须逐 env 取 `term.connection_length`，
        否则短绳行（L0=0.4 时 spawn 间隙仅约 0.108 m）一开局就满额惩罚；ratio 相应由 0.40
        调到 0.25。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("def min_clearance_violation(", mdp)
        # 阈值必须是 ratio × 本 env 连接长度，而不是写死的绝对量
        self.assertIn("threshold = ratio * term.connection_length[:, 0]", mdp)
        # 无小车环境必须屏蔽（与 clearance/collision 同一约定）
        self.assertIn("return violation * term.cart_present[:, 0]", mdp)
        self.assertIn("min_clearance = RewTerm(func=mdp.min_clearance_violation, weight=-2.0", cfg)
        self.assertIn('"ratio": 0.25', cfg)

    def test_min_clearance_is_inactive_at_spawn_for_every_grid_row(self):
        """意图守卫：**所有行**的 spawn 间隙都必须高于 min_clearance 阈值。

        用户 2026-09-23 明确要求「初始的时候这个奖励不生效」。网格里 L=0.4 m 那一行
        间隙最小（绳：target=0.5L；刚体：target=L），故只要两种最短行都高于阈值即可。
        本测试从源码与网格常数自行验算，避免把数值重新硬编码一遍。
        """
        import math
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        assets = (RL / "source/imgo2_rl/imgo2_rl/assets/imgo2.py").read_text("utf-8")

        def one(pattern, text=cfg):
            return float(re.search(pattern, text).group(1))

        ratio = one(r'"ratio": ([0-9.]+)')
        # 机器人出生高度来自训练侧 `IMGO2_CFG.init_state.pos`（upper 场景直接用该配置）
        robot_z = one(r"pos=\(0\.0, 0\.0, ([0-9.]+)\)", assets)
        cart_z = one(r"cart\.init_state\.pos = \(-?[0-9.]+, [0-9.]+, ([0-9.]+)\)")
        rear = one(r"robot_rear_surface_x: float = ([0-9.]+)", mdp)
        front = one(r"cart_front_surface_x: float = ([0-9.]+)", mdp)
        robot_attach_x = one(r"robot_attachment: tuple\[float, float, float\] = \((-?[0-9.]+)", mdp)
        cart_attach_x = one(r"cart_attachment: tuple\[float, float, float\] = \((-?[0-9.]+)", mdp)
        final_z = robot_z - cart_z
        # clearance = (robot_x − rear) − (cart_x + front)，而 cart_x 由 target 反解，
        # 化简后 = h + (−rear − robot_attach_x + cart_attach_x − front)
        base_offset = -rear - robot_attach_x + cart_attach_x - front
        for label, target in (("最短绳行", grid.SLACK_RATIO * grid.LENGTH_MIN_M),
                              ("最短刚体行", grid.LENGTH_MIN_M)):
            horizontal = math.sqrt(target ** 2 - final_z ** 2)
            gap = horizontal + base_offset
            threshold = ratio * grid.LENGTH_MIN_M
            self.assertGreater(gap, threshold,
                               f"{label}（L={grid.LENGTH_MIN_M}）spawn 间隙 {gap:.4f} m "
                               f"未高于阈值 {threshold:.4f} m；用户要求 spawn 时该奖励不生效")

    def test_min_clearance_hinge_is_exactly_zero_above_threshold(self):
        """阈值上方必须**精确为 0**（不能留 softplus 偏置）。

        `softplus(0)*softness = 0.0139` 会让间隙略高于阈值时也被扣分（实测 gap=0.40 时
        −0.0007/步），与"初始不生效"冲突，故实现里减掉了该偏置。
        """
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("torch.relu(", mdp)
        self.assertIn("bias = 0.6931471805599453 * softness", mdp)
        self.assertIn("- bias)", mdp)

    def test_heading_hold_reward_is_relative_to_spawn_heading(self):
        """朝向保持：必须相对**初始 yaw**，不得硬编码世界系 0。

        2026-09-23 用户报告「机器人开始就在自转」。原奖励只惩罚 yaw **角速度**误差
        （loco_command 的 yaw 恒为 0），匀速自转在 settle 段几乎不受罚（角速度也≈0）。
        本项补上**朝向**约束。

        取相对值的原因：初始朝向来自 `default_root_state`（含 ±0.03 rad 随机化），
        用世界系 0 会把随机化的偏移当成初始误差。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("def heading_deviation(", mdp)
        self.assertIn("def yaw_heading_l2(", mdp)
        # 必须用 default_root_state 的朝向作为基准，而不是写死 0
        self.assertIn("default_root_state[:, 3:7]", mdp)
        self.assertIn("math_utils.wrap_to_pi(yaw - yaw_init)", mdp)
        # 2026-10-09 用户决定：朝向/横向由 PD 外环负责 ⇒ 该奖励项**已关闭**
        # （helper 保留以便回开；不得再注册成 RewTerm，否则与 PD 重复约束）
        active = [line for line in cfg.splitlines()
                  if line.strip().startswith("yaw_heading = RewTerm")]
        self.assertEqual(active, [], "朝向奖励必须处于关闭状态（横向/朝向已交给 PD）")

    def test_wrap_to_pi_keeps_heading_error_bounded(self):
        """角度误差必须 wrap 到 [-π, π]，否则跨越 ±π 时会出现 2π 跳变。"""
        import math

        def yaw_of(y):
            import torch as _t
            # 仅 yaw 的纯四元数：w=cos(y/2), z=sin(y/2)
            w, z = math.cos(y / 2), math.sin(y / 2)
            return math.atan2(2 * w * z, 1 - 2 * z * z)

        def wrap(a):
            return (a + math.pi) % (2 * math.pi) - math.pi

        for deg in (-179, -90, 0, 90, 179, 181, 359):
            err = wrap(yaw_of(math.radians(deg)))
            self.assertLessEqual(abs(err), math.pi + 1e-9,
                                 f"{deg}° 的误差 {err} 超出 [-π, π]")
        # 同朝向时误差必须精确为 0
        self.assertAlmostEqual(wrap(yaw_of(0.3) - yaw_of(0.3)), 0.0, places=10)

    def test_tracking_reward_uses_measured_velocity_and_shaping_term_is_gone(self):
        """跟踪项必须比**实测速度**，且 `reference_tracking` 随残差方案彻底删除。

        2026-09-23 曾把线性误差写成 `reference_command − user_command`（上层自己的积分指令
        vs 任务指令），这与 `velocity_tracking_exp` 的名字和奖励文档（"实际速度 vs 命令期望"）
        都不符，而且和 `reference_tracking_l2` 重复。2026-10-08 动作改成关节残差后
        `reference_command` 不存在，该式若保留会恒为 0（`exp(0)=1` 变成白送的正奖励）。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        # 跟踪项：实测机体系**纵向**线速度 vs 脚本指令（2026-10-09 起横向/朝向由 PD 负责，
        # 所以跟踪误差只留 vx，`use_lateral_and_heading` 默认 False）
        self.assertIn(
            "forward_error = ((term._asset.data.root_lin_vel_b[:, 0] - term.loco_command[:, 0])", mdp)
        self.assertIn("if not use_lateral_and_heading:", mdp)
        self.assertIn("return torch.exp(-forward_error.square())", mdp)
        self.assertIn("use_lateral_and_heading=False", cfg)
        # 恢复旧口径的分支仍在（含 vy 与 yaw 角速度），但只能是显式打开
        self.assertIn("term._asset.data.root_lin_vel_b[:, 1] - term.loco_command[:, 1]", mdp)
        self.assertIn("term._asset.data.root_ang_vel_b[:, 2] - term.loco_command[:, 2]", mdp)
        # 注册项必须彻底消失（注释里保留历史说明是允许的）
        self.assertIn("原先与之并列的", cfg)          # 历史说明仍在（防止本次改动被回滚）
        self.assertNotIn("reference_tracking = RewTerm", cfg)
        self.assertNotIn("func=mdp.reference_tracking_l2", cfg)
        self.assertNotIn("def reference_tracking_l2(", mdp)
        # 代码里不能再有任何 reference_command（文档串里保留历史说明是允许的）
        self.assertNotIn("self.reference_command", mdp)
        self.assertNotIn("term.reference_command", mdp)
        # 治抖动的两项仍在，且权重沿用 2026-09-23 的重调值
        self.assertIn("def action_magnitude_l2(", mdp)
        self.assertIn("action_magnitude = RewTerm(func=mdp.action_magnitude_l2, weight=-0.05)", cfg)
        self.assertIn("action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)", cfg)

    def test_upper_action_is_applied_as_a_residual_on_frozen_joint_targets(self):
        """残差必须加在**冻结策略的关节位置目标**上，且不能污染冻结策略自己的观测。

        三件事一起守：
        1. 动作维数来自冻结策略契约的关节数（12），不是写死的 3；
        2. 每次底层刷新都用当前 `delta_joint_pos` 重算 `joint_targets + 残差`
           （残差 50 ms 变、底层输出 20 ms 变，只在 50 ms 处算一次会用错值）；
        3. `FrozenLowLevelPolicy` 的输入仍只有 `loco_command` 与本体状态——残差若进了
           它自己的 45 维观测（尤其 `last_action`），冻结契约就被改写成另一个策略了。
        """
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("return self._action_dim", mdp)
        self.assertIn("self._action_dim = len(self._policy_cfg.joint_names)", mdp)
        self.assertIn("residual_scale=tuple(self._policy_cfg.action_scale)", mdp)
        self.assertIn("self.delta_joint_pos.copy_(self._processed * self._residual_scale)", mdp)
        self.assertIn("output.joint_targets + self.delta_joint_pos", mdp)
        self.assertIn("velocity_command=self.loco_command", mdp)
        # 残差只允许出现在加法与自身状态更新处，不能出现在冻结策略的观测部件里
        parts_call = mdp[mdp.index("output = self._policy.step(parts_from_robot_state("):
                         mdp.index("self.last_loco_action.copy_(output.action)")]
        self.assertNotIn("delta_joint_pos", parts_call)

    def test_rl_lab_dimension_contract_matches_upper_logic(self):
        """rl_lab 侧的维数字面量必须与 `upper_logic` 的契约一致（单一事实来源）。

        `TowingVecEnvWrapper` 刻意不导入 isaac 侧包（保持离线可导入），所以那边写的是
        字面量；本测试是两处之间唯一的交叉校验，防止只改一边。
        """
        wrapper = (RL / "scripts/rl_lab/rl_lab/wrapper/towing_vec_env_wrapper.py").read_text("utf-8")
        runner = (RL / "scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py").read_text("utf-8")
        decoder = (RL / "scripts/rl_lab/rl_lab/modules/towing_decoder.py").read_text("utf-8")
        obs = logic.UpperObservationSpec()
        dec = logic.DecoderSpec()
        self.assertIn(f"self.num_obs != {obs.frame_dim}", wrapper)
        self.assertIn(f"self.num_decoder_obs != {dec.dim + 1}", wrapper)
        self.assertIn("decoder[:, :6], decoder[:, 6]", wrapper)
        # runner 的 actor 输入维数必须由 decoder 的输出维数推导，而不是硬编码 +5/+6
        self.assertIn("actor_obs_dim = env.num_obs + self.decoder.actor_feature_dim", runner)
        # decoder 的 frame_dim 必须等于 policy 帧维数（runner 会在启动时断言这一点，
        # 但配置写错时应当在这里就失败，而不是等训练机起环境）
        for path, label in ((PKG / "agents/upper_ppo_cfg.py", "upper_ppo_cfg"),
                            (RL / "scripts/rl_lab/rl_lab/config/towing_algorithm_cfg.py",
                             "towing_algorithm_cfg")):
            frame_dim = int(re.search(r"frame_dim[=:]\s*(?:int\s*=\s*)?(\d+)",
                                      path.read_text("utf-8")).group(1))
            self.assertEqual(frame_dim, obs.frame_dim,
                             f"{label} 的 frame_dim={frame_dim} 与 policy 帧 {obs.frame_dim} 不一致")
        # decoder 模块的 OUTPUT_DIM 就是 DecoderSpec.dim
        self.assertIn(f"OUTPUT_DIM = VELOCITY_DIM + MASS_DIM + FORCE_DIM", decoder)
        self.assertEqual(dec.dim, 6)
        self.assertEqual(obs.frame_dim, 57)
        self.assertEqual(obs.actor_dim, obs.frame_dim + dec.dim + obs.latent_dim)

    def test_towing_runner_zero_inits_actor_output_layer(self):
        """残差策略必须从 0 起步：只对 towing runner 零初始化 actor 末层。

        `ActorCritic` 默认随机初始化末层，实测首拍 |a|max ≈ 0.149（归一化，约 ±0.04 rad
        的逐关节系统性偏置），使每回合起点都偏离冻结步态；但 `ActorCriticRecurrent` 由
        ppo／amp／himloco 共用，**不能**改模块默认初始化，所以这条只在 runner 里生效。
        """
        runner = (RL / "scripts/rl_lab/rl_lab/runners/towing_on_policy_runner.py").read_text("utf-8")
        self.assertIn("isinstance(layer, torch.nn.Linear)][-1]", runner)
        self.assertIn("torch.nn.init.zeros_(actor_last_linear.weight)", runner)
        self.assertIn("torch.nn.init.zeros_(actor_last_linear.bias)", runner)
        module = (RL / "scripts/rl_lab/rl_lab/modules/actor_critic.py").read_text("utf-8")
        self.assertNotIn("zeros_", module, "共用模块不能被加进零初始化")
        for other in ("ppo_on_policy_runner.py", "amp_on_policy_runner.py",
                      "him_on_policy_runner.py"):
            text = (RL / "scripts/rl_lab/rl_lab/runners" / other).read_text("utf-8")
            self.assertNotIn("actor_last_linear", text, f"{other} 不应受 towing 改动影响")

    def test_reset_event_contract_has_all_v0_work_condition_axes(self):
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        for fragment in (
            '"speed_range": SPEED_RANGE',
            '"mass_range": (5.0, 15.0)',
            '"friction_range": (0.4, 1.2)',
            '"wheel_damping_range": (0.008, 0.032)',
            '"no_cart_fraction": 0.125',
            '"no_cart_lateral_offset": 2.0',
        ):
            self.assertIn(fragment, cfg)
        self.assertIn("default_masses[ids_cpu] * scale[:, None]", mdp)
        self.assertIn("default_inertias[ids_cpu] * scale[:, None, None]", mdp)
        # 类型/长度来自确定性网格，不再是逐 env 随机采样
        self.assertIn("specs = [env_spec(index) for index in range(env.num_envs)]", mdp)
        self.assertNotIn("torch.randint(0, len(CONNECTION_MODELS)", mdp)
        self.assertIn("self._was_stopped", mdp)
        self.assertNotIn("stop_time_range", cfg)      # STOP 由进度触发，不是随机时刻
        self.assertIn("self._apply_towing_physics()", mdp)
        self.assertIn("self._physics_step % low_level_decimation", mdp)
        self.assertIn("-self.wheel_damping * self._cart.data.joint_vel", mdp)
        self.assertIn("MultiRopeModel(", mdp)
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

    def test_three_connection_types_are_fixed_by_the_column_grid(self):
        """三类连接（弹性绳 / 低弹性绳 / 刚体球铰连杆）必须逐 env 共存，且由**列**决定。

        刚体连杆是**双边**模型，所以 `ROPE_MODELS` 仍只含两套绳，`CONNECTION_MODELS` 才是
        三类的完整集合——避免把「绳只能拉」的单边不变量套到连杆上（见 rope_model.py）。
        2026-10-08 起类型不再逐 env 随机，而是 `connection_grid` 按 env index 的列映射。
        """
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("from .mdp.connection_grid import", mdp)
        self.assertIn("models=(compliant, inextensible, rigid)", mdp)
        self.assertIn("[[spec[\"model_index\"]] for spec in specs]", mdp)
        # 类型张量在 __init__ 里一次算好、reset 不再改写
        self.assertNotIn("term.rope_model_id[env_ids, 0] =", mdp)

    def test_grid_spawn_places_the_cart_by_connection_length(self):
        """三类连接都按**本 env 的目标挂点距**摆放小车，且先摆位、后叠加无小车横移。

        否则第一物理步就有初始力（绳预张紧、刚体初始压缩/拉伸）。目标值来自网格：
        绳 = 0.5·L0（留松弛）、刚体 = L，由 `initial_distance` 张量提供。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("num_envs=COLUMNS * ROWS", cfg)
        self.assertIn("self.initial_distance = torch.tensor(", mdp)
        self.assertIn("target_distance = term.initial_distance[env_ids, 0]", mdp)
        # spawn 摆放：水平分量 = sqrt(target² − Δz²)（纯几何函数），方向沿机器人正后方，
        # 挂点用实际 spawn 位姿算
        self.assertIn("attachment_horizontal_gap(target_distance, delta_z=normal_difference)", mdp)
        self.assertIn("robot_attach_w = math_utils.quat_apply(", mdp)
        # 顺序：先按连接长度摆放，再加无小车横移（反了会覆盖横向停放）
        self.assertLess(mdp.index("along = attachment_horizontal_gap("),
                        mdp.index("cart_state[~cart_present, 1] += no_cart_lateral_offset"))
        # 旧的「只在刚体分支里摆放」已被三类统一摆放取代
        self.assertNotIn("rigid_link_horizontal_gap", mdp)

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
        self.assertIn("self.towing_force_b[:] = force_robot_b[:, 0, :3]", mdp)
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
        self.assertIn("augment_actor_observation(raw_obs, estimate, latent)", runner)
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

    def test_stop_phase_replaces_goal_termination_and_has_its_own_rewards(self):
        """2026-10-09：目标制改回**三段制**——到点只置零指令、回合继续跑到 timeout，
        两条 post_stop 奖励因此重新有相位可用（用户明确要求保留这两项）。

        用户还要求「给 cmd vel 一定要在越过坡之后」，所以 STOP 触发点必须严格大于坡面出口
        `FLAT_OUT_START_M = 9.0 m`，`upper_env_cfg.__post_init__` 有断言守着。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("time_out = DoneTerm(func=mdp.time_out, time_out=True)", cfg)
        # 到点不再终止（否则 STOP 之后没有可评分的步）
        self.assertNotIn("goal_reached = DoneTerm", cfg)
        self.assertNotIn("stop_reached = DoneTerm", cfg)
        self.assertIn("def stop_reached(env):", mdp)
        self.assertIn("progress - term.start_progress >= term.cfg.stop_distance_m", mdp)
        # STOP 触发点必须在坡面出口之后
        self.assertIn("from imgo2_rl.tasks.manager_based.towing.mdp.slope_geometry import", cfg)
        self.assertIn("FLAT_OUT_START_M", cfg)
        self.assertIn("if action.stop_distance_m <= FLAT_OUT_START_M:", cfg)
        # timeout 用**坡面弧长上界** + 停车窗口，不是水平目标距离
        self.assertIn("profile_arc_length(MAX_GRADE_DEG, action.stop_distance_m)", cfg)
        self.assertIn("margin=POST_STOP_WINDOW_S)", cfg)
        # 用户 2026-10-09 确认：正常回合**一律超时退出**，不加停车窗口终止项
        self.assertNotIn("post_stop_timeout = DoneTerm", cfg)
        self.assertNotIn("def post_stop_timeout", mdp)
        self.assertNotIn("goal_reached = DoneTerm", cfg)
        # "走没走到 STOP 点"改由诊断量回答（0/1 粘性标志，只进 TensorBoard）
        self.assertIn("obs_stop_reached = RewTerm(func=mdp.stop_reached_flag, weight=1.0e-6)", cfg)
        self.assertIn("def stop_reached_flag(env):", mdp)
        self.assertIn("return _term(env)._was_stopped.float()", mdp)
        # 倒地退出必须在位（失败退出，不算超时）
        self.assertIn("robot_fall = DoneTerm(func=mdp.robot_fall, params={\"minimum_height\": 0.18})", cfg)
        # 两条 post_stop 奖励恢复，权重与历史一致
        self.assertIn("stop_towing_force = RewTerm(func=mdp.post_stop_towing_force, weight=-1.0", cfg)
        self.assertIn("extra_distance = RewTerm(func=mdp.post_stop_distance, weight=-0.1)", cfg)
        self.assertIn('params={"force_scale": 10.0}', cfg)
        # 相位与状态：触发时记录 stop_time_s / stop_origin_x，复位回 +inf
        self.assertIn("def post_stop_towing_force(env, force_scale):", mdp)
        self.assertIn("def post_stop_distance(env):", mdp)
        self.assertIn("post_stop = (elapsed_s >= term.stop_time_s).float()", mdp)
        self.assertIn("self.stop_time_s[newly_stopped] = elapsed_s[newly_stopped]", mdp)
        self.assertIn("self.stop_origin_x[newly_stopped] = self._asset.data.root_pos_w[newly_stopped, 0]", mdp)

    def test_yaw_is_the_third_euler_component_not_the_first(self):
        """守卫：`euler_xyz_from_quat` 的返回顺序是 **(roll, pitch, yaw)**。

        2026-10-09 发现 `heading_deviation` 原先写成 `yaw, _pitch, _roll = ...`，实际取到的是
        **roll** ⇒ `yaw_heading`（−2.0）一直在罚横滚偏差，而朝向只被 `tracking_velocity` 的
        yaw **角速度**项间接约束（角速度归零 ≠ 朝向不漂）。这里用纯 yaw 四元数按 Isaac Lab 的
        公式复算，把顺序钉死（不需要 torch，纯标准库）。
        """
        import math
        for yaw in (0.0, 0.3, -0.7):
            qw, qz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
            qx = qy = 0.0
            roll = math.atan2(2.0 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
            pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
            yaw_out = math.atan2(2.0 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            self.assertAlmostEqual(roll, 0.0, places=12)
            self.assertAlmostEqual(pitch, 0.0, places=12)
            self.assertAlmostEqual(yaw_out, yaw, places=12)      # ← 第三个才是 yaw
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("def yaw_from_quat(quat):", mdp)
        self.assertIn("_roll, _pitch, yaw = math_utils.euler_xyz_from_quat(quat)", mdp)
        # 旧写法不得回潮
        self.assertNotIn("euler_xyz_from_quat(self._asset.data.root_quat_w)[0]", mdp)
        self.assertNotIn("yaw, _pitch, _roll = math_utils.euler_xyz_from_quat", mdp)
        # 朝向奖励仍以 spawn 朝向为基准
        self.assertIn("yaw_init = yaw_from_quat(term._asset.data.default_root_state[:, 3:7])", mdp)

    def test_lane_keeping_pd_matches_the_measurement_bench(self):
        """横向/朝向 PD 必须与测量台 `lane_keeping_command` 同式同号，且限幅在训练分布内。

        用户 2026-10-09 决定：**保持中线和 heading 用 PD，不让策略学**。这样不改任何观测/动作
        维数（`loco_command[:,1:3]` 本来就是冻结策略的 (vy, ω) 指令槽，此前恒 0），也不新增奖励。
        与测量台一致才能做「PD 基线 vs 上层策略」的对照。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("if self.cfg.lane_keeping:", mdp)
        self.assertIn("self.loco_command[:, 1] = self._lane_keeping_vy()", mdp)
        self.assertIn("self.loco_command[:, 2] = self._lane_keeping_wz()", mdp)
        # 与测量台逐年项同式同号
        self.assertIn("lateral_error = -local[:, 1] * torch.cos(yaw)", mdp)
        self.assertIn("heading_error = math_utils.wrap_to_pi(self.cfg.lane_yaw_target - yaw)", mdp)
        self.assertIn("self.cfg.lane_kp_y * lateral_error", mdp)
        self.assertIn("self.cfg.lane_kd_y * self._asset.data.root_lin_vel_b[:, 1]", mdp)
        self.assertIn("self.cfg.lane_kp_yaw * heading_error", mdp)
        self.assertIn("self.cfg.lane_kd_yaw * self._asset.data.root_ang_vel_b[:, 2]", mdp)
        self.assertIn("torch.clamp(command, -self.cfg.lane_max_vy, self.cfg.lane_max_vy)", mdp)
        self.assertIn("torch.clamp(command, -self.cfg.lane_max_wz, self.cfg.lane_max_wz)", mdp)
        # 默认增益与测量台一致；限幅 = AMP 训练分布（lin_vel_y ±1.0、ang_vel_z ±1.57）
        for fragment in ("lane_kp_y: float = 1.0", "lane_kd_y: float = 0.3",
                         "lane_kp_yaw: float = 1.5", "lane_kd_yaw: float = 0.3",
                         "lane_max_vy: float = 1.0", "lane_max_wz: float = 1.5708",
                         "lane_yaw_target: float = 0.0"):
            self.assertIn(fragment, mdp)
        # PD 不是奖励项：环境配置里不得把它注册成 RewTerm
        self.assertNotIn("RewTerm(func=mdp.lane_keeping", cfg)
        self.assertNotIn("RewTerm(func=mdp._lane_keeping", cfg)

    def test_towing_force_y_penalty_is_tension_invariant(self):
        """拉力 y 分量惩罚：必须用 `(F_y/‖F‖)²` 的**比值**，不是原始 `F_y²`。

        用户 2026-10-09 要求「运动过程中 3 维拉力在 y 维度保持为 0」。用比值的原因：原始
        `F_y²` 会被起步绷直的百牛级峰值（最硬弹性档 ~1 kN）放大，等价于变相惩罚大张力，
        与拖曳任务对冲；比值与张力无关、有界 [0,1]，语义是"拉力方向落在机体系 xz 平面内"。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("def towing_force_y_ratio_sq(env):", mdp)
        self.assertIn("force = term.towing_force_b", mdp)
        self.assertIn("norm = torch.linalg.vector_norm(force, dim=1).clamp_min(1.0e-6)", mdp)
        self.assertIn("return (force[:, 1] / norm).square()", mdp)
        # 守卫：不得退化成原始 F_y²（无归一化）
        body = mdp[mdp.index("def towing_force_y_ratio_sq("):]
        body = body[:body.index("\ndef ", 1)]
        code = body[body.index('"""', body.index('"""') + 3) + 3:]
        self.assertNotIn("force[:, 1].square()", code)
        self.assertIn("towing_force_y = RewTerm(func=mdp.towing_force_y_ratio_sq, weight=-10.0)", cfg)
        self.assertIn("math.inf", mdp)

    def test_mesh_origins_and_fall_test_follow_the_profile(self):
        """地形 origin 来自 mesh importer；跌倒判据量的是**离局部剖面**的高度。

        2026-10-09 起 lane 是「平地 → 上坡 → 坡顶 → 下坡 → 平地」的连续剖面，绝对 z 或
        单一 `terrain_normal_w` 都不再是「离坡面多高」，必须减去 `profile_height(grade, x)`。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("class_type=TowingSlopeTerrainImporter", cfg)
        self.assertIn("terrain_exit = DoneTerm(func=mdp.terrain_out_of_bounds)", cfg)
        self.assertIn("profile_height_tensor(term.hill_grade_deg", mdp)
        self.assertIn("from .mdp.profile_torch import profile_height_tensor", mdp)
        # 前进量仍然沿 lane 的切向（出生在平地段 ⇒ 切向 = +x）
        self.assertIn("* term.terrain_tangent_w).sum(dim=1)", mdp)
        self.assertNotIn("term.slope_angle", mdp)


if __name__ == "__main__":
    unittest.main()
