"""Offline contract tests for the first upper-RL environment milestone."""

import ast
import importlib.util
import math
from pathlib import Path
import re
import statistics
import sys
import textwrap
import types
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


def _spawn_gap_constants():
    """从源码读出「出生间隙 − 绝对死区」公式的全部常量（测试里不再硬编码一遍）。

    `min_clearance` 的阈值自 2026-10-10 起是 **`出生间隙 − deadband_m`**（绝对死区），其中

        spawn_gap(L, 类型) = sqrt(initial_attachment_distance(类型, L)² − Δz²) + base_offset

    - `deadband_m` = `UpperRewardsCfg.min_clearance` 的 `params`（默认 0.02 m）；
    - `Δz` = 两挂点高差 = 机器人/小车的出生高度之差（`assets/imgo2.py` 的 `init_state.pos`
      与 `upper_env_cfg` 里 cart 的 `init_state.pos`；两挂点的 z 偏移都是 0）；
    - `base_offset = −robot_rear_surface_x − robot_attachment_x + cart_attachment_x
      − cart_front_surface_x`（= 「后表面↔车斗前表面」比挂点距多出来的固定量，四个常量都从
      `upper_mdp.py` 的 action-term cfg 读）。
    """
    cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
    mdp = (PKG / "upper_mdp.py").read_text("utf-8")
    assets = (RL / "source/imgo2_rl/imgo2_rl/assets/imgo2.py").read_text("utf-8")

    def one(pattern, text):
        return float(re.search(pattern, text).group(1))

    deadband_m = one(r'"deadband_m": ([0-9.]+)', cfg)
    softness = one(r'"softness": ([0-9.]+)', cfg)
    robot_z = one(r"pos=\(0\.0, 0\.0, ([0-9.]+)\)", assets)
    cart_z = one(r"cart\.init_state\.pos = \(-?[0-9.]+, [0-9.]+, ([0-9.]+)\)", cfg)
    rear = one(r"robot_rear_surface_x: float = ([0-9.]+)", mdp)
    front = one(r"cart_front_surface_x: float = ([0-9.]+)", mdp)
    robot_attach_x = one(r"robot_attachment: tuple\[float, float, float\] = \((-?[0-9.]+)", mdp)
    cart_attach_x = one(r"cart_attachment: tuple\[float, float, float\] = \((-?[0-9.]+)", mdp)
    base_offset = -rear - robot_attach_x + cart_attach_x - front
    return deadband_m, softness, robot_z - cart_z, base_offset


def _spawn_gap(model_name, length, delta_z, base_offset):
    """出生瞬间的「后表面 → 车斗前表面」间隙（与 `upper_mdp.spawn_clearance` 同式）。"""
    target = grid.initial_attachment_distance(model_name, length)
    return math.sqrt(target ** 2 - delta_z ** 2) + base_offset


_FUNCTION_CACHE: dict = {}


def _mdp_function(name):
    """把 `upper_mdp.<name>` 的函数体从源码里抽出来，在只含 `torch` + `_term` 的 stub 里实跑。

    本机没有 Isaac Lab（`upper_mdp` 顶层 import isaaclab，离线 import 不了），所以只能这样
    拿到**真函数**；与 `test_towing_force_rate.py` 的做法一致。⚠ 返回的是普通函数，必须经
    `_mdp_function(name)(env, ...)` 调用，不要挂到 `TestCase` 上（会变成方法绑定）。
    """
    if name not in _FUNCTION_CACHE:
        source = (PKG / "upper_mdp.py").read_text("utf-8")
        node = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
        body = textwrap.dedent(ast.get_source_segment(source, node))
        namespace = {"torch": torch, "_term": lambda env: env.term}
        exec(body, namespace)  # noqa: S102 - 源码来自本仓库，测试专用
        _FUNCTION_CACHE[name] = namespace[name]
    return _FUNCTION_CACHE[name]


class _ClearanceTermStub:
    """`min_clearance_violation` 需要的最小 term 接口。"""

    def __init__(self, clearance, spawn_clearance, cart_present=1.0):
        self.rope_state = torch.tensor([[clearance]])
        self.spawn_clearance = torch.tensor([spawn_clearance])
        # 真实 term 里 `cart_present` 是 [num_envs, 1]（函数里取 `[:, 0]`）
        self.cart_present = torch.tensor([[cart_present]])
        self.safety_updates = 0

    def update_safety_state(self):
        self.safety_updates += 1


def _clearance_env(clearance, spawn_clearance, cart_present=1.0):
    env = types.SimpleNamespace(term=_ClearanceTermStub(clearance, spawn_clearance, cart_present))
    return env


def _evaluate_min_clearance(clearance, spawn_clearance, deadband_m, softness=0.02, cart_present=1.0):
    """在 stub term 上调用**源码里那个** `min_clearance_violation`，返回 python float。"""
    env = _clearance_env(clearance, spawn_clearance, cart_present)
    value = _mdp_function("min_clearance_violation")(env, deadband_m, softness)
    assert env.term.safety_updates == 1, "必须先 update_safety_state() 再读 rope_state"
    return float(value[0])


def _float32_threshold(spawn_clearance, deadband_m):
    """函数内那一步 `spawn_clearance − deadband_m` 的 **float32** 结果（边界用例要用同一步运算取）。

    `term.spawn_clearance` 是 float32 张量，阈值在函数里就是 float32 减标量的结果；测试里若用
    float64 的 `gap − deadband_m` 去当"阈值正好落在间隙上"的输入，会因为 float32 舍入差出
    1e-8 量级（`compliant` row 5 实测 2.98e-8），把"阈值处精确 0"误判成失败。
    """
    return float(torch.tensor([spawn_clearance]) - deadband_m)


class UpperLogicTests(unittest.TestCase):
    def test_actor_contract_matches_paper_plan(self):
        """双头契约 v3：**58** 维单帧 + 6 维 estimate = **80** 维 actor 输入；动作 **13**。

        2026-10-10 由单头（帧 57 / actor 79 / 动作 12）迁移而来：`last_action` 由 12 维
        关节残差变成 13 维（1 维 vx 偏移 + 12 维关节残差），故帧 +1、actor +1。
        命令项 `loco_command` 现在是「任务指令 + 有界偏移」（策略看得见自己下了什么指令），
        奖励参考量改读 `task_command`（防作弊红线）。
        """
        spec = logic.UpperObservationSpec()
        self.assertEqual([name for name, _ in spec.terms],
                         ["loco_command", "last_action", "base_ang_vel", "projected_gravity",
                          "last_loco_action", "joint_pos", "joint_vel"])
        self.assertEqual(spec.frame_dim, 58)
        self.assertEqual(spec.decoder_dim, 6)
        self.assertEqual(spec.actor_dim, 80)
        self.assertEqual(dict(spec.terms)["loco_command"], 3)
        self.assertEqual(dict(spec.terms)["last_action"], 13)
        # 动作布局常量：13 = 1（vx 偏移）+ 12（关节残差），切分点唯一定义在 upper_logic
        self.assertEqual(logic.CMD_ACTION_DIM, 1)

    def test_work_domain_ranges_do_not_touch_the_contract(self):
        """工作域常量（质量 5–20 kg、速度 0.5–1.5 m/s）**不得**改动契约。

        2026-10-10 用户收紧工作域：`reset_work_condition.mass_range` 与
        `DecoderSpec.mass_range` 上限 30 → 20 kg、`episode_geometry.SPEED_RANGE` 下界
        0.4 → 0.5 m/s（回合超时随之由 29.231 s 缩短到 24.185 s，见几何测试）。本守卫把
        「工作域」与「契约」隔开：三个训练域常量必须**四处一致**（env cfg 源码 /
        `DecoderSpec.mass_range` / `episode_geometry.SPEED_RANGE` / 测量台
        `TRAINING_MASS_RANGE_KG`），而契约维数 **58 / 6 / 80 / 动作 13** 与
        `towing_contract v3` 不受这些范围影响——`DecoderSpec.mass_range` 只作文档/派生用，
        改它不动 `dim`。
        """
        import dataclasses

        geometry = load("towing_episode_geometry_contract_test",
                        PKG / "mdp/episode_geometry.py")
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        play = (RL / "scripts/towing/play_towing_test.py").read_text("utf-8")

        # 工作域：训练侧四处常量逐项对齐（少改一处就会漂移）
        self.assertEqual(geometry.SPEED_RANGE, (0.5, 1.5))
        self.assertIn('"speed_range": SPEED_RANGE', cfg)
        self.assertIn('"mass_range": (5.0, 20.0)', cfg)
        self.assertEqual(logic.DecoderSpec().mass_range, (5.0, 20.0))
        match = re.search(r"TRAINING_MASS_RANGE_KG = \(([0-9.]+), ([0-9.]+)\)", play)
        self.assertIsNotNone(match, "测量台找不到 TRAINING_MASS_RANGE_KG")
        self.assertEqual((float(match.group(1)), float(match.group(2))),
                         logic.DecoderSpec().mass_range,
                         "测量台的训练侧质量范围必须与 DecoderSpec.mass_range 一致")

        # 契约：工作域改了也不许动这些维数/版本（改 mass_range 后 dim 不变）
        spec = logic.UpperObservationSpec()
        widened = dataclasses.replace(logic.DecoderSpec(), mass_range=(1.0, 99.0))
        self.assertEqual(widened.dim, logic.DecoderSpec().dim)
        self.assertEqual((spec.frame_dim, spec.decoder_dim, spec.actor_dim), (58, 6, 80))
        self.assertEqual(dict(spec.terms)["last_action"], 13)
        self.assertEqual(logic.CMD_ACTION_DIM, 1)
        runtime_src = (RL / "scripts/towing/upper_policy_runtime.py").read_text("utf-8")
        self.assertIn("CHECKPOINT_CONTRACT_VERSION = 3", runtime_src)

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
        """动作 = 13 维（1 维 vx 偏移 + 12 维归一化关节残差）。

        关节头映射为 `residual_scale ⊙ clip(u_joint, ±1)`；残差尺度取冻结策略契约的
        `action_scale`，因此幅值受底层动作范围界定。偏移头是**加性 + 有界**：
        `offset = clip(u_cmd, ±1) × cmd_offset_scale`，再限在 `[offset_min, offset_max]`。
        旧的 3 维加速度映射／参考速度积分已删除，不能留下任何可用入口。
        """
        scale = (0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25)
        spec = logic.UpperActionSpec(residual_scale=scale, control_dt=0.05)
        spec.validate()
        self.assertEqual(spec.joint_action_dim, 12)
        self.assertEqual(spec.action_dim, 13)
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

    def test_action_split_is_one_command_plus_twelve_joints(self):
        """动作切分：`u = cat(u_cmd(1), u_joint(12))`，两头各自 clamp 到 ±1。

        这是双头契约最容易静默错位的地方（差一位 ⇒ 偏移头吃掉了第一个关节的残差），
        所以切分必须只有**一个**定义处（`UpperActionSpec.split_action` / `CMD_ACTION_DIM`），
        并且维数写错时必须硬报错而不是静默截断。
        """
        spec = logic.UpperActionSpec(residual_scale=(0.25,) * 12)
        command, joint = spec.split_action([0.5] + [0.1] * 12)
        self.assertEqual((command, joint), ((0.5,), (0.1,) * 12))
        # 越界逐维裁剪（不是整体裁剪、不是线性外推）
        command, joint = spec.split_action([5.0] + [-5.0] * 12)
        self.assertEqual(command, (1.0,))
        self.assertEqual(joint, (-1.0,) * 12)
        # 维数不符必须报错：11/12/14 维都不行
        for bad in ([0.0] * 11, [0.0] * 12, [0.0] * 14):
            with self.assertRaises(ValueError):
                spec.split_action(bad)
        # `delta_joint_pos` 只吃 12 维关节头；把 13 维整动作喂进去必须报错（防切分写错）
        with self.assertRaises(ValueError):
            spec.delta_joint_pos([0.0] * 13)

    def test_command_offset_is_bounded_additive_and_gated_off_at_spawn(self):
        """偏移头：加性 + 有界，且出生段（`elapsed < tow_start`）不生效。

        2026-10-10 从源码核对后的**两层限幅**（顺序不能反）：

        1. `offset = clip(u_cmd, ±1) × cmd_offset_scale`，再限在头权限 `[offset_min, offset_max]`；
        2. **和**再限在冻结 AMP 策略的训练包络 `amp_vx_range = (−1.0, 1.5)`
           （`amp_env_cfg` 的 `lin_vel_x`）—— 超出即 OOD、步态退化。

        ⚠ 绝不能把和裁到 `[−0.2, +0.6]`：脚本速度 0.5–1.5，牵引段本来就顶在 AMP 上界 1.5，
        那样会把正常牵引指令压成 0.6 而奖励参考仍是 1.5。**退化性**：`u_cmd = 0` 或
        `apply_offset=False` 时脚本值落在包络内 ⇒ 裁剪恒等 ⇒ 与旧口径逐位一致。
        """
        spec = logic.UpperActionSpec(
            residual_scale=(0.25,) * 12, cmd_offset_scale=0.5, offset_min=-0.2, offset_max=0.6,
            amp_vx_range=(-1.0, 1.5))
        # 尺度与头权限
        self.assertAlmostEqual(spec.command_offset([0.6] + [0.0] * 12), 0.3)
        self.assertAlmostEqual(spec.bounded_command_offset([1.0] + [0.0] * 12), 0.5)
        self.assertAlmostEqual(spec.bounded_command_offset([-1.0] + [0.0] * 12), -0.2)
        self.assertAlmostEqual(spec.bounded_command_offset([-5.0] + [0.0] * 12), -0.2)
        # 牵引段顶在上界：1.5 + 0.5 被包络裁回 1.5（正半轴在高速时无梯度，只能减速）
        self.assertAlmostEqual(spec.loco_command_vx([1.0] + [0.0] * 12, 1.5), 1.5)
        self.assertAlmostEqual(spec.loco_command_vx([-1.0] + [0.0] * 12, 1.5), 1.3)
        # STOP 后 task_vx = 0：偏移头给出 ±0.5（含倒走），且不越包络
        self.assertAlmostEqual(spec.loco_command_vx([1.0] + [0.0] * 12, 0.0), 0.5)
        self.assertAlmostEqual(spec.loco_command_vx([-1.0] + [0.0] * 12, 0.0), -0.2)
        # 低速段的加性（不会被 [-0.2,0.6] 那种错误口径砍掉）
        self.assertAlmostEqual(spec.loco_command_vx([1.0] + [0.0] * 12, 0.4), 0.9)
        # 出生段门控：偏移不生效，且脚本值在包络内 ⇒ 逐位不变
        for scripted in (0.0, 0.4, 0.7, 1.5):
            self.assertEqual(spec.loco_command_vx([1.0] + [0.0] * 12, scripted,
                                                  apply_offset=False), scripted)
        # 退化性：u_cmd = 0 时逐位等于脚本值（== 旧口径）
        for scripted in (0.0, 0.4, 0.7, 1.5):
            self.assertEqual(spec.loco_command_vx([0.0] * 13, scripted), scripted)
        # 非法尺度/限幅/包络要报错，不能静默
        with self.assertRaises(ValueError):
            logic.UpperActionSpec(residual_scale=(0.25,) * 12, cmd_offset_scale=0.0).validate()
        with self.assertRaises(ValueError):
            logic.UpperActionSpec(residual_scale=(0.25,) * 12, offset_min=0.6,
                                  offset_max=-0.2).validate()
        with self.assertRaises(ValueError):
            logic.UpperActionSpec(residual_scale=(0.25,) * 12,
                                  offset_min=float("nan")).validate()
        with self.assertRaises(ValueError):
            logic.UpperActionSpec(residual_scale=(0.25,) * 12,
                                  amp_vx_range=(0.4, 1.5)).validate()   # 不含 0
        with self.assertRaises(ValueError):
            logic.UpperActionSpec(residual_scale=(0.25,) * 12,
                                  amp_vx_range=(1.5, -1.0)).validate()  # 反序

    def test_amp_vx_range_matches_the_frozen_policy_command_distribution(self):
        """`amp_vx_range` 必须逐字等于冻结 AMP 策略的指令范围（源码是唯一事实来源）。

        出处：`base_move/amp_env_cfg.py::__post_init__` 的
        `self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)`。
        它是**分布约束**而不是权限：牵引速度 0.5–1.5 本来就顶在上界，超出即 OOD。
        """
        amp_cfg = (PKG.parent / "locomotion/velocity/base_move/amp_env_cfg.py").read_text("utf-8")
        match = re.search(r"ranges\.lin_vel_x\s*=\s*\(([-0-9.]+),\s*([-0-9.]+)\)", amp_cfg)
        self.assertIsNotNone(match, "amp_env_cfg 里找不到 lin_vel_x 范围")
        amp_range = (float(match.group(1)), float(match.group(2)))
        self.assertEqual(amp_range, (-1.0, 1.5))
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("amp_vx_range: tuple[float, float] = (-1.0, 1.5)", mdp)
        self.assertEqual(logic.UpperActionSpec(residual_scale=(0.25,) * 12).amp_vx_range, amp_range)
        # 包络必须覆盖脚本速度范围（SPEED_RANGE 0.5–1.5），否则零偏移时脚本被裁
        geometry = load("towing_episode_geometry_amp_test", PKG / "mdp/episode_geometry.py")
        self.assertGreaterEqual(amp_range[0], -1.0)
        self.assertGreaterEqual(amp_range[1], geometry.SPEED_RANGE[1])
        self.assertLessEqual(geometry.SPEED_RANGE[0], amp_range[1])

    def test_scheduled_vx_matches_the_old_schedule_when_ramp_is_off(self):
        """`scripted_vx` 纯函数镜像：ramp 关闭（默认 0.0）时 == 旧口径；打开时线性降到 0。"""
        # ramp 关闭：settle 0 → tow speed → STOP 立即 0（与 2026-10-10 之前逐位一致）
        self.assertEqual(logic.scripted_vx(0.5, 0.7, tow_start_s=1.0, stop_time_s=5.0), 0.0)
        self.assertEqual(logic.scripted_vx(2.0, 0.7, tow_start_s=1.0, stop_time_s=5.0), 0.7)
        self.assertEqual(logic.scripted_vx(5.0, 0.7, tow_start_s=1.0, stop_time_s=5.0), 0.0)
        # ramp = 2 s：STOP 那一拍仍是 tow_speed，之后线性降到 0，再往后恒 0
        kwargs = {"tow_start_s": 1.0, "stop_time_s": 5.0, "ramp_s": 2.0}
        self.assertAlmostEqual(logic.scripted_vx(5.0, 0.8, **kwargs), 0.8)
        self.assertAlmostEqual(logic.scripted_vx(6.0, 0.8, **kwargs), 0.4)
        self.assertAlmostEqual(logic.scripted_vx(7.0, 0.8, **kwargs), 0.0)
        self.assertEqual(logic.scripted_vx(9.0, 0.8, **kwargs), 0.0)
        with self.assertRaises(ValueError):
            logic.scripted_vx(1.0, 0.8, ramp_s=-1.0)

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
        """端到端数值契约：58 帧 → 6 维估计(GRU 128) → 80 维 actor(GRU 256) → 13 维动作；critic 73 → 1。

        这条用**真实网络类**跑一次前向，锁住三件事：
        1. 各层宽度与 `upper_logic` 契约一致（frame 58 / decoder 6 / actor 80 / action 13）；
        2. `augment_actor_observation` 的拼接结果能直接喂进 `ActorCriticRecurrent`；
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
            # critic 特权组：58 帧 + 机器人速度 2 + 小车速度 2 + 绳状态 4 + 机体系三维拉力 3
            #              + 质量/摩擦/轮阻/有无小车 4 = 73
            critic_dim = obs.frame_dim + 2 + 2 + 4 + 3 + 4
            self.assertEqual(critic_dim, 73)
            action_dim = 13

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

    def test_low_level_position_error_expectation_is_switchable(self):
        """角度跟踪项：期望可取「含残差的下发目标」（当前）或「底层原始期望」。

        用户 2026-10-09 定为 `reference="commanded"`（= 底层输出 + 残差，误差 = 纯 PD 误差）；
        另一分支 `"frozen"`（底层自己输出，误差 = δ + e_PD）保留供对照。
        两条分支都必须存在，默认必须是 `commanded`。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("def low_level_position_error_l2(env, reference=\"commanded\"):", mdp)
        self.assertIn("actual = term._asset.data.joint_pos[:, term._policy_to_asset]", mdp)
        self.assertIn("expected = term._held_joint_targets[:, term._policy_to_asset]", mdp)
        self.assertIn("expected = term.loco_joint_targets", mdp)
        self.assertIn("return (actual - expected).square().sum(dim=1)", mdp)
        self.assertIn("reference 必须是", mdp)          # 非法值要报错，不能静默取默认
        # 期望的两种来源都必须被维护：下发目标 = 底层输出 + 残差；`loco_joint_targets` 在叠加前记录
        self.assertIn("output.joint_targets + self.delta_joint_pos", mdp)
        self.assertIn("self.loco_joint_targets.copy_(output.joint_targets)", mdp)
        self.assertLess(mdp.index("self.loco_joint_targets.copy_(output.joint_targets)"),
                        mdp.index("output.joint_targets + self.delta_joint_pos"))
        self.assertIn('params={"reference": "commanded"}', cfg)
        self.assertIn(
            "low_level_pos_error = RewTerm(func=mdp.low_level_position_error_l2, weight=-0.1,", cfg)

    def test_action_term_init_never_uses_an_attribute_before_defining_it(self):
        """`HierarchicalVelocityAction.__init__` 里不得"先用后定义"。

        2026-10-09 实跑踩到：回退 `loco_joint_targets` 时把它插到了 `_policy_to_asset` 前面，
        环境一构造就
        `AttributeError: 'HierarchicalVelocityAction' object has no attribute '_policy_to_asset'`。
        本机无 Isaac Lab（无法构造环境），所以这条必须靠**源码顺序**守住：按行扫 `__init__`，
        任何 `self.X` 在 `self.X = ...` 之前出现就失败（基类 `ActionTerm` 已提供的属性白名单）。
        """
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        start = mdp.index("    def __init__(self, cfg, env):")
        end = mdp.index("\n    def ", start + 10)
        body = mdp[start:end]
        # `ActionTerm.__init__` 已经提供的属性/属性器，允许先出现
        defined = {"_cfg", "_env", "_asset", "_asset_name", "device", "num_envs", "cfg"}
        offenders = []
        for line in body.splitlines():
            code = line.split("#", 1)[0]
            if not code.strip():
                continue
            stripped = code.strip()
            match = re.match(r"self\.([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=)", stripped)
            assigned = match.group(1) if match else None
            for used in re.findall(r"self\.([A-Za-z_]\w*)", code):
                if used == assigned or used in defined:
                    continue
                offenders.append((used, stripped[:70]))
            if assigned:
                defined.add(assigned)
        self.assertEqual(offenders, [], f"__init__ 里先用后定义：{offenders}")
        # 同时把这次的顺序要求钉死：策略关节映射必须先于用到它的缓存
        self.assertLess(
            mdp.index("        self._policy_to_asset = torch.tensor("),
            mdp.index("        self.loco_joint_targets = self._asset.data.default_joint_pos["))

    def test_collision_is_a_small_sustained_penalty_not_a_termination(self):
        """用户 2026-10-09：**碰撞不再终止**，只保留惩罚且幅度调小（−50 → −5.0）。

        标定逻辑：以前 −50 是一次性的（触发即终止），现在碰撞可以持续整个回合 ⇒ 必须小一个
        量级（每步 −0.25，顶住 1 s 约 −5，与一个回合的正奖励同量级）。
        `mdp.cart_collision` 函数本身要保留（奖励与日志在用），只是不再注册成 DoneTerm。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        active = [ln.strip() for ln in cfg.splitlines()
                  if ln.strip().startswith("cart_collision = DoneTerm")]
        self.assertEqual(active, [], f"碰撞不应再是终止条件：{active}")
        self.assertIn("collision = RewTerm(func=mdp.cart_collision_cost, weight=-5.0)", cfg)
        self.assertIn("def cart_collision(env):", mdp)
        # 剩下的终止项：超时 + 倒地 + 出界
        for name in ("time_out", "robot_fall", "terrain_exit"):
            self.assertIn(f"{name} = DoneTerm(", cfg)

    def test_robot_domain_rand_matches_the_amp_velocity_task(self):
        """机器人侧 DR 必须与 AMP vel 跟踪任务逐项一致（用户 2026-10-09 要求「参考 amp 恢复」）。

        来源：`velocity_env_cfg.py` 的 `EventCfg` + `base_move/amp_env_cfg.py` 的覆盖（作用体/
        关节收窄）。四项：base 加质量、其余乘质量、base 质心、策略关节执行器增益。
        另两项**刻意不恢复**（照抄会变成空操作），守卫把原因也钉住，防止以后被「补全」。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        for name in ("apply_external_force_torque", "randomize_actuator_gains",
                     "randomize_rigid_body_com", "randomize_rigid_body_mass"):
            self.assertIn(name, mdp)
        self.assertIn("class UpperRobotDomainRandCfg:", cfg)
        self.assertIn('body_names=["base"]', cfg)
        self.assertIn('"mass_distribution_params": (-1.0, 3.0)', cfg)
        self.assertIn('"operation": "add"', cfg)
        self.assertIn('body_names=["^(?!.*base).*"]', cfg)
        self.assertIn('"mass_distribution_params": (0.7, 1.3)', cfg)
        self.assertIn('"operation": "scale"', cfg)
        self.assertEqual(cfg.count('"recompute_inertia": True'), 2)
        self.assertIn('"com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}', cfg)
        self.assertIn("randomize_robot_actuator_gains = EventTerm(", cfg)
        self.assertIn("mdp.randomize_actuator_gains", cfg)
        self.assertIn('"stiffness_distribution_params": (0.5, 2.0)', cfg)
        self.assertIn('"damping_distribution_params": (0.5, 2.0)', cfg)
        self.assertIn('"distribution": "uniform"', cfg)
        self.assertIn("joint_names=_policy_joint_names()", cfg)
        self.assertIn('def _policy_joint_names() -> list[str]:', cfg)
        self.assertIn('return list(get_policy("amp").joint_names)', cfg)
        # 不得有**生效的**注册行（docstring 里会把它们当反例引用，所以不能用子串断言）
        active = [ln.strip() for ln in cfg.splitlines()
                  if ln.strip().startswith(("randomize_rigid_body_material =",
                                            "randomize_robot_material =",
                                            "randomize_robot_external_force_torque =",
                                            "apply_external_force_torque ="))]
        self.assertEqual(active, [], f"这两项刻意不恢复，不应注册：{active}")
        # 两条"为什么不恢复"的理由必须留在源码里（换行会断开，所以只查关键词）
        self.assertIn("**每回合**都会把 robot 与", cfg)
        self.assertIn("**每个物理步**都调用", cfg)
        self.assertIn("robot_domain_rand: UpperRobotDomainRandCfg = UpperRobotDomainRandCfg()", cfg)

    def test_feet_slide_matches_the_ppo_rough_criterion(self):
        """支撑脚不许打滑：与 PPO rough 的 `feet_slide` 同式（Isaac Lab 正本），权重 −0.05。

        正本：`Σ_feet ‖v_foot,xy‖ · 1(net_forces_w_history 历史最大合力 > 1 N)`，
        足端正则 `.*_FOOT`。两处必须一致才能把 PPO 的干净步态结论搬过来。
        另外本仓库改成按**传感器** `body_names` 映射到资产 body id，不依赖正本那种
        "传感器与资产按下标同序"的隐含假设——配错会静默把 A 脚的速度算到 B 脚上。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        # 场景：足端接触传感器（传感器自身的 prim_path 允许多个 prim；filter 才受"每项一个"约束）
        self.assertIn('prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT"', cfg)
        self.assertIn("history_length=3,", cfg)
        self.assertIn("track_air_time=True", cfg)
        self.assertIn("foot_contacts = ContactSensorCfg(", cfg)
        # 奖励：与正本同式
        self.assertIn("def feet_slide(env, sensor_name, contact_threshold=1.0):", mdp)
        self.assertIn("sensor.data.net_forces_w_history.norm(dim=-1).max(dim=1)[0]", mdp)
        self.assertIn("> contact_threshold)", mdp)
        self.assertIn("body_vel = term._asset.data.body_lin_vel_w[:, term._foot_body_ids, :2]", mdp)
        self.assertIn("return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)", mdp)
        self.assertIn("feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.05,", cfg)
        self.assertIn('params={"sensor_name": "foot_contacts"}', cfg)
        # body 顺序按传感器映射（不按下标配对）
        self.assertIn("for name in foot_sensor.body_names", mdp)
        self.assertIn("self._asset.body_names.index(name)", mdp)
        self.assertIn('foot_contact_sensor_name: str = "foot_contacts"', mdp)
        # 传感器缺失/体名不匹配要报错，不能静默
        self.assertIn("场景里没有足端接触传感器", mdp)
        self.assertIn("不在机器人资产里", mdp)
        # 空的 body 列表会静默变成"永远为 0 的假奖励"，必须硬报错
        self.assertIn("应为 4 个（FL/FR/RL/RR_FOOT）", mdp)

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

    def test_min_clearance_reward_uses_spawn_geometry_and_is_gated(self):
        """最小间距奖励：阈值按**出生几何 − 绝对死区**（`出生间隙 − deadband_m`）给出，且对无小车环境屏蔽。

        2026-09-23 用户要求「维持小车与机器人距离不低于连接长度的 ratio 倍」；2026-10-10
        用户批准改口径：**不再用 `ratio × 连接长度`**，先落成 `spawn_margin(0.85) × 出生间隙`，
        **同日再微调成绝对死区** `出生间隙 − deadband_m(0.02)`——理由：①语义就是用户要的
        「不许比出生时更近」，比 0.85 相对余量更贴；②解析出生间隙与仿真实际间隙有**几毫米**差
        （落地/穿透/初始沉降）⇒ 2 cm 绝对值即可吸收，而相对余量对长绳行会浪费几十厘米；
        ③出生点在 lane 的**后向平段**（坡从 +2.25 m 才起）⇒ 出生/停车都在平地、无重力驱动 ⇒
        **没有被动的间隙漂移**，小死区就够。权重保持 **−5.0**（每步代价量级不变）。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("def min_clearance_violation(env, deadband_m, softness=0.02):", mdp)
        # 阈值必须是「出生间隙 − 绝对死区」（逐 env 缓存的 `term.spawn_clearance`）
        self.assertIn("threshold = term.spawn_clearance - deadband_m", mdp)
        # 出生间隙的构造：目标三维挂点距来自 `connection_grid`，水平分量走勾股解，
        # 再减去四个表面常量派生的固定偏移（全部从 cfg 取，不手抄数字）
        self.assertIn("self.spawn_clearance = (", mdp)
        self.assertIn("attachment_horizontal_gap(self.initial_distance[:, 0],", mdp)
        self.assertIn("- cfg.robot_rear_surface_x - cfg.robot_attachment[0]", mdp)
        self.assertIn("+ cfg.cart_attachment[0] - cfg.cart_front_surface_x)", mdp)
        # 无小车环境必须屏蔽（与 clearance/collision 同一约定）
        self.assertIn("return violation * term.cart_present[:, 0]", mdp)
        # 新权重与 params；旧口径不得回潮
        self.assertIn("min_clearance = RewTerm(func=mdp.min_clearance_violation, weight=-5.0", cfg)
        self.assertIn('params={"deadband_m": 0.02, "softness": 0.02})', cfg)
        self.assertNotIn('"spawn_margin"', cfg)
        self.assertNotIn('"ratio": 0.25', cfg)
        self.assertNotIn("min_clearance = RewTerm(func=mdp.min_clearance_violation, weight=-2.0", cfg)
        self.assertNotIn("threshold = spawn_margin * term.spawn_clearance", mdp)
        self.assertNotIn("threshold = ratio * term.connection_length", mdp)
        self.assertNotIn("def min_clearance_violation(env, spawn_margin", mdp)
        # 三条改口径的理由（语义 / 几毫米解析差 / 后向平段无被动漂移）必须写进 docstring
        self.assertIn("不许比出生时更近", mdp)
        self.assertIn("几毫米", mdp)
        self.assertIn("后向平段", mdp)
        self.assertIn("没有重力驱动", mdp)
        self.assertIn("坡上会溜车所以要留大余量", mdp)
        self.assertIn("`min_clearance` 在 spawn 精确为 0", mdp)
        # "高于阈值精确为 0" 的既有实现方式必须保留：带截断的 softplus（减去 softplus(0) 的偏置），
        # 不允许改成会在阈值上方留尾巴的裸 softplus
        self.assertIn("bias = 0.6931471805599453 * softness", mdp)
        self.assertIn("torch.nn.functional.softplus(gap / softness) * softness - bias", mdp)
        # 停车段天然生效这一新事实 + 备选方案（gap_at_stop）必须写进 docstring
        self.assertIn("0.64 · L", mdp)
        self.assertIn("gap_at_stop", mdp)
        self.assertIn("不需要", mdp)
        # 权重标定依据必须写进 docstring / cfg 注释
        self.assertIn("−0.05/步", mdp)
        self.assertIn("−0.02/步（满额的 40%）", cfg)
        # 语义（绳行触发 / 杆行永不触发）也必须写清
        self.assertIn("杆行", mdp)
        self.assertIn("永不触发", mdp)

    @unittest.skipIf(torch is None, "PyTorch is not installed in the offline-check interpreter")
    def test_min_clearance_deadband_positive_and_spawn_value_is_exactly_zero(self):
        """① `deadband_m > 0`；② 阈值严格小于出生间隙 ⇒ **spawn 处 func 精确为 0**，且余量恰为 `deadband_m`。

        本测试**真的把源码里的 `min_clearance_violation` 抽出来在 stub term 上实跑**（不是复述公式）：
        在 `clearance == spawn_clearance`（出生那一拍）给 0.0，在 `clearance == spawn − deadband_m`
        （阈值正好落在间隙上）也给 0.0（无 softplus 尾巴），再低 1 µm 才 > 0 —— 这同时钉住
        「余量恰等于 `deadband_m`」与「阈值上方精确为 0」两条。
        """
        deadband_m, softness, delta_z, base_offset = _spawn_gap_constants()
        self.assertAlmostEqual(deadband_m, 0.02, places=12)
        self.assertGreater(deadband_m, 0.0, "deadband_m 必须 > 0（绝对死区为正）")
        self.assertAlmostEqual(softness, 0.02, places=12)
        self.assertGreater(softness, 0.0)
        for model_name in grid.LENGTH_RANGES:
            for row in range(grid.ROWS):
                length = grid.row_length(row, model_name)
                gap = _spawn_gap(model_name, length, delta_z, base_offset)
                # ①+② 余量恰为 deadband_m，且阈值严格小于出生间隙
                self.assertAlmostEqual(gap - (gap - deadband_m), deadband_m, places=12)
                self.assertLess(gap - deadband_m, gap)
                # ② spawn 那一拍：真实函数给出**精确 0**
                self.assertEqual(
                    _evaluate_min_clearance(gap, gap, deadband_m, softness), 0.0,
                    f"{model_name} row {row} 在 spawn 处非 0（应精确为 0）")
                # 阈值正好落在间隙上（用函数内同一步 float32 运算取边界）：仍然精确 0
                # —— 截断 softplus 的偏置修正生效，阈值上方没有尾巴
                boundary = _float32_threshold(gap, deadband_m)
                self.assertEqual(
                    _evaluate_min_clearance(boundary, gap, deadband_m, softness), 0.0,
                    f"{model_name} row {row} 在阈值处非 0")
                # 再近 1 µm 就严格为正（说明阈值真的画在 spawn − deadband_m，不偏不倚）
                self.assertGreater(
                    _evaluate_min_clearance(boundary - 1e-6, gap, deadband_m, softness), 0.0)
        # 无小车环境：即使深违规也必须被 `cart_present` 屏蔽为 0
        self.assertEqual(_evaluate_min_clearance(0.0, 0.3646, deadband_m, softness, False), 0.0)

    def test_min_clearance_threshold_formula_matches_the_grid_row_by_row(self):
        """③ 阈值公式逐类型逐行核对：`出生间隙 − deadband_m`，绳 0.689–0.780·L、杆 0.905–0.968·L。

        本测试只用仓库里既有的常量与网格（`connection_grid` + 源码 regex）复算，不重复硬编码；
        若日后改 `SLACK_RATIO` / 长度区间 / 挂点常量 / 死区，这里的数值会自动跟着变，
        只有量级断言会提醒。⑥ 同时断言**阈值恒 > 0**（最短绳行 0.3646 − 0.02 = 0.3446 m）。
        """
        deadband_m, softness, delta_z, base_offset = _spawn_gap_constants()
        self.assertAlmostEqual(deadband_m, 0.02, places=12)
        self.assertAlmostEqual(softness, 0.02, places=12)
        self.assertAlmostEqual(delta_z, 0.17, places=6)
        self.assertAlmostEqual(base_offset, 0.0025, places=6)
        # ③ 逐行：绳行阈值 = 出生间隙 − 0.02（= 0.689–0.780·L）；杆行 = 0.905–0.968·L
        for model_name, lo_ratio, hi_ratio in (("compliant", 0.68, 0.79),
                                               ("inextensible", 0.68, 0.79),
                                               ("rigid", 0.90, 0.97)):
            ratios = []
            for row in range(grid.ROWS):
                length = grid.row_length(row, model_name)
                gap = _spawn_gap(model_name, length, delta_z, base_offset)
                threshold = gap - deadband_m
                ratios.append(threshold / length)
                self.assertAlmostEqual(threshold, gap - 0.02, places=12,
                                       msg=f"{model_name} row {row} 阈值 ≠ 出生间隙 − 0.02")
                self.assertLess(threshold, gap, f"{model_name} row {row} 阈值未低于出生间隙")
                # ⑥ 阈值恒 > 0（最短绳行 0.3646 − 0.02 = 0.3446）
                self.assertGreater(threshold, 0.0)
            self.assertGreater(min(ratios), lo_ratio, f"{model_name} 阈值比低于预期区间")
            self.assertLess(max(ratios), hi_ratio, f"{model_name} 阈值比高于预期区间")
            # 端点：绳 L=0.5 → 0.3446 m、L=1.5 → 1.1704 m；杆 L=0.5 → 0.4527 m、L=1.0 → 0.9679 m
            if model_name == "compliant":
                self.assertAlmostEqual(
                    _spawn_gap("compliant", grid.ROPE_LENGTH_MIN_M, delta_z, base_offset) - deadband_m,
                    0.3446, places=4)
                self.assertAlmostEqual(
                    _spawn_gap("compliant", grid.ROPE_LENGTH_MAX_M, delta_z, base_offset) - deadband_m,
                    1.1704, places=4)
            if model_name == "rigid":
                self.assertAlmostEqual(
                    _spawn_gap("rigid", grid.RIGID_LENGTH_MIN_M, delta_z, base_offset) - deadband_m,
                    0.4527, places=4)
                self.assertAlmostEqual(
                    _spawn_gap("rigid", grid.RIGID_LENGTH_MAX_M, delta_z, base_offset) - deadband_m,
                    0.9679, places=4)

    def test_min_clearance_doc_table_matches_the_source_constants(self):
        """**文档表格与源码逐项一致**：`docs/towing_reward_retune_2026-10-10.md` §2.2 的
        两张逐行阈值表（绳/杆各 20 行）必须能用仓库常量复算到 4 位小数。

        文档漂移是这一仓库的既有风险（DOC-01），奖励口径又刚改过两次，所以把表格钉进测试：
        日后改 `SLACK_RATIO` / 长度区间 / 挂点常量 / `deadband_m`，要么同步改表，要么这条测试先红。
        """
        doc = (RL.parent / "docs/towing_reward_retune_2026-10-10.md").read_text("utf-8")
        deadband_m, _softness, delta_z, base_offset = _spawn_gap_constants()
        row_pattern = re.compile(
            r"^\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|$")
        current = None
        seen = {}
        for line in doc.splitlines():
            if "绳行" in line and "compliant" in line:
                current = "compliant"
                continue
            if "杆行" in line and "rigid" in line:
                current = "rigid"
                continue
            if current is None:
                continue
            match = row_pattern.match(line.strip())
            if match is None:
                continue
            row_index = int(match.group(1))
            length_doc, gap_doc, threshold_doc, ratio_doc = (float(match.group(i))
                                                             for i in range(2, 6))
            length = grid.row_length(row_index, current)
            gap = _spawn_gap(current, length, delta_z, base_offset)
            threshold = gap - deadband_m
            self.assertAlmostEqual(length_doc, length, places=4,
                                   msg=f"{current} row {row_index} 文档 L 与网格不一致")
            self.assertAlmostEqual(gap_doc, gap, places=4,
                                   msg=f"{current} row {row_index} 文档出生间隙与源码不一致")
            self.assertAlmostEqual(threshold_doc, threshold, places=4,
                                   msg=f"{current} row {row_index} 文档阈值与源码不一致")
            self.assertAlmostEqual(ratio_doc, threshold / length, places=4,
                                   msg=f"{current} row {row_index} 文档阈值/L 与源码不一致")
            seen.setdefault(current, []).append(row_index)
        for model_name in ("compliant", "rigid"):
            self.assertEqual(seen.get(model_name), list(range(grid.ROWS)),
                             f"{model_name} 的文档表未覆盖全部 {grid.ROWS} 行")

    def test_min_clearance_is_inactive_at_spawn_for_every_grid_row(self):
        """意图守卫：**所有行、所有连接类型**的 spawn 间隙都必须高于 `min_clearance` 阈值。

        用户 2026-09-23 明确要求「初始的时候这个奖励不生效」。2026-10-10 阈值改成
        `出生间隙 − deadband_m` 后，这一条**构造上成立**（死区 > 0 ⇒ 阈值 < 出生间隙）；
        本测试把它逐行逐类型再钉一遍，并断言 `deadband_m > 0`（若日后有人把死区改成 0 或负数，
        出生瞬间缺口会在浮点/沉降下变成非零，破坏该不变量）。
        """
        deadband_m, _softness, delta_z, base_offset = _spawn_gap_constants()
        self.assertGreater(deadband_m, 0.0,
                           "deadband_m 必须 > 0，否则出生瞬间缺口恰为 0，任何沉降/微动都会误罚")
        for model_name in grid.LENGTH_RANGES:
            for row in range(grid.ROWS):
                length = grid.row_length(row, model_name)
                gap = _spawn_gap(model_name, length, delta_z, base_offset)
                threshold = gap - deadband_m
                self.assertGreater(
                    gap, threshold,
                    f"{model_name} row {row}（L={length:.4f}）spawn 间隙 {gap:.4f} m "
                    f"未高于阈值 {threshold:.4f} m；用户要求 spawn 时该奖励不生效")
                # spawn 时 func = threshold − clearance = gap − threshold = deadband_m > 0
                # ⇒ 铰链（含 softplus 偏置修正）**精确为 0**
                self.assertGreater(gap - threshold, 0.0)
                self.assertAlmostEqual(gap - threshold, deadband_m, places=12)

    def test_min_clearance_threshold_increases_with_length(self):
        """④ 阈值随 L 单调递增（每类型独立核对）。

        这是「绳行的阈值 ≈ 0.689–0.780·L、杆行永不触发」两条结论的前提：出生间隙随 L 增大
        （`sqrt((r·L)² − Δz²)` 在 L > Δz/r 后单调增），阈值 = 它 − 常数。
        """
        deadband_m, _softness, delta_z, base_offset = _spawn_gap_constants()
        for model_name in grid.LENGTH_RANGES:
            lengths = [grid.row_length(row, model_name) for row in range(grid.ROWS)]
            thresholds = [_spawn_gap(model_name, length, delta_z, base_offset) - deadband_m
                          for length in lengths]
            for lower, upper in zip(thresholds, thresholds[1:]):
                self.assertLess(lower, upper, f"{model_name} 阈值未随 L 单调递增")
            # 端点严格递增（20 档等距）
            self.assertLess(thresholds[0], thresholds[-1])

    def test_min_clearance_never_fires_for_rigid_rows(self):
        """⑤ **杆行永不触发**：连杆把三维挂点距固定在 L ⇒ 在出生高差上间隙恒 = 出生间隙 > 阈值。

        论证分三层：

        1. **恒等式**：连杆约束 `|p_robot_attach − p_cart_attach| = L`（`RigidLink` 的双边约束，
           已由 `test_towing_rope_models` 等钉住），且 Δz = 出生值 0.17 m 时实际间隙 =
           `sqrt(L² − Δz²) + base_offset` = 出生间隙 ⇒ 缺口 = `deadband_m = 0.02 m` > 0
           ⇒ `func` 精确为 0（逐行断言）；
        2. **Δz 扫描**：间隙随 Δz 单调减。在出生值 ±0.04 m（即 `Δz ∈ [0.13, 0.21]`，已宽于
           步态里 base 高度的变化）上全部 20 行的间隙仍高于阈值，最小余量 ≈ +0.0036 m
           （出现在最短杆 L=0.5、Δz=0.21）；
        3. **触发需要多大的 Δz**：解 `sqrt(L²−Δz²)+base_offset = 出生间隙 − deadband_m` 得
           `Δz*(L=0.5) = 0.2175 m` … `Δz*(L=1.0) = 0.2606 m`，即最短杆要 Δz 比出生高差再大
           **+0.0475 m**（杆越长要求越大）。

        **口径变更的代价（必须记住）**：旧的 `spawn_margin = 0.85` 相对余量下，间隙恒比阈值大
        `0.15 × 间隙`（最短杆 +0.071 m）、Δz* 是 0.301/0.547 m（最短杆余量 +0.131 m）；
        改成绝对死区 2 cm 后余量只剩 `deadband_m`，Δz 余量缩到 **+0.0475 m**。杆行"永不触发"
        在物理上仍成立（Δz 由两体的地面接触与腿长决定、刚性连杆还会用压小水平间隙来抵抗 Δz 增大），
        但对 Δz 漂移的稳健性明显变弱——训练机若看到最短杆行 `min_clearance` 非 0，先查 Δz。
        """
        deadband_m, _softness, delta_z, base_offset = _spawn_gap_constants()
        self.assertAlmostEqual(grid.initial_attachment_distance("rigid", 0.75), 0.75, places=12)
        for row in range(grid.ROWS):
            length = grid.row_length(row, "rigid")
            gap = _spawn_gap("rigid", length, delta_z, base_offset)
            threshold = gap - deadband_m
            # 1) 恒等式：连杆下间隙 == 出生间隙 ⇒ 缺口恒为 deadband_m > 0
            self.assertAlmostEqual(gap - threshold, deadband_m, places=12)
            self.assertGreater(gap - threshold, 0.0)
            # 2) Δz 扫描：出生值 ±0.04 m（远宽于步态里 base 高度的变化）
            for step in range(9):
                swept = delta_z - 0.04 + 0.08 * step / 8.0
                if swept >= length:
                    continue
                swept_gap = _spawn_gap("rigid", length, swept, base_offset)
                self.assertGreater(swept_gap, threshold,
                                   f"rigid row {row}（L={length:.3f}）在 Δz={swept:.3f} 触发")
            # 3) 触发所需的 Δz*（二分）必须比出生高差再大至少 0.045 m（实测 0.0475–0.0906）
            low, high = 0.0, length
            for _ in range(80):
                middle = (low + high) / 2.0
                if _spawn_gap("rigid", length, middle, base_offset) > threshold:
                    low = middle
                else:
                    high = middle
            self.assertGreater(low - delta_z, 0.045,
                               f"rigid row {row}（L={length:.3f}）只需 Δz 再大 "
                               f"{low - delta_z:.3f} m 就触发，余量不足")
            self.assertLess(low - delta_z, 0.10,
                            f"rigid row {row}（L={length:.3f}）Δz 余量 {low - delta_z:.3f} m "
                            f"超出新口径的预期量级（绝对死区下应约 0.05–0.09 m）")
        # L 上界（1.0 m）也核对一次：需要 Δz ≈ 0.26 m 才触发，仍不可达
        self.assertGreater(
            _spawn_gap("rigid", grid.RIGID_LENGTH_MAX_M, 0.21, base_offset),
            _spawn_gap("rigid", grid.RIGID_LENGTH_MAX_M, delta_z, base_offset) - deadband_m)

    def test_extra_distance_reward_term_is_removed_but_helper_survives(self):
        """2026-10-10：`extra_distance` 奖励项删除，`mdp.post_stop_distance` 函数保留、公式不变。

        删除理由（用户决定，见函数 docstring）：与 `tracking_velocity` 在停车段高度冗余
        （`vx→0` 已隐含位移只剩不可避免的惯性滑行），且量级小 20 倍（每步 −0.0025 vs +0.050）。
        本测试用 **AST** 判「Reward 配置里没有这个项」（源码注释里可以提到它，字符串断言会被
        注释误伤），并真的把函数体从源码里抽出来跑一遍，钉住 `relu(x − x_stop − allowance)`。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        # ① 奖励表里没有 `extra_distance`（AST：类体里的赋值目标）
        tree = ast.parse(cfg)
        rewards = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "UpperRewardsCfg")
        assigned = {target.id
                    for node in rewards.body if isinstance(node, ast.Assign)
                    for target in node.targets if isinstance(target, ast.Name)}
        self.assertNotIn("extra_distance", assigned)
        self.assertIn("min_clearance", assigned)
        self.assertIn("stop_towing_force", assigned)
        # 也不得以别的名字把 post_stop_distance 注册回奖励表
        self.assertNotIn("func=mdp.post_stop_distance", cfg)
        # ② 函数本体与形参保留（`mdp.post_stop_distance` 仍可导入、签名不变）
        mdptree = ast.parse(mdp)
        helper = next(node for node in mdptree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "post_stop_distance")
        self.assertEqual([arg.arg for arg in helper.args.args], ["env", "post_stop_allowance_m"])
        self.assertEqual(len(helper.args.defaults), 1)
        self.assertIsInstance(helper.args.defaults[0], ast.Constant)
        self.assertEqual(helper.args.defaults[0].value, 0.0)
        self.assertIn("def post_stop_towing_force(env, force_scale):", mdp)
        self.assertIn("已不作为奖励项", mdp)
        self.assertIn("目前未接入奖励", cfg)
        # ③ 公式不变：抽出函数体在 stub term 上实跑（`relu(x − x_stop − allowance)`）
        if torch is None:
            self.skipTest("PyTorch is not installed in the offline-check interpreter")
        namespace = {"torch": torch, "_term": lambda env: env.term}
        exec(textwrap.dedent(ast.get_source_segment(mdp, helper)), namespace)
        function = namespace["post_stop_distance"]

        class _Data:
            # env0：停车后多走 0.3 m；env1：还没到停车时刻（`stop_time_s = +inf`）
            root_pos_w = torch.tensor([[1.3], [1.0]])

        class _Asset:
            data = _Data()

        class _Term:
            stop_time_s = torch.tensor([0.0, float("inf")])
            stop_origin_x = torch.tensor([1.0, 1.0])
            _asset = _Asset()

        class _Env:
            episode_length_buf = torch.tensor([20, 20])
            step_dt = 0.05
            term = _Term()

        moved = float(function(_Env())[0])
        self.assertAlmostEqual(moved, 0.3, places=6)        # 1.3 − 1.0 − 0.0
        self.assertAlmostEqual(float(function(_Env())[1]), 0.0, places=12)  # 未停车 ⇒ 精确 0
        allowed = float(function(_Env(), post_stop_allowance_m=0.1)[0])
        self.assertAlmostEqual(allowed, 0.2, places=6)      # 1.3 − 1.0 − 0.1

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
        """跟踪项必须比**实测速度 vs 任务指令**，且 `reference_tracking` 随残差方案彻底删除。

        2026-09-23 曾把线性误差写成 `reference_command − user_command`（上层自己的积分指令
        vs 任务指令），这与 `velocity_tracking_exp` 的名字和奖励文档（"实际速度 vs 命令期望"）
        都不符，而且和 `reference_tracking_l2` 重复。2026-10-08 动作改成关节残差后
        `reference_command` 不存在，该式若保留会恒为 0（`exp(0)=1` 变成白送的正奖励）。

        2026-10-10 双头迁移后参考量进一步收紧为 **`task_command`**（脚本调度、不含策略偏移），
        见 `test_reward_terms_never_read_loco_command`。
        """
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        # 跟踪项：实测机体系**纵向**线速度 vs 任务指令（2026-10-09 起横向/朝向由 PD 负责，
        # 所以跟踪误差只留 vx，`use_lateral_and_heading` 默认 False）
        self.assertIn(
            "forward_error = ((term._asset.data.root_lin_vel_b[:, 0] - term.task_command[:, 0])", mdp)
        self.assertIn("if not use_lateral_and_heading:", mdp)
        self.assertIn("return torch.exp(-forward_error.square())", mdp)
        self.assertIn("use_lateral_and_heading=False", cfg)
        # 恢复旧口径的分支仍在（含 vy 与 yaw 角速度），但只能是显式打开
        self.assertIn("term._asset.data.root_lin_vel_b[:, 1] - term.task_command[:, 1]", mdp)
        self.assertIn("term._asset.data.root_ang_vel_b[:, 2] - term.task_command[:, 2]", mdp)
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

    def test_reward_terms_never_read_loco_command(self):
        """**防作弊红线（AST 守卫）**：奖励函数一律不得读 `loco_command`。

        `loco_command` = 任务指令 + 策略自己的 1 维 vx 偏移（送冻结策略、进 actor 帧）。
        只要**任何**奖励项读了它，策略就能靠"把偏移开大"自己给自己发目标、白拿跟踪奖励
        （`tracking_velocity` 尤其致命：把参考量抬到实测速度就恒等于 1）。

        实现：从 `upper_env_cfg.UpperRewardsCfg` 解析出所有 `RewTerm(func=mdp.X)` 的 X，
        再到 `upper_mdp` 里取这些函数的 AST，禁止出现 `.loco_command` 属性访问；
        `velocity_tracking_exp` 还必须显式读 `task_command`。
        """
        cfg_tree = ast.parse((PKG / "upper_env_cfg.py").read_text("utf-8"))
        mdp_tree = ast.parse((PKG / "upper_mdp.py").read_text("utf-8"))
        funcs = {node.name: node for node in ast.walk(mdp_tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        reward_cfg = next(node for node in ast.walk(cfg_tree)
                          if isinstance(node, ast.ClassDef) and node.name == "UpperRewardsCfg")
        reward_funcs = []
        for node in reward_cfg.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            for keyword in node.value.keywords:
                if (keyword.arg == "func" and isinstance(keyword.value, ast.Attribute)
                        and isinstance(keyword.value.value, ast.Name)
                        and keyword.value.value.id == "mdp"):
                    reward_funcs.append(keyword.value.attr)
        self.assertGreater(len(reward_funcs), 5, f"没解析出奖励函数：{reward_funcs}")
        self.assertIn("velocity_tracking_exp", reward_funcs)
        offenders = []
        for name in reward_funcs:
            self.assertIn(name, funcs, f"奖励项 mdp.{name} 在 upper_mdp 里找不到")
            for sub in ast.walk(funcs[name]):
                if (isinstance(sub, ast.Attribute) and sub.attr == "loco_command"):
                    offenders.append(f"{name}:{sub.lineno}")
                if (isinstance(sub, ast.Name) and sub.id == "loco_command"):
                    offenders.append(f"{name}:{sub.lineno}")
        self.assertEqual(offenders, [], f"奖励函数不得读 loco_command（防作弊）：{offenders}")
        # 跟踪项必须显式读任务指令
        tracking = funcs["velocity_tracking_exp"]
        reads_task = any(isinstance(sub, ast.Attribute) and sub.attr == "task_command"
                         for sub in ast.walk(tracking))
        self.assertTrue(reads_task, "velocity_tracking_exp 必须读 task_command 当参考量")

    def test_task_and_loco_command_are_decoupled_in_the_action_term(self):
        """两个命令必须**分成两个张量**，且偏移只在 `elapsed_s >= tow_start_s` 后生效。

        源码级守卫（本机没有 Isaac Lab，跑不了 `process_actions`）：

        - `task_command`（脚本，只给奖励）与 `loco_command`（送策略）是两份状态；
        - `loco_command` 由 `task_command` 拷贝 + **有界**偏移组成；
        - 偏移读取的是**本拍** `_processed`（顺序上 `_processed` 必须先更新）；
        - 门控是 `elapsed_s >= self.tow_start_s`（出生段不许推机器人），且 STOP 之后
          偏移仍然生效（`_was_stopped` 不参与门控）——这正是「停机后继续走两步」的口。
        """
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        body = mdp[mdp.index("    def process_actions(self, actions):"):]
        body = body[:body.index("\n    def ")]
        # 两份状态
        self.assertIn("self.task_command = torch.zeros(env.num_envs, 3", mdp)
        self.assertIn("self.loco_command = torch.zeros(env.num_envs, 3", mdp)
        self.assertIn("self.task_command.zero_()", body)
        self.assertIn("self.loco_command.copy_(self.task_command)", body)
        # 偏移：有界 + 只加在 vx 通道
        self.assertIn("offset = self._processed[:, :CMD_ACTION_DIM] * self.cfg.cmd_offset_scale",
                      body)
        self.assertIn("offset = offset.clamp(self.cfg.offset_min, self.cfg.offset_max)", body)
        self.assertIn("self.loco_command[:, :CMD_ACTION_DIM] += torch.where(", body)
        # 出生段门控（不是 STOP 门控）
        self.assertIn("offset_active = (elapsed_s >= self.tow_start_s).unsqueeze(1)", body)
        self.assertIn("torch.zeros_like(offset))", body)
        # 顺序：`_processed` 必须先于偏移合成被更新（否则偏移用的是上一拍动作）
        self.assertLess(body.index("self._processed.copy_(actions.clamp(-1.0, 1.0))"),
                        body.index("offset = self._processed[:, :CMD_ACTION_DIM]"))
        # 关节残差只吃后 12 维
        self.assertIn("self._processed[:, CMD_ACTION_DIM:] * self._residual_scale", body)
        # cfg 字段齐备且默认值 = 用户建议值 / 关闭
        for fragment in ("cmd_offset_scale: float = 0.5", "offset_min: float = -0.2",
                         "offset_max: float = 0.6", "stop_command_ramp_s: float = 0.0"):
            self.assertIn(fragment, mdp)
        # 偏移头在 `UpperActionsCfg` 里没有第二份默认值（唯一来源是 action term cfg）
        self.assertNotIn("cmd_offset_scale", cfg)

    def test_two_new_knobs_default_off_so_v3_reduces_to_v2_behaviour(self):
        """**退化性（源码级）**：新开关默认关闭 ⇒ 行为与单头口径逐位一致。

        本轮用户明确要求「只加字段、不偷偷改语义」：`stop_command_ramp_s = 0.0` 必须是默认值，
        且**只有**显式 >0 时才走新分支。同批落地的 `post_stop_allowance_m`（默认 0.0）在
        2026-10-10 随 `extra_distance` 一起从奖励表移除，现在**只保留为
        `mdp.post_stop_distance` 的形参**（未接入奖励，见
        `test_extra_distance_reward_term_is_removed_but_helper_survives`）。
        纯函数层面的逐位对照见 `test_command_offset_is_bounded_additive_and_gated_off_at_spawn`
        与 `test_scheduled_vx_matches_the_old_schedule_when_ramp_is_off`；这条守的是训练侧
        `process_actions` 真的按这个门控执行（ramp 代码块在 `> 0.0` 里，不能被无条件执行）。
        """
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        cfg = (PKG / "upper_env_cfg.py").read_text("utf-8")
        body = mdp[mdp.index("    def process_actions(self, actions):"):]
        body = body[:body.index("\n    def ")]
        # ramp 只在 > 0 时启用（默认 0.0 时整段不执行 ⇒ 与旧口径一致）
        self.assertIn("if self.cfg.stop_command_ramp_s > 0.0:", body)
        self.assertIn("stop_command_ramp_s: float = 0.0", mdp)
        # 旧口径的脚本项原样保留（settle 0 → tow speed → STOP 0）
        self.assertIn(
            "scripted_vx = torch.where(towing, self.tow_speed, torch.zeros_like(self.tow_speed))",
            body)
        # STOP 之后偏移头仍然生效（`_was_stopped` 不进偏移门控）——这是「停机续走」的表达口
        self.assertNotIn("offset_active = towing", body)
        self.assertNotIn("offset_active = (elapsed_s >= self.tow_start_s) & ~self._was_stopped",
                         body)
        # allowance 默认 0.0 只留在函数形参上（不是 cfg 默认值、也不再是奖励 params）
        self.assertNotIn('params={"post_stop_allowance_m": 0.0}', cfg)
        self.assertNotIn("post_stop_allowance_m: float", cfg)
        self.assertIn("def post_stop_distance(env, post_stop_allowance_m=0.0):", mdp)
        # `__post_init__` 必须断言 AMP 包络覆盖脚本速度范围（否则零偏移退化性失效）
        self.assertIn("amp_vx_range {(amp_vx_min, amp_vx_max)} 必须覆盖脚本速度范围", cfg)

    def test_amp_envelope_clamp_is_the_last_step_in_process_actions(self):
        """源码级：裁剪的顺序必须是「先限偏移、再裁和进 AMP 包络」。

        若顺序反了（先裁和到包络、再叠偏移），叠完的 vx 会重新越界；若用偏移头的
        `[offset_min, offset_max]` 去裁和，牵引段（0.5–1.5）会被砍成 ≤0.6。
        """
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        body = mdp[mdp.index("    def process_actions(self, actions):"):]
        body = body[:body.index("\n    def ")]
        offset_clamp = body.index("offset = offset.clamp(self.cfg.offset_min, self.cfg.offset_max)")
        envelope = body.index("amp_vx_min, amp_vx_max = self.cfg.amp_vx_range")
        final_clamp = body.index("clamp(\n            amp_vx_min, amp_vx_max)")
        self.assertLess(offset_clamp, envelope)
        self.assertLess(envelope, final_clamp)
        # 不得拿偏移头的界限去裁 `loco_command`
        self.assertNotIn("self.loco_command[:, :CMD_ACTION_DIM].clamp(\n            "
                         "self.cfg.offset_min", body)

    def test_reset_clears_the_task_command_too(self):
        """复位必须同时清 `task_command` 与 `loco_command`（漏一个会让下一回合带着旧指令）。"""
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        body = mdp[mdp.index("    def reset(self, env_ids=None):"):]
        body = body[:body.index("\n\n@configclass")]
        self.assertIn("self.task_command[env_ids] = 0", body)
        self.assertIn("self.loco_command[env_ids] = 0", body)

    def test_upper_action_is_applied_as_a_residual_on_frozen_joint_targets(self):
        """残差必须加在**冻结策略的关节位置目标**上，且不能污染冻结策略自己的观测。

        四件事一起守：
        1. 动作总维数 = 1（vx 偏移头）+ 冻结策略契约的关节数（12），不是写死的 3；
        2. 每次底层刷新都用当前 `delta_joint_pos` 重算 `joint_targets + 残差`
           （残差 50 ms 变、底层输出 20 ms 变，只在 50 ms 处算一次会用错值）；
        3. `FrozenLowLevelPolicy` 的输入仍只有 `loco_command` 与本体状态——残差若进了
           它自己的 45 维观测（尤其 `last_action`），冻结契约就被改写成另一个策略了；
        4. 残差只取 13 维动作的**后 12 维**（前 1 维是 vx 偏移，不许混进关节）。
        """
        mdp = (PKG / "upper_mdp.py").read_text("utf-8")
        self.assertIn("return CMD_ACTION_DIM + self._joint_action_dim", mdp)
        self.assertIn("self._joint_action_dim = len(self._policy_cfg.joint_names)", mdp)
        self.assertIn("residual_scale=tuple(self._policy_cfg.action_scale)", mdp)
        self.assertIn("self._processed[:, CMD_ACTION_DIM:] * self._residual_scale", mdp)
        self.assertIn("output.joint_targets + self.delta_joint_pos", mdp)
        self.assertIn("velocity_command=self.loco_command", mdp)
        # **两层串联的顺序**：偏移进的是底层**输入**（`velocity_command`，推理之前），
        # 残差加的是底层**输出**（`joint_targets`，推理之后）。顺序反了就等于把偏移变成
        # 输出侧扰动，与训练/部署契约都不符（部署待办同此顺序要求，见落地记录 §6）。
        apply_actions = mdp[mdp.index("    def apply_actions(self):"):]
        apply_actions = apply_actions[:apply_actions.index("\n    @staticmethod")]
        self.assertLess(apply_actions.index("velocity_command=self.loco_command"),
                        apply_actions.index("output.joint_targets + self.delta_joint_pos"))
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
        # `TowingDynamicsDecoder()` 的默认 frame_dim 也不能退回旧契约（不传参会静默建错网络）
        default_frame = re.search(r"def __init__\(self, frame_dim=(\d+)", decoder)
        self.assertIsNotNone(default_frame, "找不到 decoder 的 frame_dim 默认值")
        self.assertEqual(int(default_frame.group(1)), obs.frame_dim)
        self.assertEqual(dec.dim, 6)
        self.assertEqual(obs.frame_dim, 58)
        self.assertEqual(obs.actor_dim, obs.frame_dim + dec.dim + obs.latent_dim)
        self.assertEqual(obs.actor_dim, 80)
        # 训练侧 runner 的契约版本 = 运行时 = 3（v2 的 57/79 已失效）
        self.assertIn("TOWING_CONTRACT_VERSION = 3", runner)
        self.assertIn('"version": TOWING_CONTRACT_VERSION', runner)
        runtime_src = (RL / "scripts/towing/upper_policy_runtime.py").read_text("utf-8")
        self.assertIn("CHECKPOINT_CONTRACT_VERSION = 3", runtime_src)

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
            '"mass_range": (5.0, 20.0)',
            # 出生**固定**（用户 2026-10-09）：三个抖动必须都是 (0, 0)
            '"robot_x_range": (0.0, 0.0)',
            '"robot_y_range": (0.0, 0.0)',
            '"robot_yaw_range": (0.0, 0.0)',
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
        停车段奖励因此重新有相位可用（当时是两条；2026-10-10 起**只剩
        `stop_towing_force`**，`extra_distance` 被用户删除、函数本体保留）。

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
        # post_stop 奖励只剩 `stop_towing_force`（权重与历史一致）；
        # `extra_distance`（`post_stop_distance`，−0.1）已于 2026-10-10 删除，函数本体保留
        self.assertIn("stop_towing_force = RewTerm(func=mdp.post_stop_towing_force, weight=-1.0", cfg)
        self.assertNotIn("extra_distance = RewTerm", cfg)
        self.assertNotIn('params={"post_stop_allowance_m": 0.0}', cfg)
        self.assertIn("def post_stop_distance(env, post_stop_allowance_m=0.0):", mdp)
        self.assertIn('params={"force_scale": 10.0}', cfg)
        # 相位与状态：触发时记录 stop_time_s / stop_origin_x，复位回 +inf
        self.assertIn("def post_stop_towing_force(env, force_scale):", mdp)
        # 公式（保留在函数里，供复用）：`relu(x − x_stop − allowance)`；allowance = 0 时退化为历史口径
        self.assertIn(
            "travelled = term._asset.data.root_pos_w[:, 0] - term.stop_origin_x "
            "- post_stop_allowance_m", mdp)
        self.assertIn("return torch.relu(travelled) * post_stop", mdp)
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
        # PD 写的是**任务指令**（`task_command`），然后整体拷进 `loco_command` 再叠偏移：
        # 奖励参考量与送策略的指令因此共享同一份 (vy, wz)，而 vx 只在 loco 侧被偏移改写。
        self.assertIn("self.task_command[:, 1] = self._lane_keeping_vy()", mdp)
        self.assertIn("self.task_command[:, 2] = self._lane_keeping_wz()", mdp)
        self.assertIn("self.loco_command.copy_(self.task_command)", mdp)
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
