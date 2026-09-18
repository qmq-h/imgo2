"""rl_amp（fan-ziqi）配方的离线契约测试（纯标准库，不需要 GPU / Isaac Sim）。

用户 2026-09-18 的决定：粗糙地形版要「忠于 AMP」——按 `rl_amp` 的设置，
**任务奖励只留线速度/角速度跟踪**（其余全 0），AMP 侧比例也照它的 `coef 2.0 / lerp 0.3`。
为让"平地 vs 粗糙"成为单变量对照，同一配方同时做成 `Imgo2-basemove-flat-amp-rlamp` 与
`Imgo2-basemove-rough-amp-rlamp`。

参考数值（`~/Desktop/AMP/rl_amp/legged_gym/legged_gym/envs/a1/a1_amp_config.py`）：
    tracking_lin_vel = 1.5 * 1./(.005*6)  = 50.0     → 每步 50.0 × 0.02 = 1.0
    tracking_ang_vel = 0.5 * 1./(.005*6)  = 16.6667  → 每步 16.6667 × 0.02 = 0.3333
    其余（lin_vel_z/ang_vel_xy/orientation/torques/dof_vel/dof_acc/base_height/
          feet_air_time/collision/feet_stumble/action_rate/stand_still/dof_pos_limits）全 0

Run: python3 -m unittest discover -s tests -p test_amp_rlamp_recipe.py
"""

import ast
import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AMP_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py"
AGENT_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/agents/amp_rsl_rl_cfg.py"
TASKS_INIT = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/__init__.py"
REWARDS_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
RLAMP_CANDIDATES = [
    Path.home() / "Desktop/AMP/rl_amp/legged_gym/legged_gym/envs/a1/a1_amp_config.py",
    Path.home() / "RL/isaac/AMP/rl_amp/legged_gym/legged_gym/envs/a1/a1_amp_config.py",
]
# `legged_robot.py`：reset 根状态/关节、摩擦、质量、推力、噪声的实现
RLAMP_ROBOT_ENV_CANDIDATES = [
    Path.home() / "Desktop/AMP/rl_amp/legged_gym/legged_gym/envs/base/legged_robot.py",
    Path.home() / "RL/isaac/AMP/rl_amp/legged_gym/legged_gym/envs/base/legged_robot.py",
]
# `legged_robot_amp.py`：actor 观测 = `privileged_obs_buf[:, 6:]`（切掉线速度与角速度）
RLAMP_AMP_ENV_CANDIDATES = [
    Path.home() / "Desktop/AMP/rl_amp/legged_gym/legged_gym/envs/base/legged_robot_amp.py",
    Path.home() / "RL/isaac/AMP/rl_amp/legged_gym/legged_gym/envs/base/legged_robot_amp.py",
]
# `normalization` / `obs_scales` / `clip_actions` 定义在基类配置里（a1 没有覆盖）
RLAMP_BASE_CANDIDATES = [
    Path.home() / "Desktop/AMP/rl_amp/legged_gym/legged_gym/envs/base/legged_robot_config.py",
    Path.home() / "RL/isaac/AMP/rl_amp/legged_gym/legged_gym/envs/base/legged_robot_config.py",
]

# rl_amp 里**非零**的跟踪项 → 我们的项名
RLAMP_REFERENCE_MAP = {
    "tracking_lin_vel": "track_lin_vel_xy_exp",
    "tracking_ang_vel": "track_ang_vel_z_exp",
}
# 参考里显式为 0 的项（我们这边必须被清空，不得残留任何权重）
RLAMP_REFERENCE_ZEROED = (
    "lin_vel_z", "ang_vel_xy", "orientation", "torques", "dof_vel", "dof_acc",
    "base_height", "feet_air_time", "collision", "feet_stumble", "action_rate",
    "stand_still", "dof_pos_limits", "termination",
)


def _module(path):
    return ast.parse(Path(path).read_text(encoding="utf-8-sig"))


def _num(node):
    """把「只含数字与四则运算」的 AST 求值（参考里写的是 `1.5 * 1./(.005*6)` 这类表达式）。"""
    return eval(compile(ast.Expression(body=node), "<cfg>", "eval"), {"__builtins__": {}}, {})


def _class(tree, name):
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name]
    assert hits, f"找不到类 {name}"
    return hits[0]


def _func_src(tree, cls_name, fn_name):
    cls = _class(tree, cls_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    return ast.unparse(fn)


def _reward_term_names():
    cfg = _class(_module(REWARDS_CFG), "RewardsCfg")
    names = set()
    for node in cfg.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _class_assigns(cls, *, literal=False):
    """把一个类体里的赋值读成 {名字: 值}（值可以是 AST 节点，或 literal=True 时求值）。

    literal 模式下无法求值的项（例如引用模块级 `MOTION_FILES` 的）会被跳过，而不是报错。
    """
    out = {}
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    if not literal:
                        out[t.id] = node.value
                        continue
                    try:
                        out[t.id] = _num(node.value)
                    except Exception:
                        pass
    return out


def _nested_class(cls, name):
    return next(n for n in cls.body if isinstance(n, ast.ClassDef) and n.name == name)


def _fake_env_cfg(*, reference_init=True):
    """重建 `apply_rlamp_env_settings()` 接触到的那层 cfg 结构（鸭子类型，无需 Isaac Lab）。

    初值抄自**真实的 AMP 配置**：`EventCfg` / `CommandsCfg` 的对齐前取值，外加
    `Imgo2AmpMoveEnvCfg.__post_init__`（`amp_env_cfg.py:149-159`）对 AMP 任务做的覆盖
    —— 其中 `randomize_reset_base` / `randomize_reset_joints` 被设成 **None**。这一点必须照抄：
    2026-09-18 的冒烟训练正是死在这里（helper 对 None 取 `.params`）。
    `reference_init=False` 对应粗糙版（它把参考状态初始化关掉了）。
    """
    from types import SimpleNamespace as NS
    return NS(
        commands=NS(base_velocity=NS(ranges=NS(
            lin_vel_x=(-1.0, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.57, 1.57)))),
        events=NS(
            randomize_rigid_body_material=NS(params={
                "static_friction_range": (0.3, 1.0),
                "dynamic_friction_range": (0.3, 0.8),
                "restitution_range": (0.0, 0.5)}),
            randomize_rigid_body_mass_base=NS(params={"mass_distribution_params": (-1.0, 3.0)}),
            randomize_rigid_body_mass_others=NS(params={"mass_distribution_params": (0.7, 1.3)}),
            randomize_com_positions=NS(params={"com_range": {"x": (-0.05, 0.05)}}),
            randomize_apply_external_force_torque=NS(params={"force_range": (-10.0, 10.0)}),
            randomize_actuator_gains=NS(mode="reset", params={
                "stiffness_distribution_params": (0.5, 2.0),
                "damping_distribution_params": (0.5, 2.0)}),
            randomize_push_robot=NS(mode="interval", interval_range_s=(10.0, 15.0),
                                    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}),
            # AMP 基类把这两项关掉了 ⇒ None（我们要"重建"，不是"改值"）
            randomize_reset_base=None,
            randomize_reset_joints=None,
            reference_state_initialization=(
                NS(params={"reference_state_initialization_prob": 1.0}) if reference_init else None),
        ),
        observations=NS(policy=NS(base_ang_vel=NS(noise="JUNK"), joint_pos=NS(noise="JUNK"))),
    )


def _fake_helper_namespace():
    """helper 依赖的模块级名字（`Unoise` / `EventTerm` / `mdp`）的替身。"""
    from types import SimpleNamespace as NS
    return {
        "Unoise": lambda n_min, n_max: ("Unoise", n_min, n_max),
        "EventTerm": lambda func, mode, params: NS(func=func, mode=mode, params=params),
        "mdp": NS(reset_root_state_uniform="reset_root_state_uniform",
                  reset_joints_by_scale="reset_joints_by_scale"),
    }


class RLAmpRecipeTests(unittest.TestCase):
    def setUp(self):
        self.tree = _module(AMP_CFG)
        self.helper = _func_src(self.tree, "Imgo2AmpRLAmpEnvCfg", "__post_init__")

    def test_only_two_rewards_are_kept(self):
        """只留线/角速度跟踪；其余项不得出现在 keep 名单里。"""
        keep = next(n for n in self.tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "RLAMP_KEEP_REWARDS" for t in n.targets))
        self.assertEqual(ast.literal_eval(keep.value),
                         ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"))

    def test_helper_nulls_everything_else(self):
        """helper 必须遍历清空非保留项，并把两项权重按『原始权重 × dt 换算』写进去。"""
        src = _func_src(self.tree, "_dummy", "_dummy") if False else ast.unparse(
            next(n for n in self.tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "apply_rlamp_task_rewards"))
        self.assertIn("attr not in RLAMP_KEEP_REWARDS", src)
        self.assertIn("setattr(cfg.rewards, attr, None)", src)
        self.assertIn("RLAMP_TRACK_LIN_VEL_RAW * RLAMP_DT / step_dt", src)
        self.assertIn("RLAMP_TRACK_ANG_VEL_RAW * RLAMP_DT / step_dt", src)
        # 不得偷偷加回高度/姿态/步态项
        for forbidden in ("base_height_l2", "lin_vel_z_l2", "ang_vel_xy_l2", "joint_pos_limits",
                          "feet_air_time", "action_rate_l2", "undesired_contacts"):
            self.assertNotIn(f"rewards.{forbidden}", src)

    def test_kept_terms_exist_in_rewards_cfg(self):
        available = _reward_term_names()
        for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
            self.assertIn(name, available)

    def test_raw_weights_match_reference_source(self):
        """本机有参考项目时直接读它的 scales，逐项比对（防止我抄错数）。"""
        ref = next((p for p in RLAMP_CANDIDATES if p.exists()), None)
        if ref is None:
            self.skipTest("本机没有 rl_amp 参考项目，跳过与参考源码的直接比对")
        scales = _class(_module(ref), "scales")
        values = {t.id: _num(node.value) for node in scales.body if isinstance(node, ast.Assign)
                  for t in node.targets if isinstance(t, ast.Name)}
        # 参考里非零的只有两项
        nonzero = {k: v for k, v in values.items() if v != 0}
        self.assertEqual(sorted(nonzero), sorted(RLAMP_REFERENCE_MAP), "参考的非零项变了")
        # 我们的常量必须等于参考表达式算出的值
        consts = {n.targets[0].id: _num(n.value) for n in self.tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id.startswith("RLAMP_")}
        self.assertAlmostEqual(consts["RLAMP_TRACK_LIN_VEL_RAW"], values["tracking_lin_vel"], places=9)
        self.assertAlmostEqual(consts["RLAMP_TRACK_ANG_VEL_RAW"], values["tracking_ang_vel"], places=9)
        self.assertAlmostEqual(consts["RLAMP_DT"], 0.02, places=12)
        # 参考里显式为 0 的那些项：我们的实现必须靠"清空"来对齐（不能给非零权重）
        for ref_name in RLAMP_REFERENCE_ZEROED:
            self.assertIn(ref_name, values, f"参考 scales 里应有 {ref_name}=0")
            self.assertEqual(values[ref_name], 0, f"{ref_name} 在参考里应为 0（配方变了？）")

    def test_per_step_values_and_style_task_ratio(self):
        """每步系数：线 1.0 / 角 0.3333；AMP 比例沿用参考的 coef 2.0 / lerp 0.3。"""
        consts = {n.targets[0].id: _num(n.value) for n in self.tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id.startswith("RLAMP_")}
        lin = consts["RLAMP_TRACK_LIN_VEL_RAW"] * consts["RLAMP_DT"]
        ang = consts["RLAMP_TRACK_ANG_VEL_RAW"] * consts["RLAMP_DT"]
        self.assertAlmostEqual(lin, 1.0, places=9)
        self.assertAlmostEqual(ang, 1.0 / 3.0, places=6)
        # AMP 比例：风格上限 2.0×(1−0.3)=1.40/步，任务上限 0.3×(1.0+0.333)=0.40/步
        style_cap = 2.0 * (1 - 0.3)
        task_cap = 0.3 * (lin + ang)
        self.assertAlmostEqual(style_cap, 1.40, places=9)
        self.assertAlmostEqual(task_cap, 0.40, places=6)
        self.assertGreater(style_cap / task_cap, 3.0)   # 风格明显主导

    def test_obs_and_action_scales_match_reference(self):
        """critic 缩放、commands 缩放、动作缩放与裁剪都要与参考一致（用户 2026-09-18 要求）。"""
        src = ast.unparse(next(n for n in self.tree.body
                               if isinstance(n, ast.FunctionDef)
                               and n.name == "apply_rlamp_obs_and_action_scales"))
        consts = {n.targets[0].id: _num(n.value) if not isinstance(n.value, ast.Tuple)
                  else tuple(_num(e) for e in n.value.elts)
                  for n in self.tree.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith("RLAMP_")}
        self.assertAlmostEqual(consts["RLAMP_ACTION_SCALE"], 0.25, places=9)
        self.assertEqual(consts["RLAMP_COMMANDS_SCALE"], (2.0, 2.0, 0.25))
        self.assertEqual(consts["RLAMP_CLIP_ACTIONS"], (-100.0, 100.0))
        # helper 必须覆盖 policy 与 critic 两组、且 critic 额外有 lin_vel
        self.assertIn('for group in (\'policy\', \'critic\')', src)
        self.assertIn('obs.base_ang_vel.scale = RLAMP_OBS_SCALES[\'ang_vel\']', src)
        self.assertIn('obs.velocity_commands.scale = RLAMP_COMMANDS_SCALE', src)
        self.assertIn('if group == \'critic\':', src)
        self.assertIn('obs.base_lin_vel.scale = RLAMP_OBS_SCALES[\'lin_vel\']', src)
        self.assertIn('cfg.actions.joint_pos.scale = RLAMP_ACTION_SCALE', src)
        self.assertIn("clip = {'.*': RLAMP_CLIP_ACTIONS}", src)
        # 两个 rlamp 任务都调用
        for cls in ("Imgo2AmpRLAmpEnvCfg", "Imgo2AmpRoughEnvCfg"):
            self.assertIn("apply_rlamp_obs_and_action_scales(self)",
                          _func_src(self.tree, cls, "__post_init__"))

    def test_reference_source_obs_and_action_values(self):
        """本机有参考项目时，直接读它的 normalization / action_scale 比对。"""
        ref = next((p for p in RLAMP_CANDIDATES if p.exists()), None)
        base_ref = next((p for p in RLAMP_BASE_CANDIDATES if p.exists()), None)
        if ref is None or base_ref is None:
            self.skipTest("本机没有 rl_amp 参考项目，跳过")
        tree = _module(ref)
        norm = _class(_module(base_ref), "normalization")
        scales = next(n for n in norm.body if isinstance(n, ast.ClassDef) and n.name == "obs_scales")
        got = {t.id: _num(v) for node in scales.body if isinstance(node, ast.Assign)
               for t, v in zip(node.targets, [node.value]) if isinstance(t, ast.Name)}
        self.assertAlmostEqual(got["lin_vel"], 2.0, places=9)
        self.assertAlmostEqual(got["ang_vel"], 0.25, places=9)
        self.assertAlmostEqual(got["dof_pos"], 1.0, places=9)
        self.assertAlmostEqual(got["dof_vel"], 0.05, places=9)
        # action_scale 在 a1_amp_config 的 control 块里
        ctrl = _class(tree, "control")
        action_scale = next(_num(n.value) for n in ctrl.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "action_scale" for t in n.targets))
        self.assertAlmostEqual(action_scale, 0.25, places=9)
        # clip_actions 在 normalization 块里
        clip_actions = next(_num(n.value) for n in norm.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "clip_actions" for t in n.targets))
        self.assertAlmostEqual(clip_actions, 100.0, places=9)

    # -------------------------------------------------------------------------------
    # 2026-09-18：用户决定「除关节 kp/kd 以外全部对齐参考」后的第二块（指令/域随机化/噪声）
    # -------------------------------------------------------------------------------

    def _rlamp_consts(self):
        return {n.targets[0].id: _num(n.value) for n in self.tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id.startswith("RLAMP_")}

    def test_env_setting_constants(self):
        """指令范围 / 摩擦 / 质量 / 增益 / 推力 / 噪声的常量值必须与参考一致。"""
        consts = self._rlamp_consts()
        self.assertEqual(consts["RLAMP_COMMAND_RANGES"],
                         {"lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-0.3, 0.3),
                          "ang_vel_z": (-1.57, 1.57)})
        self.assertEqual(consts["RLAMP_FRICTION_RANGE"], (0.25, 1.75))
        self.assertEqual(consts["RLAMP_BASE_MASS_RANGE"], (-1.0, 1.0))
        self.assertEqual(consts["RLAMP_GAIN_MULTIPLIER_RANGE"], (0.9, 1.1))
        self.assertEqual(consts["RLAMP_PUSH_INTERVAL_S"], (15.0, 15.0))
        self.assertAlmostEqual(consts["RLAMP_PUSH_VEL_XY"], 1.0, places=9)
        self.assertEqual(consts["RLAMP_NOISE_SCALES"], {"ang_vel": 0.3, "dof_pos": 0.03})
        self.assertEqual(consts["RLAMP_RESET_JOINT_SCALE_RANGE"], (0.5, 1.5))
        self.assertEqual(consts["RLAMP_RESET_VEL_RANGE"], (-0.5, 0.5))
        self.assertEqual(consts["RLAMP_ROUGH_RESET_XY_RANGE"], (-1.0, 1.0))
        self.assertAlmostEqual(consts["RLAMP_REFERENCE_INIT_PROB"], 0.85, places=9)

    def test_env_settings_helper_and_wiring(self):
        """helper 必须逐项改到该改的字段，且两个 rlamp 任务都调用它。"""
        src = ast.unparse(next(n for n in self.tree.body
                               if isinstance(n, ast.FunctionDef)
                               and n.name == "apply_rlamp_env_settings"))
        for needle in (
            "for name, value in RLAMP_COMMAND_RANGES.items()",
            "setattr(cfg.commands.base_velocity.ranges, name, value)",
            "mat.params['static_friction_range'] = RLAMP_FRICTION_RANGE",
            "mat.params['dynamic_friction_range'] = RLAMP_FRICTION_RANGE",
            "mat.params['restitution_range'] = (0.0, 0.0)",
            "cfg.events.randomize_rigid_body_mass_base.params['mass_distribution_params'] = "
            "RLAMP_BASE_MASS_RANGE",
            "cfg.events.randomize_rigid_body_mass_others = None",
            "cfg.events.randomize_com_positions = None",
            "cfg.events.randomize_apply_external_force_torque = None",
            "gains.mode = 'startup'",
            "gains.params['stiffness_distribution_params'] = RLAMP_GAIN_MULTIPLIER_RANGE",
            "gains.params['damping_distribution_params'] = RLAMP_GAIN_MULTIPLIER_RANGE",
            "push.interval_range_s = RLAMP_PUSH_INTERVAL_S",
            "push.params['velocity_range']",
            # reset 两项必须**新建 EventTerm**（AMP 基类里是 None，改值会崩）
            "cfg.events.randomize_reset_base = EventTerm(func=mdp.reset_root_state_uniform",
            "cfg.events.randomize_reset_joints = EventTerm(func=mdp.reset_joints_by_scale",
            "'position_range': RLAMP_RESET_JOINT_SCALE_RANGE",
            "init.params['reference_state_initialization_prob'] = RLAMP_REFERENCE_INIT_PROB",
            "if init is not None:",
            "cfg.observations.policy.base_ang_vel.noise = Unoise(",
            "cfg.observations.policy.joint_pos.noise = Unoise(",
        ):
            self.assertIn(needle, src, f"helper 缺少：{needle}")
        # 必须用"新建"而不是就地改字段：`randomize_reset_base.params[...]` / `joints.params[...]`
        self.assertNotIn("randomize_reset_base.params[", src)
        self.assertNotIn("joints.params[", src)
        self.assertIn("apply_rlamp_env_settings(self)", _func_src(self.tree, "Imgo2AmpRLAmpEnvCfg",
                                                                "__post_init__"))
        # 粗糙版走参考的 custom_origins 分支（初始 xy ±1 m）
        self.assertIn("apply_rlamp_env_settings(self, custom_origins=True)",
                      _func_src(self.tree, "Imgo2AmpRoughEnvCfg", "__post_init__"))
        self.assertIn("RLAMP_ROUGH_RESET_XY_RANGE if custom_origins else (0.0, 0.0)", src)

    def test_env_settings_helper_behaviour_on_fake_cfg(self):
        """真跑一遍 helper（AST 抽出来 exec，喂一个鸭子类型 cfg），确认改的是**值**不是字符串。

        Isaac Lab 的 configclass 在无 GPU 机器上无法实例化（`import omni.kit` 失败），
        所以这里只用标准库重建 helper 依赖的那一层结构，验证它的实际副作用。
        """
        names = {n.targets[0].id for n in self.tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id.startswith("RLAMP_")}
        body = [n for n in self.tree.body
                if (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in names)
                or (isinstance(n, ast.FunctionDef) and n.name == "apply_rlamp_env_settings")]
        self.assertTrue(any(isinstance(n, ast.FunctionDef) for n in body), "helper 没抽出来")
        ns = _fake_helper_namespace()
        exec(compile(ast.Module(body=body, type_ignores=[]), "<amp_env_cfg>", "exec"), ns)

        cfg = _fake_env_cfg()
        # 前置条件：AMP 基类把 reset 随机化关成了 None（冒烟训练就是死在这一步）
        self.assertIsNone(cfg.events.randomize_reset_base)
        self.assertIsNone(cfg.events.randomize_reset_joints)
        ns["apply_rlamp_env_settings"](cfg)

        # 指令
        r = cfg.commands.base_velocity.ranges
        self.assertEqual((r.lin_vel_x, r.lin_vel_y, r.ang_vel_z),
                         ((-1.0, 2.0), (-0.3, 0.3), (-1.57, 1.57)))
        # 摩擦与恢复系数
        mat = cfg.events.randomize_rigid_body_material.params
        self.assertEqual(mat["static_friction_range"], (0.25, 1.75))
        self.assertEqual(mat["dynamic_friction_range"], (0.25, 1.75))
        self.assertEqual(mat["restitution_range"], (0.0, 0.0))
        # 质量：base ±1 kg；其它 body / CoM / 外力力矩被关掉
        self.assertEqual(cfg.events.randomize_rigid_body_mass_base.params["mass_distribution_params"],
                         (-1.0, 1.0))
        self.assertIsNone(cfg.events.randomize_rigid_body_mass_others)
        self.assertIsNone(cfg.events.randomize_com_positions)
        self.assertIsNone(cfg.events.randomize_apply_external_force_torque)
        # PD 增益：范围 ×[0.9,1.1] 且改成启动时随机一次
        gains = cfg.events.randomize_actuator_gains
        self.assertEqual(gains.mode, "startup")
        self.assertEqual(gains.params["stiffness_distribution_params"], (0.9, 1.1))
        self.assertEqual(gains.params["damping_distribution_params"], (0.9, 1.1))
        # 推力：每 15 s、x/y 各 ±1.0 m/s
        push = cfg.events.randomize_push_robot
        self.assertEqual(push.interval_range_s, (15.0, 15.0))
        self.assertEqual(push.params["velocity_range"], {"x": (-1.0, 1.0), "y": (-1.0, 1.0)})
        # reset：**新建**的事件（mode/func 也要对），平地不加 xy/偏航扰动、根速度 ±0.5（6 维）
        base_ev = cfg.events.randomize_reset_base
        self.assertEqual((base_ev.mode, base_ev.func), ("reset", "reset_root_state_uniform"))
        self.assertEqual(base_ev.params["pose_range"],
                         {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)})
        self.assertEqual(base_ev.params["velocity_range"], {k: (-0.5, 0.5) for k in
                                                            ("x", "y", "z", "roll", "pitch", "yaw")})
        joints_ev = cfg.events.randomize_reset_joints
        self.assertEqual((joints_ev.mode, joints_ev.func), ("reset", "reset_joints_by_scale"))
        self.assertEqual(joints_ev.params["position_range"], (0.5, 1.5))
        self.assertEqual(joints_ev.params["velocity_range"], (0.0, 0.0))
        # 参考状态初始化概率：1.0 → 参考的 0.85（否则上面这套 reset 会被完全覆盖）
        self.assertEqual(cfg.events.reference_state_initialization.params[
            "reference_state_initialization_prob"], 0.85)
        # 噪声：只改 policy 的 ang_vel / joint_pos，数值等于参考的 noise_scales
        self.assertEqual(cfg.observations.policy.base_ang_vel.noise, ("Unoise", -0.3, 0.3))
        self.assertEqual(cfg.observations.policy.joint_pos.noise, ("Unoise", -0.03, 0.03))

        # 粗糙版：同样的 helper，但 reset xy 用参考的 ±1 m（`custom_origins=True`），
        # 且参考状态初始化已被关掉（`None`）—— helper 不能因此崩，也不该往 None 上写
        cfg2 = _fake_env_cfg(reference_init=False)
        ns["apply_rlamp_env_settings"](cfg2, custom_origins=True)
        self.assertEqual(cfg2.events.randomize_reset_base.params["pose_range"],
                         {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (0.0, 0.0)})
        self.assertEqual(cfg2.commands.base_velocity.ranges.lin_vel_x, (-1.0, 2.0))
        self.assertIsNone(cfg2.events.reference_state_initialization)

    def test_reference_source_commands_dr_and_noise(self):
        """本机有参考项目时，直接读它的 commands / domain_rand / noise 逐项比对。"""
        ref = next((p for p in RLAMP_CANDIDATES if p.exists()), None)
        base_ref = next((p for p in RLAMP_BASE_CANDIDATES if p.exists()), None)
        if ref is None or base_ref is None:
            self.skipTest("本机没有 rl_amp 参考项目，跳过")
        consts = self._rlamp_consts()
        tree = _module(ref)

        # --- commands ---（参考的第三项叫 `ang_vel_yaw`，Isaac Lab 里是 `ang_vel_z`）
        ref_names = {"lin_vel_x": "lin_vel_x", "lin_vel_y": "lin_vel_y", "ang_vel_z": "ang_vel_yaw"}
        cmds_cls = _class(tree, "commands")
        ranges = _class_assigns(_nested_class(cmds_cls, "ranges"), literal=True)
        for key, value in consts["RLAMP_COMMAND_RANGES"].items():
            self.assertEqual(tuple(ranges[ref_names[key]]), value, f"参考的 {key} 变了")
        cmds = _class_assigns(cmds_cls, literal=True)
        self.assertAlmostEqual(cmds["resampling_time"], 10.0, places=9)
        self.assertFalse(cmds["heading_command"])
        self.assertFalse(cmds["curriculum"])   # 参考没有指令课程

        # --- domain_rand ---
        dr_cls = _class(tree, "domain_rand")
        dr = _class_assigns(dr_cls, literal=True)
        self.assertTrue(dr["randomize_friction"])
        self.assertEqual(tuple(dr["friction_range"]), consts["RLAMP_FRICTION_RANGE"])
        self.assertTrue(dr["randomize_base_mass"])
        self.assertEqual(tuple(dr["added_mass_range"]), consts["RLAMP_BASE_MASS_RANGE"])
        self.assertTrue(dr["randomize_gains"])
        self.assertEqual(tuple(dr["stiffness_multiplier_range"]), consts["RLAMP_GAIN_MULTIPLIER_RANGE"])
        self.assertEqual(tuple(dr["damping_multiplier_range"]), consts["RLAMP_GAIN_MULTIPLIER_RANGE"])
        self.assertTrue(dr["push_robots"])
        self.assertAlmostEqual(dr["push_interval_s"], consts["RLAMP_PUSH_INTERVAL_S"][0], places=9)
        self.assertAlmostEqual(dr["max_push_vel_xy"], consts["RLAMP_PUSH_VEL_XY"], places=9)
        # 「参考没有 CoM/惯量/外力随机化」这个结论的依据：它的 domain_rand **不继承基类**
        self.assertEqual(dr_cls.bases, [],
                         "参考的 domain_rand 若开始继承基类，'没有这些字段'的结论就不成立")
        for absent in ("randomize_com", "randomize_rigid_body_inertia",
                       "randomize_apply_external_force_torque", "randomize_restitution"):
            self.assertNotIn(absent, dr, f"参考里出现了 {absent}，对齐清单需要更新")

        # --- noise：a1 相对基类只覆盖 dof_pos / ang_vel ---
        noise_cls = _class(tree, "noise")
        ns = _class_assigns(_nested_class(noise_cls, "noise_scales"), literal=True)
        self.assertAlmostEqual(ns["dof_pos"], consts["RLAMP_NOISE_SCALES"]["dof_pos"], places=9)
        self.assertAlmostEqual(ns["ang_vel"], consts["RLAMP_NOISE_SCALES"]["ang_vel"], places=9)
        base_ns = _class_assigns(
            _nested_class(_class(_module(base_ref), "noise"), "noise_scales"), literal=True)
        for key in ("lin_vel", "dof_vel", "gravity"):
            self.assertAlmostEqual(base_ns[key], ns[key], places=9,
                                   msg=f"{key} 被参考覆盖了，对齐清单需要更新")
        self.assertAlmostEqual(_class_assigns(noise_cls, literal=True)["noise_level"], 1.0, places=9)

    def test_reference_source_reset_distribution(self):
        """本机有参考项目时，直接读它的 reset 实现，确认我们对齐的是它的真实行为。"""
        ref = next((p for p in RLAMP_ROBOT_ENV_CANDIDATES if p.exists()), None)
        if ref is None:
            self.skipTest("本机没有 rl_amp 参考项目，跳过")
        consts = self._rlamp_consts()
        src = Path(ref).read_text(encoding="utf-8-sig")

        def _rand_float(needle):
            """抓 `torch_rand_float(a, b, <needle>...)` 的两个端点（源码里写的是 `-1.` 这种字面量）。"""
            m = re.search(r"torch_rand_float\(\s*(-?[\d.]+),\s*(-?[\d.]+),\s*" + needle, src)
            self.assertIsNotNone(m, f"参考里找不到 torch_rand_float(..., {needle})")
            return float(m.group(1)), float(m.group(2))

        # 根速度：6 维全部 U(-0.5, 0.5)
        self.assertEqual(_rand_float(r"\(len\(env_ids\), 6\)"), consts["RLAMP_RESET_VEL_RANGE"])
        # 地形分支：xy 再各加 ±1 m（`custom_origins` 才走）
        self.assertEqual(_rand_float(r"\(len\(env_ids\), 2\)"), consts["RLAMP_ROUGH_RESET_XY_RANGE"])
        self.assertIn("if self.custom_origins:", src)
        # 关节：乘默认角 ×U(0.5, 1.5)，速度置 0
        self.assertEqual(_rand_float(r"\(len\(env_ids\), self.num_dof\)"),
                         consts["RLAMP_RESET_JOINT_SCALE_RANGE"])
        self.assertIn("self.dof_vel[env_ids] = 0.", src)

    def test_dataclass_field_order_survives_event_recreation(self):
        """用标准库 dataclass 复现 `_custom_post_init` 的机制，验证"重建事件"不改变执行顺序。

        依据：`configclass.py:392` 的 `setattr(obj, key, deepcopy(value))` 按 `dir(obj)`（字母序）遍历，
        但**给已存在的键赋值不改变 `__dict__` 的插入位置** ⇒ `reference_state_initialization`
        仍然最后执行；拼错的新名字会被追加到最后（所以上面那条测试要守住字段名）。
        """
        from dataclasses import dataclass

        @dataclass
        class Ev:
            material: object = None
            reset_joints: object = None
            reset_base: object = None
            reference_init: object = None

        ev = Ev(material="m")
        order_before = list(ev.__dict__)
        for key in sorted(dir(ev)):        # 模仿 `_custom_post_init` 的 deepcopy 赋值顺序
            if not key.startswith("__"):
                setattr(ev, key, ev.__dict__[key])
        self.assertEqual(list(ev.__dict__), order_before, "deepcopy 循环不该改变属性顺序")
        ev.reset_base = "REBUILT"          # 重建（字段名已存在）
        self.assertEqual(list(ev.__dict__), order_before)
        self.assertEqual(list(ev.__dict__)[-1], "reference_init")
        ev.typo_name = "OOPS"              # 拼错 ⇒ 新键被追加到最后（顺序会反转）
        self.assertEqual(list(ev.__dict__)[-1], "typo_name")

    def test_helper_never_mutates_events_the_amp_base_nulls(self):
        """通用防回归：helper 里**就地改**的 event，不能是 AMP 基类设成 `None` 的那些。

        2026-09-18 冒烟训练就崩在这里（`Imgo2AmpMoveEnvCfg.__post_init__` 把
        `randomize_reset_base` / `randomize_reset_joints` 设成 None，helper 却去取 `.params`）。
        这条用源码交叉检查：helper 读到的 `cfg.events.X` 只要没被"新建 EventTerm"覆盖，
        就不允许出现在基类置 None 的名单里。

        顺带守住第二件事：**新建的 event 名必须是 `EventCfg` 里真实存在的字段**。
        `EventManager` 用 `cfg.__dict__` 顺序执行（`event_manager.py:337`），真实字段名赋值会
        保持原有位置（在 `reference_state_initialization` 之前），而拼错的名字会成为新键、被追加到
        最后 ⇒ 顺序反转 ⇒ 那 85% 参考状态初始化反过来被 reset 随机化覆盖。
        """
        base_src = _func_src(self.tree, "Imgo2AmpMoveEnvCfg", "__post_init__")
        nulled = set(re.findall(r"self\.events\.(\w+) = None", base_src))
        self.assertIn("randomize_reset_base", nulled, "AMP 基类不再关这两项？对齐清单需要复核")
        self.assertIn("randomize_reset_joints", nulled)

        helper = ast.unparse(next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                                  and n.name == "apply_rlamp_env_settings"))
        read = set(re.findall(r"cfg\.events\.(\w+)", helper))
        created = set(re.findall(r"cfg\.events\.(\w+) = EventTerm", helper))
        self.assertEqual(created, {"randomize_reset_base", "randomize_reset_joints"})
        for name in sorted(read - created):
            self.assertNotIn(name, nulled,
                             f"{name} 在 AMP 基类里是 None：helper 必须新建 EventTerm，不能就地改")

        fields = [n.targets[0].id for n in _class(_module(REWARDS_CFG), "EventCfg").body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)]
        for name in sorted(created):
            self.assertIn(name, fields, f"{name} 不是 EventCfg 的字段（拼错会破坏 reset 执行顺序）")

    def test_reference_init_runs_last_among_reset_terms(self):
        """`EventManager` 按 `cfg.__dict__` 顺序执行 reset 项 ⇒ 参考状态初始化必须排在最后。

        源码依据：`event_manager.py:337` 用 `self.cfg.__dict__.items()` 迭代（不排序），
        而 `AMPEventCfg` 的 `reference_state_initialization` 是**子类字段** ⇒ dataclass 字段序里
        排在父类 `EventCfg` 的全部字段之后。这一点是「85% 参考状态 / 15% 自然 reset」成立的前提：
        若顺序反过来，reset 随机化会覆盖那 85%，`reference_state_initialization_prob=0.85`
        就退化成 100% 随机 reset（平地版会变成与我们原来的配置完全不同的分布）。
        """
        base = _class(_module(REWARDS_CFG), "EventCfg")
        reset_terms = []
        for node in base.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                kw = {k.arg: k.value for k in node.value.keywords}
                mode = kw.get("mode")
                if isinstance(mode, ast.Constant) and mode.value == "reset":
                    reset_terms.append(node.targets[0].id)
        self.assertIn("randomize_reset_base", reset_terms)
        self.assertIn("randomize_reset_joints", reset_terms)
        sub_terms = [n.targets[0].id for n in _class(self.tree, "AMPEventCfg").body
                     if isinstance(n, ast.Assign)]
        self.assertEqual(sub_terms, ["reference_state_initialization"],
                         "AMPEventCfg 若再加字段，'参考初始化最后跑' 的结论要重新确认")
        order = reset_terms + sub_terms
        self.assertEqual(order[-1], "reference_state_initialization", order)
        self.assertLess(order.index("randomize_reset_base"),
                        order.index("reference_state_initialization"))
        self.assertLess(order.index("randomize_reset_joints"),
                        order.index("reference_state_initialization"))

    def test_other_variants_keep_their_commands_and_dr(self):
        """flat-amp / go2 变体不得被这一轮对齐动到（它们各自是独立对照）。"""
        for cls in ("Imgo2AmpMoveEnvCfg", "Imgo2AmpGo2StyleEnvCfg"):
            self.assertNotIn("apply_rlamp_env_settings",
                             _func_src(self.tree, cls, "__post_init__"), cls)
        flat = _func_src(self.tree, "Imgo2AmpMoveEnvCfg", "__post_init__")
        self.assertIn("self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)", flat)
        self.assertIn("self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)", flat)
        self.assertIn("heading_command = False", flat)
        # 基础 EventCfg 的原始值必须原样保留（我们只改 rlamp 任务里的副本字段）
        events = ast.unparse(_class(_module(REWARDS_CFG), "EventCfg"))
        self.assertIn("'static_friction_range': (0.3, 1.0)", events)
        self.assertIn("'mass_distribution_params': (0.7, 1.3)", events)
        self.assertIn("interval_range_s=(10.0, 15.0)", events)

    def test_reference_actor_drops_lin_and_ang_vel(self):
        """参考 actor（42 维）**不看线速度也不看角速度** —— 这决定了「actor 45 vs 42」的取舍。

        源码依据：`legged_robot_amp.py` 里 `num_obs == num_privileged_obs - 6` 时
        `obs_buf = privileged_obs_buf[:, 6:]`，而前 6 维正是 `base_lin_vel`+`base_ang_vel`。
        这条同时说明参考的 `noise_scales.ang_vel = 0.3` 在它那边是死代码（噪声先加在
        `[:, 3:6]` 上，随后被切掉）；我们保留 `base_ang_vel`，所以那条噪声在我们这边有效。
        """
        ref = next((p for p in RLAMP_AMP_ENV_CANDIDATES if p.exists()), None)
        if ref is None:
            self.skipTest("本机没有 rl_amp 参考项目，跳过")
        src = Path(ref).read_text(encoding="utf-8-sig")
        self.assertIn("if self.num_obs == self.num_privileged_obs - 6:", src)
        self.assertIn("self.obs_buf = self.privileged_obs_buf[:, 6:]", src)
        cfg = _module(next(p for p in RLAMP_CANDIDATES if p.exists()))
        env_cls = _class(cfg, "env")
        env = _class_assigns(env_cls, literal=True)
        self.assertEqual(env["num_observations"], 42)
        self.assertEqual(env["num_privileged_obs"], 48)
        # 参考的 reset 分布：85% 用参考动作状态、15% 走自然 reset（我们原来写的是 1.0）
        self.assertTrue(env["reference_state_initialization"])
        self.assertAlmostEqual(env["reference_state_initialization_prob"], 0.85, places=9)
        self.assertAlmostEqual(self._rlamp_consts()["RLAMP_REFERENCE_INIT_PROB"], 0.85, places=9)

    def test_rlamp_runner_aligns_min_normalized_std(self):
        """rlamp 四个任务改用 `AMPRLAmpRunnerCfg`（`min_normalized_std=[0.01]*12`，参考值）。"""
        agent = _module(AGENT_CFG)
        cls = _class(agent, "AMPRLAmpRunnerCfg")
        self.assertEqual([ast.unparse(b) for b in cls.bases], ["AMPRunnerCfg"])
        src = ast.unparse(cls)
        self.assertIn("min_normalized_std = [0.01, 0.01, 0.01] * 4", src)
        self.assertIn("experiment_name = 'base_move_amp_rlamp'", src)
        tasks = Path(TASKS_INIT).read_text(encoding="utf-8")
        self.assertEqual(tasks.count("amp_rsl_rl_cfg:AMPRLAmpRunnerCfg"), 4)
        self.assertEqual(tasks.count("amp_rsl_rl_cfg:AMPRunnerCfg"), 2)
        self.assertEqual(tasks.count("amp_rsl_rl_cfg:AMPGo2RunnerCfg"), 2)

    def test_other_variants_keep_deployment_contract(self):
        """其它变体（含已部署的 flat-amp）不得被动到：髋 0.125、clip ±3、commands 1.0。"""
        for cls in ("Imgo2AmpMoveEnvCfg", "Imgo2AmpGo2StyleEnvCfg"):
            src = _func_src(self.tree, cls, "__post_init__")
            self.assertNotIn("RLAMP_ACTION_SCALE", src)
            self.assertNotIn("apply_rlamp_obs_and_action_scales", src)
        # flat-amp 的 action 配置仍是髋 0.125 / 其余 0.25、clip ±3
        flat = _func_src(self.tree, "Imgo2AmpMoveEnvCfg", "__post_init__")
        self.assertIn("'.*_hip_joint': 0.125", flat)
        self.assertIn("clip = {'.*': (-3.0, 3.0)}", flat)

    def test_amp_runner_uses_reference_ratio(self):
        """两个 rlamp 任务都用 `AMPRunnerCfg`（coef 2.0 / lerp 0.3），即参考的比例。"""
        agent = _module(AGENT_CFG)
        src = ast.unparse(_class(agent, "AMPRunnerCfg"))
        self.assertIn("amp_reward_coef = 2.0", src)
        self.assertIn("amp_task_reward_lerp = 0.3", src)
        tasks = Path(TASKS_INIT).read_text(encoding="utf-8")
        for task in ("Imgo2-basemove-flat-amp-rlamp", "Imgo2-basemove-flat-amp-rlamp-play",
                     "Imgo2-basemove-rough-amp-rlamp", "Imgo2-basemove-rough-amp-rlamp-play"):
            self.assertIn(task, tasks)

    def test_flat_and_rough_share_the_same_recipe(self):
        """平地版与粗糙版必须用同一配方（地形才是单变量）。"""
        for cls in ("Imgo2AmpRLAmpEnvCfg", "Imgo2AmpRoughEnvCfg"):
            self.assertIn("apply_rlamp_task_rewards(self)", _func_src(self.tree, cls, "__post_init__"),
                          f"{cls} 没有套用 rl_amp 配方")
        # 粗糙版里不得再单独设置 base_height（该奖励已被清空）
        rough = _func_src(self.tree, "Imgo2AmpRoughEnvCfg", "__post_init__")
        self.assertNotIn("base_height_l2", rough)

    def test_inherited_six_term_recipe_is_still_intact(self):
        """改成 rlamp 不得动到原来的 AMP-only 配方（24500 轮那次）。"""
        keep = _func_src(self.tree, "Imgo2AmpMoveEnvCfg", "_keep_only_amp_task_rewards")
        for needle in ("TRACK_LIN_VEL_PER_STEP = 4.0", "TRACK_ANG_VEL_PER_STEP = 2.0",
                       "BASE_HEIGHT_PER_STEP = -5.0"):
            self.assertIn(needle, keep)


if __name__ == "__main__":
    unittest.main()
