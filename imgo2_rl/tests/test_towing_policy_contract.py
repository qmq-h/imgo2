"""Frozen low-level policy contract tests; standard Python, no simulator.

P4 的接口契约必须与部署侧一致（那份配置已在 Gazebo 用三条契约验证过），所以这里
**直接读部署 yaml 交叉核对**，防止 `policy_cfg.py` 与部署配置两处漂移。适配器部分用
临时 TorchScript 假策略做端到端检查（观测组装、裁剪、last_action 回灌、维数自检），
本机无 GPU 时用 CPU；torch / pyyaml 缺失则跳过对应项。
"""

import importlib
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

RL = Path(__file__).resolve().parents[1]
REPO = RL.parent
UTILS_REL = "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/utils"
DEPLOY_REL = "imgo2_deploy/policy/imgo2"


def _load_utils_package():
    """按路径注册合成包，使 `utils/` 内的相对导入可用（父包会 import isaaclab）。"""
    package_dir = RL / UTILS_REL
    name = "imgo2_towing_utils_under_test"
    spec = importlib.util.spec_from_file_location(
        name, package_dir / "__init__.py", submodule_search_locations=[str(package_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name


PKG = _load_utils_package()
policy_cfg = importlib.import_module(f"{PKG}.policy_cfg")

# 部署 yaml 的观测名 → 本仓契约的观测名
TERM_ALIASES = {
    "ang_vel": "base_ang_vel",
    "gravity_vec": "projected_gravity",
    "commands": "velocity_commands",
    "dof_pos": "joint_pos_rel",
    "dof_vel": "joint_vel_rel",
    "actions": "last_action",
}


def _load_yaml(relative):
    try:
        import yaml
    except ImportError:  # pragma: no cover - 环境无 pyyaml 时跳过
        raise unittest.SkipTest("pyyaml not available")
    path = REPO / DEPLOY_REL / relative
    if not path.is_file():  # pragma: no cover - 防御
        raise unittest.SkipTest(f"deploy config missing: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class AmpContractTests(unittest.TestCase):
    """AMP 45 维契约：自洽 + 与部署 yaml 一致。"""

    def setUp(self):
        self.cfg = policy_cfg.AMP_POLICY

    def test_contract_is_self_consistent(self):
        self.cfg.validate()                                  # 不抛错即通过
        self.assertEqual(self.cfg.total_observation_dim, 45)
        self.assertEqual(self.cfg.num_observations, 45)

    def test_observation_layout_is_contiguous_and_ordered(self):
        layout = self.cfg.observation_layout()
        self.assertEqual([term for term, _, _ in layout],
                         list(self.cfg.observation_terms))
        expected_start = 0
        for term, start, dim in layout:
            self.assertEqual(start, expected_start, term)
            expected_start += dim
        self.assertEqual(expected_start, self.cfg.num_observations)

    def test_matches_deploy_yaml(self):
        deploy = _load_yaml("amp/config.yaml")["imgo2/amp"]
        base = _load_yaml("base.yaml")["imgo2"]

        self.assertEqual(self.cfg.num_observations, deploy["num_observations"])
        self.assertEqual(self.cfg.num_joints, deploy["num_of_dofs"])
        self.assertEqual(self.cfg.model_name, deploy["model_name"])
        self.assertEqual(self.cfg.clip_obs, deploy["clip_obs"])
        self.assertEqual(list(self.cfg.observation_terms),
                         [TERM_ALIASES[t] for t in deploy["observations"]])
        self.assertEqual(list(self.cfg.observation_history), list(deploy["observations_history"]))
        for ours, theirs, label in (
            (self.cfg.action_scale, deploy["action_scale"], "action_scale"),
            (self.cfg.default_dof_pos, deploy["default_dof_pos"], "default_dof_pos"),
            (self.cfg.clip_actions_lower, deploy["clip_actions_lower"], "clip_actions_lower"),
            (self.cfg.clip_actions_upper, deploy["clip_actions_upper"], "clip_actions_upper"),
            (self.cfg.joint_mapping, deploy["joint_mapping"], "joint_mapping"),
        ):
            self.assertEqual([float(v) for v in ours], [float(v) for v in theirs], label)
        # 观测缩放：yaml 按项给，契约按维展开
        scales, offset = [], 0
        per_term = {
            "base_ang_vel": [deploy["ang_vel_scale"]] * 3,
            "projected_gravity": [1.0] * 3,
            "velocity_commands": [float(v) for v in deploy["commands_scale"]],
            "joint_pos_rel": [deploy["dof_pos_scale"]] * 12,
            "joint_vel_rel": [deploy["dof_vel_scale"]] * 12,
            "last_action": [1.0] * 12,
        }
        for term in self.cfg.observation_terms:
            scales.extend(per_term[term])
            offset += len(per_term[term])
        self.assertEqual(offset, self.cfg.num_observations)
        self.assertEqual([float(v) for v in self.cfg.observation_scales], scales)
        # 策略周期 = base.yaml 的物理/关节控制步长 × 降采样。
        self.assertEqual(self.cfg.control_dt, base["dt"] * base["decimation"])

    def test_exported_policy_exists_with_recorded_hash(self):
        path = self.cfg.model_path
        self.assertTrue(path.is_file(), path)
        # README AMP-05 记录的 24500 导出件；哈希变了要在文档里同步
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest[:16], "cba59d44e387834e",
                         "部署 policy.pt 与 README AMP-05 记录的哈希不一致")

    def test_joint_targets_follow_deploy_semantics(self):
        action = [0.5] * self.cfg.num_joints
        targets = self.cfg.joint_targets(action)
        for got, default, scale in zip(targets, self.cfg.default_dof_pos, self.cfg.action_scale):
            self.assertAlmostEqual(got, default + scale * 0.5, places=12)
        with self.assertRaises(ValueError):
            self.cfg.joint_targets([0.0] * 3)


# 2026-09-20 训练机实跑 `tow_drag.py` 时报告的 PhysX DOF 顺序：按运动学树广度优先，
# 全部 hip → 全部 thigh → 全部 shank。**不是** URDF 的声明顺序（URDF 是逐腿）。
ISAAC_LAB_ASSET_ORDER = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_shank_joint", "FR_shank_joint", "RL_shank_joint", "RR_shank_joint",
)


class JointOrderTests(unittest.TestCase):
    """PhysX 的 DOF 顺序与策略顺序不同，必须显式置换（否则动作打到错误关节）。"""

    def setUp(self):
        self.cfg = policy_cfg.AMP_POLICY
        self.perm = self.cfg.asset_permutation(ISAAC_LAB_ASSET_ORDER)

    def test_permutation_is_a_valid_reordering(self):
        self.assertEqual(sorted(self.perm), list(range(12)))
        for position, name in enumerate(self.cfg.joint_names):
            self.assertEqual(ISAAC_LAB_ASSET_ORDER[self.perm[position]], name)

    def test_known_landmarks(self):
        # 策略第 0/1/2 个是 FL 的 hip/thigh/shank ⇒ 资产里的下标是 0/4/8
        self.assertEqual(self.perm[:3], (0, 4, 8))
        # 策略第 3 个是 FR_hip ⇒ 资产下标 1
        self.assertEqual(self.perm[3], 1)
        # 恒等置换会错：资产的第 1 号是 FR_hip，而策略的第 1 号是 FL_thigh
        self.assertNotEqual(list(self.perm), list(range(12)))
        self.assertEqual(ISAAC_LAB_ASSET_ORDER[1], "FR_hip_joint")

    def test_permutation_reproduces_the_contract_default_pose(self):
        """资产默认角按置换重排后必须等于契约 default_dof_pos。

        URDF 用正则把每类关节设成同一个值（hip 0 / thigh 0.87 / shank -1.82），
        所以这条能把「置换对不对」和「契约对不对」一起验证。
        """
        by_type = {"hip": 0.0, "thigh": 0.87, "shank": -1.82}
        asset_defaults = [by_type[name.split("_")[1]] for name in ISAAC_LAB_ASSET_ORDER]
        reordered = [asset_defaults[index] for index in self.perm]
        self.assertEqual(reordered, [float(v) for v in self.cfg.default_dof_pos])

    def test_urdf_declaration_order_is_leg_by_leg(self):
        """记录这条差异的来源：URDF 逐腿声明，Isaac Lab 广度优先重排。"""
        import xml.etree.ElementTree as ET
        root = ET.parse(REPO / "imgo2_description/urdf/imgo2.urdf").getroot()
        declared = [j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"]
        self.assertEqual(declared, list(self.cfg.joint_names),
                         "URDF 的可动关节顺序应与策略顺序一致（逐腿）")
        self.assertNotEqual(declared, list(ISAAC_LAB_ASSET_ORDER),
                            "Isaac Lab 会按广度优先重排，正是需要置换的原因")

    def test_rejects_mismatched_joint_sets(self):
        with self.assertRaises(ValueError):
            self.cfg.asset_permutation(ISAAC_LAB_ASSET_ORDER[:-1])
        with self.assertRaises(ValueError):
            self.cfg.asset_permutation(ISAAC_LAB_ASSET_ORDER[:-1] + ("wheel_fl_joint",))
        with self.assertRaises(ValueError):
            self.cfg.asset_permutation(("same",) * 12)


class ContractValidationTests(unittest.TestCase):
    """坏契约必须显式报错，不能静默降级。"""

    def setUp(self):
        self.base = policy_cfg.AMP_POLICY

    def _replace(self, **changes):
        fields = {f: getattr(self.base, f) for f in self.base.__dataclass_fields__}
        fields.update(changes)
        return policy_cfg.LowLevelPolicyCfg(**fields)

    def test_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            self._replace(num_observations=48).validate()
        with self.assertRaises(ValueError):
            self._replace(observation_dims=(3, 3, 3, 12, 12)).validate()

    def test_rejects_bad_scale_lengths(self):
        with self.assertRaises(ValueError):
            self._replace(action_scale=(0.25,) * 11).validate()
        with self.assertRaises(ValueError):
            self._replace(observation_scales=(1.0,) * 44).validate()

    def test_rejects_non_permutation_joint_mapping(self):
        with self.assertRaises(ValueError):
            self._replace(joint_mapping=(0,) * 12).validate()

    def test_rejects_inverted_action_clip(self):
        with self.assertRaises(ValueError):
            self._replace(clip_actions_lower=(3.0,) * 12).validate()

    def test_rejects_nonpositive_scale_and_dt(self):
        with self.assertRaises(ValueError):
            self._replace(action_scale=(0.0,) * 12).validate()
        with self.assertRaises(ValueError):
            self._replace(control_dt=0.0).validate()

    def test_rejects_duplicate_joint_names(self):
        with self.assertRaises(ValueError):
            self._replace(joint_names=("same",) * 12).validate()

    def test_unknown_policy_name_raises(self):
        with self.assertRaises(KeyError):
            policy_cfg.get_policy("nope")
        self.assertIs(policy_cfg.get_policy("amp"), self.base)


class AdapterTests(unittest.TestCase):
    """用临时 TorchScript 假策略端到端验证适配器（无 GPU 时走 CPU）。"""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("torch not available")
        cls.torch = torch

    def _write_echo_policy(self, directory, out_features=12):
        torch = self.torch

        class Echo(torch.nn.Module):
            def __init__(self, width: int):
                super().__init__()
                self.width = width

            def forward(self, obs):
                return obs[:, :self.width]

        path = Path(directory) / "policy.pt"
        torch.jit.save(torch.jit.script(Echo(out_features)), str(path))
        return path

    def _cfg_with_model(self, path, **changes):
        fields = {f: getattr(policy_cfg.AMP_POLICY, f)
                  for f in policy_cfg.AMP_POLICY.__dataclass_fields__}
        fields.update(policy_dir=Path(path).parent, model_name=Path(path).name)
        fields.update(changes)
        return policy_cfg.LowLevelPolicyCfg(**fields)

    def test_rejects_export_with_wrong_output_dim(self):
        from importlib import import_module
        adapter = import_module(f"{PKG}.low_level_policy")
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_echo_policy(tmp, out_features=7)
            with self.assertRaises(ValueError):
                adapter.FrozenLowLevelPolicy(self._cfg_with_model(path), device="cpu")

    def test_missing_export_file_is_reported(self):
        from importlib import import_module
        adapter = import_module(f"{PKG}.low_level_policy")
        cfg = self._cfg_with_model(Path(tempfile.gettempdir()) / "definitely_missing.pt")
        with self.assertRaises(FileNotFoundError):
            adapter.FrozenLowLevelPolicy(cfg, device="cpu")

    def test_observation_assembly_order_scales_and_clip(self):
        torch = self.torch
        from importlib import import_module
        adapter = import_module(f"{PKG}.low_level_policy")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg_with_model(self._write_echo_policy(tmp))
            policy = adapter.FrozenLowLevelPolicy(cfg, device="cpu")
            parts = adapter.parts_from_robot_state(
                base_ang_vel=torch.full((2, 3), 2.0),
                projected_gravity=torch.tensor([[0.0, 0.0, -1.0]] * 2),
                velocity_command=torch.tensor([0.5, 0.0, 0.0]),
                joint_pos=torch.full((2, 12), 0.1),
                joint_vel=torch.full((2, 12), 4.0))
            obs = policy.build_observation(parts)
            self.assertEqual(tuple(obs.shape), (2, 45))
            # ang_vel 缩放 0.25
            for i in range(3):
                self.assertAlmostEqual(float(obs[0, i]), 0.5, places=6)
            # gravity 无缩放（下标 3..5，z 分量是 -1）
            self.assertAlmostEqual(float(obs[0, 4]), 0.0, places=6)
            self.assertAlmostEqual(float(obs[0, 5]), -1.0, places=6)
            # commands 无缩放（AMP 的 commands_scale 是 [1,1,1]）
            self.assertAlmostEqual(float(obs[0, 6]), 0.5, places=6)
            # joint_pos_rel = 0.1 - default，无缩放
            self.assertAlmostEqual(float(obs[0, 9]), 0.1 - 0.0, places=6)
            self.assertAlmostEqual(float(obs[0, 10]), 0.1 - 0.87, places=6)
            self.assertAlmostEqual(float(obs[0, 11]), 0.1 - (-1.82), places=6)
            # joint_vel 缩放 0.05
            self.assertAlmostEqual(float(obs[0, 21]), 4.0 * 0.05, places=6)
            # last_action 初始为 0
            for i in range(33, 45):
                self.assertAlmostEqual(float(obs[0, i]), 0.0, places=6)

    def test_clip_obs_is_applied_to_concatenated_vector(self):
        torch = self.torch
        from importlib import import_module
        adapter = import_module(f"{PKG}.low_level_policy")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg_with_model(self._write_echo_policy(tmp), clip_obs=3.0)
            policy = adapter.FrozenLowLevelPolicy(cfg, device="cpu")
            parts = adapter.parts_from_robot_state(
                base_ang_vel=torch.full((1, 3), 100.0),          # 缩放后 25 → 被 clip 到 3
                projected_gravity=torch.zeros(1, 3),
                velocity_command=torch.zeros(3),
                joint_pos=torch.zeros(1, 12),
                joint_vel=torch.zeros(1, 12))
            obs = policy.build_observation(parts)
            for i in range(3):
                self.assertAlmostEqual(float(obs[0, i]), 3.0, places=6)

    def test_action_is_clipped_and_targets_follow_default_offset(self):
        torch = self.torch
        from importlib import import_module
        adapter = import_module(f"{PKG}.low_level_policy")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg_with_model(self._write_echo_policy(tmp))
            # echo 策略回传观测前 12 维 = ang_vel*0.25 = 100 ⇒ 必须被 clip 到 3
            policy = adapter.FrozenLowLevelPolicy(cfg, device="cpu")
            parts = adapter.parts_from_robot_state(
                base_ang_vel=torch.full((1, 3), 100.0),
                projected_gravity=torch.zeros(1, 3),
                velocity_command=torch.zeros(3),
                joint_pos=torch.zeros(1, 12),
                joint_vel=torch.zeros(1, 12))
            out = policy.step(parts)
            self.assertTrue(bool((out.action <= 3.0 + 1e-6).all()))
            self.assertTrue(bool((out.action >= -3.0 - 1e-6).all()))
            expected = torch.tensor(cfg.default_dof_pos) + torch.tensor(cfg.action_scale) * out.action
            self.assertTrue(bool(torch.allclose(out.joint_targets, expected, atol=1e-6)))

    def test_last_action_feeds_back_into_next_observation(self):
        torch = self.torch
        from importlib import import_module
        adapter = import_module(f"{PKG}.low_level_policy")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg_with_model(self._write_echo_policy(tmp))
            policy = adapter.FrozenLowLevelPolicy(cfg, device="cpu")
            parts = adapter.parts_from_robot_state(
                base_ang_vel=torch.zeros(1, 3),
                projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
                velocity_command=torch.zeros(3),
                joint_pos=torch.zeros(1, 12),
                joint_vel=torch.zeros(1, 12))
            first = policy.step(parts)
            second = policy.build_observation(parts)
            # 第二步观测的 last_action 段应等于第一步的动作
            for i in range(12):
                self.assertAlmostEqual(float(second[0, 33 + i]), float(first.action[0, i]), places=6)
            # reset 后回到 0，且返回站定步数
            settle = policy.reset()
            self.assertEqual(settle, int(round(cfg.reset_settle_s / cfg.control_dt)))
            self.assertTrue(bool((policy.build_observation(parts)[0, 33:] == 0).all()))

    def test_reset_settle_steps_are_positive(self):
        """reset 契约必须给出明确的站定步数，不能是 0（否则 reset 后立刻压速度指令）。"""
        self.assertGreater(policy_cfg.AMP_POLICY.reset_settle_s, 0.0)


if __name__ == "__main__":
    unittest.main()
