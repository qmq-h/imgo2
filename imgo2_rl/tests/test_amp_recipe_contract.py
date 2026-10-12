"""Offline contracts for the two retained AMP tasks."""

import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move"
AGENT_CFG = BASE / "agents/amp_rsl_rl_cfg.py"
ENV_CFG = BASE / "amp_env_cfg.py"
TASKS = BASE / "__init__.py"


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def _class(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _class_source(path, name):
    return ast.unparse(_class(_tree(path), name))


class AMPRunnerContractTests(unittest.TestCase):
    def test_height_air_time_deficit_selects_feet_and_fanziqi_removes_it(self):
        """Execute the reward selectors without importing Isaac Sim."""
        tree = _tree(ENV_CFG)
        cls = _class(tree, "Imgo2AmpMoveEnvCfg")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_keep_only_amp_task_rewards")
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "apply_rlamp_task_rewards")
        constants = [node for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id in {
                         "RLAMP_TRACK_LIN_VEL_RAW", "RLAMP_TRACK_ANG_VEL_RAW", "RLAMP_KEEP_REWARDS",
                         "FEET_AIR_TIME_THRESHOLD_S", "FEET_AIR_TIME_DEFICIT_PER_STEP",
                     } for target in node.targets)]
        deficit_marker = object()
        namespace = {
            "RewTerm": lambda **kwargs: SimpleNamespace(**kwargs),
            "SceneEntityCfg": lambda name, **kwargs: SimpleNamespace(name=name, **kwargs),
            "mdp": SimpleNamespace(feet_air_time_deficit=deficit_marker),
        }
        exec(compile(ast.Module(body=constants + [method, helper], type_ignores=[]),
                     str(ENV_CFG), "exec"), namespace)
        base_cfg = _class(_tree(BASE.parent / "velocity_env_cfg.py"), "RewardsCfg")
        terms = {
            node.targets[0].id: SimpleNamespace(weight=0.0, params={
                "sensor_cfg": SimpleNamespace(body_names=""),
                "asset_cfg": SimpleNamespace(body_names=""),
            })
            for node in base_cfg.body if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
        }
        cfg = SimpleNamespace(rewards=SimpleNamespace(**terms),
                              sim=SimpleNamespace(dt=0.005), decimation=4,
                              foot_link_name=".*_FOOT", base_link_name="base",
                              foot_body_names=["FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT"])
        namespace[method.name](cfg)
        self.assertIsNone(cfg.rewards.feet_air_time)
        air = cfg.rewards.feet_air_time_deficit
        self.assertIs(air.func, deficit_marker)
        self.assertAlmostEqual(air.weight * 0.02, namespace["FEET_AIR_TIME_DEFICIT_PER_STEP"])
        self.assertLess(air.weight, 0)
        self.assertEqual(air.params["threshold"], namespace["FEET_AIR_TIME_THRESHOLD_S"])
        robot = ET.parse(ROOT.parent / "imgo2_description/urdf/imgo2.urdf").getroot()
        matched = [link.attrib["name"] for link in robot.findall("link")
                   if any(re.fullmatch(pattern, link.attrib["name"])
                          for pattern in air.params["sensor_cfg"].body_names)]
        self.assertEqual(set(matched), {"FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT"})
        # Keep the incoming different-rate coverage, adapted to the deficit recipe.
        expected = {"track_lin_vel_xy_exp", "track_ang_vel_z_exp", "base_height_l2",
                    "lin_vel_z_l2", "ang_vel_xy_l2", "joint_pos_limits", "feet_air_time_deficit"}
        for step_dt in (0.01, 0.02, 0.04):
            cfg.sim.dt = step_dt / cfg.decimation
            namespace[method.name](cfg)
            self.assertIsNone(cfg.rewards.feet_air_time)
            self.assertEqual({name for name, term in vars(cfg.rewards).items() if term is not None}, expected)
            self.assertAlmostEqual(cfg.rewards.track_lin_vel_xy_exp.weight * step_dt, 4.0)
            self.assertAlmostEqual(cfg.rewards.feet_air_time_deficit.weight * step_dt,
                                   namespace["FEET_AIR_TIME_DEFICIT_PER_STEP"])
        namespace[helper.name](cfg)
        active = {name for name, term in vars(cfg.rewards).items() if term is not None}
        self.assertEqual(active, {"track_lin_vel_xy_exp", "track_ang_vel_z_exp"})

    def test_only_two_runner_configs_remain(self):
        tree = _tree(AGENT_CFG)
        classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        self.assertEqual(classes, ["AMPHeightRunnerCfg", "FanziqiAMPRunnerCfg"])

    def test_fanziqi_runner_matches_reference_values(self):
        src = _class_source(AGENT_CFG, "FanziqiAMPRunnerCfg")
        for fragment in (
            "num_steps_per_env = 24",
            "max_iterations = 500000",
            "save_interval = 50",
            "amp_num_preload_transitions = 2000000",
            "amp_reward_coef = 2.0",
            "amp_discr_hidden_dims = [1024, 512]",
            "amp_task_reward_lerp = 0.3",
            "min_normalized_std = [0.05, 0.02, 0.05] * 4",
            "_algorithm_cfg(clamp_noise_std=True)",
        ):
            self.assertIn(fragment, src)

    def test_height_runner_keeps_the_existing_training_budget(self):
        src = _class_source(AGENT_CFG, "AMPHeightRunnerCfg")
        self.assertIn("max_iterations = 40000", src)
        self.assertIn("save_interval = 500", src)
        self.assertIn("_algorithm_cfg(clamp_noise_std=False)", src)

    def test_only_two_amp_task_pairs_are_registered(self):
        src = TASKS.read_text(encoding="utf-8")
        ids = [
            "Imgo2-basemove-flat-amp-height",
            "Imgo2-basemove-flat-amp-height-play",
            "Imgo2-basemove-flat-amp-fanziqi",
            "Imgo2-basemove-flat-amp-fanziqi-play",
        ]
        self.assertEqual(src.count('entry_point="rl_lab.envs:AmpManagerBasedRLEnv"'), 4)
        for task_id in ids:
            self.assertEqual(src.count(f'id="{task_id}"'), 1)
        for removed in ("amp-go2", "rough-amp-rlamp", "amp_rsl_rl_cfg:AMPRunnerCfg",
                        "AMPGo2RunnerCfg"):
            self.assertNotIn(removed, src)

    def test_fanziqi_environment_uses_reference_actor_and_randomization(self):
        src = _class_source(ENV_CFG, "Imgo2AmpRLAmpEnvCfg")
        self.assertIn("apply_rlamp_task_rewards(self)", src)
        self.assertIn("apply_rlamp_obs_and_action_scales(self)", src)
        self.assertIn("apply_rlamp_env_settings(self, external_force=False)", src)
        self.assertIn("self.decimation = 6", src)
        self.assertIn("self.observations.policy.base_ang_vel = None", src)


if __name__ == "__main__":
    unittest.main()
