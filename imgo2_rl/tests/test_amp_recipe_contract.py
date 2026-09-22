"""Offline contracts for the two retained AMP tasks."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move"
AGENT_CFG = BASE / "agents/amp_rsl_rl_cfg.py"
ENV_CFG = BASE / "amp_env_cfg.py"
TASKS = BASE / "__init__.py"


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _class(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _class_source(path, name):
    return ast.unparse(_class(_tree(path), name))


class AMPRunnerContractTests(unittest.TestCase):
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
