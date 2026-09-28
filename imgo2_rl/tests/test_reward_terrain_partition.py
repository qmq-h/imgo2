"""步态/姿态 shaping 的**地形分区**锁（2026-09-28 用户决定）＋ 新掩码项接线检查。

背景（用户决策）：
* `feet_air_time` 0.3 → **1.0**、恢复 `feet_slide −0.05`（对齐 PPO rough）；
* `flat_orientation_l2` −0.1 → **−5.0**（对齐 PPO），但**不在障碍地形生效** ⇒ 需要新的掩码类
  `MaskedFlatOrientationL2` 与两张地形清单 `EASY_TERRAIN_NAMES` / `OBSTACLE_TERRAIN_NAMES`；
* 相位核 `feet_gait` **去掉**（保留三项 1e-6 分类器探针）。

这份测试管的是**最容易被漏掉的那一类错**：地形清单动了、掩码引用了不存在的名字、或新增地形没归类
——这些在运行期都是**静默失效**（历史教训：`forward_only_terrain_names` 漏项曾静默退回全向命令）。

Run: python3 -m unittest discover -s tests -p test_reward_terrain_partition.py
"""

import ast
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "scripts" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_terrain_columns as columns  # noqa: E402

CMOE_CFG = REPO / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/CMoE_env_cfg.py"
MDP_REWARDS = REPO / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/rewards.py"


def _cfg_source() -> str:
    return CMOE_CFG.read_text(encoding="utf-8")


def _tuple_literal(name: str) -> tuple[str, ...]:
    """从配置源码里取模块级元组常量（如 `EASY_TERRAIN_NAMES`）。"""
    tree = ast.parse(_cfg_source())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"配置里找不到模块级常量 {name}")


class TestTerrainPartition(unittest.TestCase):
    def setUp(self):
        self.easy = _tuple_literal("EASY_TERRAIN_NAMES")
        self.obstacle = _tuple_literal("OBSTACLE_TERRAIN_NAMES")
        props, _cols = columns.cmoe_overrides()
        base = [name for name, _p in columns.base_sub_terrains()]
        # 任务侧新增/覆盖的键 + 基类键（保持出现顺序，去重）
        self.terrain_keys = tuple(dict.fromkeys([*base, *props.keys()]))

    def test_covers_every_sub_terrain(self):
        """两张清单必须覆盖**全部** `sub_terrains` 键 —— 漏一个就等于那个地形没被归类。"""
        covered = set(self.easy) | set(self.obstacle)
        missing = [key for key in self.terrain_keys if key not in covered]
        self.assertEqual(missing, [], f"这些地形没有归类（漏了会静默不生效）：{missing}")

    def test_disjoint(self):
        self.assertEqual(set(self.easy) & set(self.obstacle), set(),
                         "两张清单不能有交集（同一地形既算易地形又算障碍）")

    def test_names_all_exist(self):
        """清单里的名字必须真的存在（打字错 ⇒ 掩码静默失效）。"""
        for name in (*self.easy, *self.obstacle):
            self.assertIn(name, self.terrain_keys, f"{name} 不在 sub_terrains 里")

    def test_easy_is_the_obstacle_free_set(self):
        """易地形就是用户说的四类：平地、两种斜坡、普通粗糙。"""
        self.assertEqual(set(self.easy),
                         {"flat", "hf_pyramid_slope", "hf_pyramid_slope_inv", "random_rough"})

    def test_level_orientation_only_on_flat_and_rough(self):
        """2026-09-28 用户："斜坡和台阶都不需要保持水平" ⇒ 只有 flat/random_rough 要求机身水平。"""
        level_names = _tuple_literal("LEVEL_ORIENTATION_TERRAIN_NAMES")
        self.assertEqual(set(level_names), {"flat", "random_rough"})
        free = set(self.terrain_keys) - set(level_names)
        for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv",   # 坡面要贴坡
                     "pyramid_stairs", "pyramid_stairs_inv"):      # 台阶要抬头/低头
            self.assertIn(name, free, f"{name} 不该被要求保持水平")

    def test_obstacle_count_matches_total(self):
        self.assertEqual(len(self.terrain_keys), len(self.easy) + len(self.obstacle))


class TestMaskedFlatOrientationWiring(unittest.TestCase):
    def test_cfg_uses_the_masked_class_and_computed_mask(self):
        src = _cfg_source()
        self.assertIn("self.rewards.flat_orientation_l2.func = mdp.MaskedFlatOrientationL2", src)
        self.assertIn("self.rewards.flat_orientation_l2.weight = -5.0", src, "应对齐 PPO rough 的 −5.0")
        # 豁免表是"全部地形键 − 要求水平的地形"现算出来的（不是硬编码一张表）
        self.assertIn("LEVEL_ORIENTATION_TERRAIN_NAMES", src)
        self.assertIn('self.rewards.flat_orientation_l2.params["free_terrain_names"] = tuple(', src)

    def test_masked_class_follows_the_house_pattern(self):
        """新掩码类必须与其它 `Masked*` 同类：`__init__` 缓存静态掩码、`__call__` 乘 `~mask`。"""
        src = MDP_REWARDS.read_text(encoding="utf-8")
        block = re.search(r"class MaskedFlatOrientationL2\(ManagerTermBase\):(.*?)\nclass ", src, re.S)
        self.assertIsNotNone(block, "mdp/rewards.py 里没有 MaskedFlatOrientationL2")
        body = block.group(1)
        self.assertIn("_terrain_type_mask(env", body)
        self.assertIn("flat_orientation_l2(env, asset_cfg)", body)
        self.assertIn("(~self._free_mask)", body)

    def test_phase_kernel_is_off_and_probes_stay(self):
        src = _cfg_source()
        self.assertIn("self.rewards.feet_gait.weight = 0.0", src, "相位核应按用户决定去掉")
        for probe in ("gait_metric_trot", "gait_metric_bound", "gait_metric_pace"):
            self.assertIn(f"{probe} = RewTerm(", src, f"{probe} 探针必须保留（逐列步态唯一读数）")

    def test_air_time_and_slide_match_ppo(self):
        src = _cfg_source()
        self.assertIn("self.rewards.feet_air_time.weight = 1.0", src, "对齐 PPO rough 的 1.0")
        self.assertIn("self.rewards.feet_slide.weight = -0.05", src, "对齐 PPO rough 的 −0.05")
        self.assertIn('self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_FOOT"', src)


if __name__ == "__main__":
    unittest.main()
