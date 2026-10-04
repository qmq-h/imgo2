#!/usr/bin/env python3
"""离线算出每个地形子类型**实际拿到几列**（只用标准库，不需要 Isaac Lab / GPU）。

为什么需要：地形是按 `proportion` **按累积和切列**分配的，而不是"比例 × 列数"那样四舍五入。
Isaac Lab 的规则是（`isaaclab/terrains/terrain_generator.py:240`）：

    for index in range(num_cols):
        sub_index = argmin_k ( index / num_cols + 0.001 < cumsum(proportions_normalized)[k] )

⇒ 0.05 在 `num_cols=20` 时正好 1 列，但在 `num_cols=10` 时**不一定是 0 或 1**；
改一个 `proportion` 也可能"看起来变了、列数没变"。另外**某个地形拿到 0 列时不会有任何报错**，
而引用它名字的奖励掩码（`free_terrain_names` / `bound_terrain_names` / `no_trot_terrain_names`）
会静默退化成"全都不命中"。这类"改了但没生效"只能靠离线算列数发现。

用法::

    python3 imgo2_rl/scripts/tools/check_terrain_columns.py            # 默认任务（cmoe-rough）当前的列数
    python3 imgo2_rl/scripts/tools/check_terrain_columns.py --cols 20 40
    python3 imgo2_rl/scripts/tools/check_terrain_columns.py --task cmoe-rough
    python3 imgo2_rl/scripts/tools/check_terrain_columns.py --task mix-test
    python3 imgo2_rl/scripts/tools/check_terrain_columns.py --task Imgo2-basemove-rough-cmoe-mix-test

**2026-10-04 起支持"按任务区分"**：新增 `--task`（默认 `cmoe-rough`＝训练/play 的 11 类按比例场景）。
`mix-test`（`Imgo2-basemove-rough-cmoe-mix-test`）是**只有 `mix` 一种地形**的受控测试场景：
上面那张 `MASKED_NAMES` 判据里的其它地形名在该场景**根本不存在** ⇒ 引用它们的掩码项在该场景
**恒为 0**，这是**预期**而不是配置错误。因此该任务走**单独的分支**：仍然要求"唯一地形 `mix` 必须
占满全部列"、`num_cols=20`（＝`MIX_TEST_LANES`，20 条**并列**的 mix 道）、`num_rows=1`（**唯一一行
难度**）、`difficulty_range=(0.70, 0.70)`、`sub_terrains.clear()`，但把"其它地形名 0 列"**明确标注为
预期**，而不是报 ❌。**原任务的判据一字未放宽**（`--task cmoe-rough` 仍要求每一项都 ≥1 列）。

比例来源：基类顺序与默认比例读**已安装的 Isaac Lab** `isaaclab/terrains/config/rough.py`
（`ROUGH_TERRAINS_CFG`）；读不到时退回下面记录的默认值。任务侧的覆盖从
`CMoE_env_cfg.py` 的 `sub_terrains[...]` 赋值里解析（含新增键，新增键按赋值顺序追加到末尾）。
**2026-10-04 起解析按类作用域**（只走指定类的 `__post_init__`）：否则同一文件里
`Imgo2CMoEMixTestEnvCfg` 的 `sub_terrains["mix"] = ...(proportion=1.0)` 会污染训练侧的解析结果。
**2026-10-04（第二批）起还会解析模块级常量**：`num_cols`/`num_rows` 现在写成可读常量
（`MIX_TEST_LANES`/`MIX_TEST_LEVELS`），工具用顶层 `NAME = <字面量>` 求值后再读。
**2026-10-04（第三批）**：mix-test 的网格由 20 行缩到 **1 行**（用户："不需要还保持那么多行"），
难度改由 `terrain_generator.difficulty_range = (0.70, 0.70)` **精确固定**（等价旧"第 14 行"的名义
0.70）⇒ 本工具对 mix-test 增加这一项的检查（`MIX_TEST_DIFFICULTY_EXPECTED`）。

**2026-10-04（第四批）**：mix-test 的 `mix` 图案要**占满整条道**（用户："mix 还是太小了…让 mix
占满整条道"）⇒ 本工具对 mix-test 再增加三项检查（**原任务 `cmoe-rough` 的判据一字未动**）：
① `MIX_TEST_PATTERN_SPACING_SCALE` 必须等于**反算值**（`(size[0] − pattern_start_x − 尾部余量)
/ (160·x_unit)`）；② mix 调用必须显式 `fill_stretched_gaps=True`（把拉开出来的空档铺成 `height=0`
可走面）；③ 图案末端 `pattern_start_x + 160·x_unit·scale` 必须 `≥` 目标下界且 `≤ size[0]`
（溢出保护在运行期仍会 raise，上限未放宽）。

**2026-10-04（第五批）**：用户："整体地形放大，原本是10m长就改成20m长，还是布满，但是障碍数量不变，
设置不变，只把间隔改大" ⇒ 单块瓦片 X **8 m → 20 m**（`terrain_generator.size = (20, 4)`），乘子按同一
反算式重算为 **6.00** ⇒ 图案末端 **19.50 m**（目标下界抬到 **19.0 m**）。因此本工具：
① `terrain_size()` 改成**按类作用域**解析（原来全文件扫描会先撞到训练类的 `(8, 4)`），现在
`terrain_size(MIX_TEST_CLASS) = (20, 4)`、`terrain_size(TRAIN_CLASS) = (8, 4)`；
② mix-test 分支新增"**训练侧 `size` 仍是 (8, 4)**"的核对（用户明确要求：`size` 是共享字段，只有评测
cfg 能改）；③ 期望乘子 2.25 → **6.00**、目标下界 7.0 → **19.0**、上限 2.40625 → **6.15625**；
④ 新增"单局时长"检查：道 20 m ÷ 1.0 m/s = 20 s ⇒ 评测 cfg 的 `episode_length_s` 必须 ≥ 25 s。
`--task cmoe-rough` 的判据与输出**一字未改、一字未放宽**。
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CMOE_CFG = REPO / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/CMoE_env_cfg.py"

# 上游 ROUGH_TERRAINS_CFG 的键顺序与默认比例（读不到安装文件时的兜底；2026-09-24 校对过）
DEFAULT_BASE = [
    ("pyramid_stairs", 0.2),
    ("pyramid_stairs_inv", 0.2),
    ("boxes", 0.2),
    ("random_rough", 0.2),
    ("hf_pyramid_slope", 0.1),
    ("hf_pyramid_slope_inv", 0.1),
]
ISAACLAB_ROUGH = Path("/root/IsaacLab/source/isaaclab/isaaclab/terrains/config/rough.py")

# 奖励掩码引用的地形名（改配方时同步；名单不存在 ⇒ 掩码静默失效）
MASKED_NAMES = {
    "joint_mirror.free_terrain_names": ("boxes",),
    "joint_mirror.bound_terrain_names": ("gap",),
    "feet_air_time.free_terrain_names": ("boxes", "gap"),
    "feet_height_body.free_terrain_names": ("boxes", "gap"),
    "feet_air_time_variance.free_terrain_names": ("boxes", "gap"),
    "feet_gait.free_terrain_names": ("boxes", "gap"),
    # 2026-09-30：`base_height_flat_l2`（−35，去重力门的高度罚）用的是**白名单**
    # `active_terrain_names=("flat",)` —— 只有平地上生效。同样是"名字写错 ⇒ 静默失效"
    # （`is_env_assigned_to_terrain` 对未登记的名字返回全 False），所以一并纳入本表。
    "base_height_flat_l2.active_terrain_names": ("flat",),
}
# 注意：`lin_pos_y` / `yaw_abs` 的 `terrain_names=()` 表示**全局生效**（2026-09-24 用户决定
# "所有场景都给脱离中心的惩罚"），因此不在上面这张"必须 ≥1 列"的名单里。

# ------------------------------------------------------------------ 2026-10-04：按任务区分
TRAIN_CLASS = "Imgo2CMoERoughEnvCfg"
# 2026-10-04（第五批）：play 类（评测场景的父类）也要能按类解析 `size`/`episode_length_s`
# ⇒ 工具补一个类名常量（原来只有训练类与评测类两个）。
PLAY_CLASS = "Imgo2CMoERoughPlayEnvCfg"
MIX_TEST_CLASS = "Imgo2CMoEMixTestEnvCfg"
MIX_TEST_TASK = "Imgo2-basemove-rough-cmoe-mix-test"
# 任务 id（或短名）→ 规范任务 id
TASK_ALIASES = {
    "cmoe": "cmoe-rough",
    "cmoe-rough": "cmoe-rough",
    "mix-test": MIX_TEST_TASK,
    MIX_TEST_TASK: MIX_TEST_TASK,
}
TASK_CLASS = {"cmoe-rough": TRAIN_CLASS, MIX_TEST_TASK: MIX_TEST_CLASS}
# mix-test 的期望网格（道沿 Y、难度行沿 X；顺序与 `CMoE_env_cfg.py` 的常量名一致，值在此**写死**
# 以便"常量被改错"能被工具发现；常量的字面值由 `tests/test_check_terrain_columns.py` 单独钉住）。
MIX_TEST_LANES_EXPECTED = 20
# 2026-10-04（第三批）：行数 20 → **1**（只留唯一一行难度；19 行原本永远用不到）。
MIX_TEST_LEVELS_EXPECTED = 1
# 唯一一行的难度：由 `terrain_generator.difficulty_range = (d, d)` 精确固定（等价旧"第 14 行"的
# 名义 14/20 = 0.70）。工具解析 `difficulty_range` 并要求上下界都等于该值。
MIX_TEST_DIFFICULTY_EXPECTED = 0.70
# 2026-10-04（第四批）：`mix` 图案要**占满整条道** ⇒ 乘子必须是"刚好占满"的反算值
#   (size[0] − pattern_start_x − 尾部余量) / (160 · x_unit)
# 第五批（用户："整体地形放大，原本是10m长就改成20m长…只把间隔改大"）：单块瓦片 X 8 m → **20 m**，
# 同一算式重算为 `(20 − 0.30 − 0.50)/(160 × 0.02) = 6.00`，目标下界随之抬到 `图案末端 ≥ 19.0 m`
# （≥ 95 % 的 20 m 道）。值在此**写死**，以便"常量被改错"能被工具发现（常量的字面值由
# `tests/test_check_terrain_columns.py` 单独钉住）。
MIX_TEST_PATTERN_SPACING_SCALE_EXPECTED = 6.00
MIX_TEST_PATTERN_END_MIN_X = 19.0
# 单块瓦片尺寸期望值（**按类作用域**解析）：评测 cfg = (20, 4)；训练/play = (8, 4) 且**不得被改**。
MIX_TEST_TILE_SIZE_EXPECTED = (20.0, 4.0)
TRAIN_TILE_SIZE_EXPECTED = (8.0, 4.0)
# 单局时长（第五批）：道 20 m ÷ 恒定 1.0 m/s = 20 s ⇒ 评测 cfg 的 `episode_length_s` 必须 ≥ 25 s
# （留余量）。`MIX_TEST_EPISODE_LENGTH_EXPECTED` 是当前选定值（35 s），下界用于判定"够不够"。
MIX_TEST_EPISODE_LENGTH_EXPECTED = 35.0
MIX_TEST_EPISODE_LENGTH_MIN_S = 25.0
MIX_TEST_FORWARD_SPEED = 1.0
# `track_mix_terrain` 的图案最后一个索引（`pattern_start_x + 160·x_unit·scale` = 图案末端）
MIX_PATTERN_END_UNITS = 160.0

# 类作用域的目标匹配（`ast.unparse` 用单引号，正则同时接受两种引号）
_SUB_PROP = re.compile(r"sub_terrains\[['\"]([a-z_0-9]+)['\"]\]\.proportion$")
_SUB_TERRAIN = re.compile(r"sub_terrains\[['\"]([a-z_0-9]+)['\"]\]$")
_NUM_COLS = re.compile(r"terrain_generator\.num_cols$")
_NUM_ROWS = re.compile(r"terrain_generator\.num_rows$")
_DIFFICULTY_RANGE = re.compile(r"terrain_generator\.difficulty_range$")
_TERRAIN_SIZE = re.compile(r"terrain_generator\.size$")


def module_constants() -> dict[str, object]:
    """`CMoE_env_cfg.py` 顶层的 `NAME = <可求值字面量>` 常量表（2026-10-04 起 num_cols 用常量写）。"""
    tree = ast.parse(CMOE_CFG.read_text(encoding="utf-8-sig"))
    ns: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                ns[node.targets[0].id] = eval(  # noqa: S307 - 只求值仓库自己的常量表达式
                    compile(ast.Expression(node.value), str(CMOE_CFG), "eval"), {}, dict(ns)
                )
            except Exception:
                continue
    return ns


def literal(node: ast.AST, ns: dict[str, object]) -> object:
    """先按字面量求值；失败则用模块常量命名空间求值；再失败退回源码文本（便于报错时看得见原式）。"""
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    try:
        return eval(compile(ast.Expression(node), str(CMOE_CFG), "eval"), {}, dict(ns))  # noqa: S307
    except Exception:
        return ast.unparse(node)


def base_sub_terrains() -> list[tuple[str, float]]:
    """从已安装的 Isaac Lab 读基类顺序/比例；失败则用兜底值。"""
    if not ISAACLAB_ROUGH.is_file():
        return list(DEFAULT_BASE)
    # 该文件是纯声明式配置：抓 "name": <Ctor>( 与其后的 proportion=...
    src = ISAACLAB_ROUGH.read_text(encoding="utf-8")
    body = src[src.index("sub_terrains={"):]
    names = re.findall(r'"([a-z_0-9]+)":\s*\w+\(', body)
    props = re.findall(r"proportion=([0-9.]+)", body)
    if len(names) != len(props) or not names:
        return list(DEFAULT_BASE)
    return [(name, float(p)) for name, p in zip(names, props)]


def class_post_init(class_name: str) -> ast.FunctionDef | None:
    """取 `CMoE_env_cfg.py` 里指定类的 `__post_init__` 节点（类作用域解析的入口）。"""
    tree = ast.parse(CMOE_CFG.read_text(encoding="utf-8-sig"))
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        return None
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"), None)


def scene_overrides(class_name: str = TRAIN_CLASS) -> dict[str, object]:
    """只解析指定类的 `__post_init__`，返回 ``{props, num_cols, num_rows, difficulty_range,
    sub_terrain_calls, cleared}``。

    2026-10-04：从"整文件 `ast.walk`"改成"类作用域"，因为同一文件里的 mix-test 类也会写
    `sub_terrains["mix"] = ...(proportion=1.0)`；不隔离就会污染训练侧的解析结果（且是静默的）。
    2026-10-04（第三批）：额外解析 `terrain_generator.difficulty_range`（mix-test 用它精确固定难度）。
    2026-10-04（第四批）：额外记录每个 `sub_terrains[name] = <Cfg>(...)` 调用的**全部 kwargs**
    （`sub_terrain_calls`），用于核对 mix-test 的 `pattern_spacing_scale` / `fill_stretched_gaps`。
    """
    info: dict[str, object] = {
        "props": {}, "num_cols": None, "num_rows": None, "difficulty_range": None,
        "sub_terrain_calls": {}, "cleared": False,
    }
    fn = class_post_init(class_name)
    if fn is None:
        return info
    ns = module_constants()
    props: dict[str, float] = info["props"]  # type: ignore[assignment]
    calls: dict[str, dict[str, object]] = info["sub_terrain_calls"]  # type: ignore[assignment]
    for node in ast.walk(fn):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            # `self.scene.terrain.terrain_generator.sub_terrains.clear()` ⇒ 清空基类地形
            if ast.unparse(node.value.func).endswith("sub_terrains.clear"):
                info["cleared"] = True
            continue
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = ast.unparse(node.targets[0])
        m = _SUB_PROP.search(target)
        if m:
            props[m.group(1)] = float(literal(node.value, ns))  # type: ignore[arg-type]
            continue
        m = _SUB_TERRAIN.search(target)
        if m and isinstance(node.value, ast.Call):
            kwargs = {kw.arg: literal(kw.value, ns) for kw in node.value.keywords if kw.arg is not None}
            calls[m.group(1)] = kwargs
            if "proportion" in kwargs:
                props[m.group(1)] = float(kwargs["proportion"])  # type: ignore[arg-type]
            continue
        if _NUM_COLS.search(target):
            info["num_cols"] = int(literal(node.value, ns))  # type: ignore[arg-type]
            continue
        if _NUM_ROWS.search(target):
            info["num_rows"] = int(literal(node.value, ns))  # type: ignore[arg-type]
            continue
        if _DIFFICULTY_RANGE.search(target):
            value = literal(node.value, ns)
            if isinstance(value, (tuple, list)) and len(value) == 2:
                info["difficulty_range"] = (float(value[0]), float(value[1]))
            continue
    return info


def terrain_size(class_name: str = TRAIN_CLASS,
                 default: tuple[float, float] = TRAIN_TILE_SIZE_EXPECTED) -> tuple[float, float]:
    """**指定类**的 `terrain_generator.size`（只在该类的 `__post_init__` 里找）。

    2026-10-04（第四批）：mix-test 的"图案是否占满整条道"要用它核算 ⇒ 从源码里真读，不用猜。
    2026-10-04（第五批）：`size` 是**共享字段**（父类 `Imgo2CMoERoughEnvCfg` 设 `(8, 4)`，评测类
    显式覆盖成 `(20, 4)`）⇒ 原来的"全文件第一个匹配"会**先撞到训练类**（类定义在前）⇒ 改成
    **按类作用域**解析。这样 `terrain_size(MIX_TEST_CLASS) = (20, 4)`、
    `terrain_size(TRAIN_CLASS) = (8, 4)`，也就能断言"训练侧未被改"。
    """
    fn = class_post_init(class_name)
    if fn is None:
        return default
    ns = module_constants()
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        if not _TERRAIN_SIZE.search(ast.unparse(node.targets[0])):
            continue
        value = literal(node.value, ns)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
    return default


def episode_length_s(class_name: str = TRAIN_CLASS) -> float | None:
    """**指定类**的 `episode_length_s` 赋值（只在该类的 `__post_init__` 里找；没写返回 None）。

    2026-10-04（第五批）：道 20 m ÷ 1.0 m/s = 20 s ⇒ 评测场景必须把继承来的 20 s 提到 ≥ 25 s。
    返回 `float`（取值走模块常量求值），没写则 `None`（＝沿用继承来的值）。
    """
    fn = class_post_init(class_name)
    if fn is None:
        return None
    ns = module_constants()
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        if ast.unparse(node.targets[0]) != "self.episode_length_s":
            continue
        value = literal(node.value, ns)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return None


def cmoe_overrides() -> tuple[dict[str, float], dict[str, float]]:
    """返回 (任务侧 proportion 覆盖, num_cols 覆盖)。保持旧签名（＝训练/play 那条链）。"""
    info = scene_overrides(TRAIN_CLASS)
    cols: dict[str, float] = {}
    if info["num_cols"] is not None:
        cols["num_cols"] = info["num_cols"]  # type: ignore[assignment]
    return info["props"], cols  # type: ignore[return-value]


def merged_scene(class_name: str = TRAIN_CLASS) -> tuple[list[tuple[str, float]], dict[str, object]]:
    """基类地形（Isaac Lab rough）+ 任务侧覆盖；若该类 `sub_terrains.clear()` 过 ⇒ 只用覆盖项。"""
    info = scene_overrides(class_name)
    props: dict[str, float] = info["props"]  # type: ignore[assignment]
    if info["cleared"]:
        return [(name, value) for name, value in props.items()], info
    merged: list[tuple[str, float]] = []
    base = base_sub_terrains()
    for name, default in base:
        merged.append((name, props.get(name, default)))
    for name, value in props.items():          # 新增键（如 gap/flat）按赋值顺序追加
        if name not in [n for n, _ in merged]:
            merged.append((name, value))
    return merged, info


def forward_only_names(class_name: str = TRAIN_CLASS) -> tuple[str, ...] | None:
    """读指定类里 `self.commands.base_velocity.forward_only_terrain_names`（没写则 None）。"""
    fn = class_post_init(class_name)
    if fn is None:
        return None
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        if ast.unparse(node.targets[0]) != "self.commands.base_velocity.forward_only_terrain_names":
            continue
        return tuple(ast.literal_eval(node.value))
    return None


def allocate(sub_terrains: list[tuple[str, float]], num_cols: int) -> list[str]:
    """逐字复现 terrain_generator.py:240 的累积和切列规则。"""
    total = sum(p for _, p in sub_terrains)
    norm = [p / total for _, p in sub_terrains]
    cols: list[str] = []
    for index in range(num_cols):
        threshold = index / num_cols + 0.001
        acc = 0.0
        chosen = sub_terrains[-1][0]
        for (name, _), p in zip(sub_terrains, norm):
            acc += p
            if threshold < acc:
                chosen = name
                break
        cols.append(chosen)
    return cols


def report(sub_terrains: list[tuple[str, float]], num_cols: int, label: str) -> dict[str, int]:
    # 2026-09-29：`num_cols == 类数` ⇒ 按**等比例**解释（每类恰好 1 列）。
    # 回放配置（`Imgo2CMoERoughPlayEnvCfg`）就是这么设的：`num_cols = 11` + `sub.proportion = 1.0`
    # （循环赋值，静态读不到）⇒ 若在这里仍按训练比例算，会误报 flat/slope 为 0 列。
    if num_cols == len(sub_terrains):
        sub_terrains = [(name, 1.0) for name, _ in sub_terrains]
        label = f"{label}｜num_cols==类数 ⇒ 等比例（每类 1 列）"
    cols = allocate(sub_terrains, num_cols)
    counts = {name: cols.count(name) for name, _ in sub_terrains}
    print(f"\n=== {label}（num_cols={num_cols}，共 {len(cols)} 列）===")
    for name, _ in sub_terrains:
        print(f"  {name:22s} {counts[name]:>3d} 列")
    zero = [name for name, c in counts.items() if c == 0]
    if zero:
        print(f"  ⚠️ 0 列（引用它的掩码会静默失效）：{', '.join(zero)}")
    return counts


def check_mask_columns(counts: dict[str, int], num_cols: int) -> int:
    """原判据（**不放宽**）：掩码引用的每个地形名都必须存在且 ≥1 列。"""
    problems = 0
    for key, names in MASKED_NAMES.items():
        for name in names:
            if name not in counts:
                print(f"\n❌ 掩码 {key} 引用的地形名 '{name}' 不在 sub_terrains 里 ⇒ 静默失效")
                problems += 1
            elif counts[name] == 0:
                print(f"\n❌ 掩码 {key} 引用的 '{name}' 在 num_cols={num_cols} 时只有 0 列 ⇒ 静默失效")
                problems += 1
    return problems


def mix_test_report() -> tuple[int, list[str]]:
    """`Imgo2-basemove-rough-cmoe-mix-test`（只 mix）的**单独**判定。

    与原判据的区别只有一处、且是**新增的标注**而不是放宽：本场景**只有 mix** ⇒ 其它地形名不在
    `sub_terrains` 里、引用它们的掩码项在本场景**恒为 0**（`is_env_assigned_to_terrain` 对未登记的
    名字返回全 False），这里把每一处**逐条打印成"预期"**；同时仍然强制：`sub_terrains.clear()` 过、
    只有 mix、mix 占满全部道、`num_cols=20`（＝`MIX_TEST_LANES`，20 条**并列**的 mix 道）、
    `num_rows=1`（**唯一一行难度**，2026-10-04 第三批：19 行原本永远用不到）、
    `difficulty_range=(0.70, 0.70)`（精确固定该行难度）。
    原任务（`--task cmoe-rough`）走 :func:`check_mask_columns`，判据一字未改。

    2026-10-04（第二批）：`num_cols` 由 **1 → 20**（用户："地形不要按照列排，放在行里面"）——20 条
    mix 道沿世界 Y 并排，难度沿世界 X；因此这里期望的列数也改成 20，且要求"每道恰好 1 列"。
    2026-10-04（第三批，用户："我不需要还保持那么多行，我需要他们并列"）：行数 **20 → 1**、世界
    X 由 160 m 缩到 8 m；难度不再靠"钉第 14 行"，改为本函数新检查的
    `difficulty_range = (0.70, 0.70)`。
    """
    info = scene_overrides(MIX_TEST_CLASS)
    props: dict[str, float] = info["props"]  # type: ignore[assignment]
    num_cols = info["num_cols"] if info["num_cols"] is not None else MIX_TEST_LANES_EXPECTED
    # 单块瓦片尺寸：**按类作用域**解析（评测 cfg 第五批覆盖成 (20,4)；训练/play 仍是 (8,4)）。
    size = terrain_size(MIX_TEST_CLASS)
    train_size = terrain_size(TRAIN_CLASS)
    lines: list[str] = []
    problems = 0

    def say(text: str = ""):
        lines.append(text)
        print(text)

    say(f"\n=== [{MIX_TEST_TASK}] 受控测试场景（只 mix；cfg 类 {MIX_TEST_CLASS}）===")
    if not info["cleared"]:
        say("  ❌ 该任务必须先 `sub_terrains.clear()` 再只放 mix（否则会继承基类的 11 类地形）")
        problems += 1
    if list(props.keys()) != ["mix"]:
        say(f"  ❌ 该任务的 sub_terrains 必须**有且只有** mix，实测 {list(props.keys())}")
        problems += 1
    elif props["mix"] != 1.0:
        say(f"  ❌ mix 的 proportion 应为 1.0，实测 {props['mix']}")
        problems += 1
    if info["num_cols"] != MIX_TEST_LANES_EXPECTED:
        say(f"  ❌ num_cols 应为 {MIX_TEST_LANES_EXPECTED}（＝`MIX_TEST_LANES`：20 条并列的 mix 道，"
            f"道数＝可同时评估的环境数上限），实测 {info['num_cols']}")
        problems += 1
    if info["num_rows"] != MIX_TEST_LEVELS_EXPECTED:
        say(f"  ❌ num_rows 应为 {MIX_TEST_LEVELS_EXPECTED}（＝`MIX_TEST_LEVELS`：只留**唯一一行**难度，"
            f"沿世界 +X ⇒ 世界 X 只有 8 m），实测 {info['num_rows']}")
        problems += 1
    difficulty_range = info["difficulty_range"]
    if difficulty_range != (MIX_TEST_DIFFICULTY_EXPECTED, MIX_TEST_DIFFICULTY_EXPECTED):
        say(f"  ❌ difficulty_range 应为 ({MIX_TEST_DIFFICULTY_EXPECTED}, "
            f"{MIX_TEST_DIFFICULTY_EXPECTED})（上下界同值 ⇒ 课程公式里 `upper−lower = 0`，"
            f"行内 U(0,1) 抖动被消掉 ⇒ 每块瓦片精确 d = "
            f"{MIX_TEST_DIFFICULTY_EXPECTED}），实测 {difficulty_range}")
        problems += 1

    names = list(props.keys())
    counts = report([(n, props[n]) for n in names], num_cols, f"{MIX_TEST_TASK}｜只 mix（20 道并列）")
    for name in names:
        if counts[name] < 1:
            say(f"  ❌ 唯一地形 '{name}' 在 num_cols={num_cols} 下 0 列 ⇒ 命令与掩码都会静默失效")
            problems += 1
        elif counts[name] != MIX_TEST_LANES_EXPECTED:
            say(f"  ❌ 唯一地形 '{name}' 应占满全部 {MIX_TEST_LANES_EXPECTED} 道（每道 1 列），"
                f"实测 {counts[name]} 列")
            problems += 1
    if not problems:
        say(f"  ✅ 唯一地形 mix 有 {counts.get('mix', 0)} 列（num_cols={num_cols}、num_rows={info['num_rows']}）"
            f"⇒ 网格 **{MIX_TEST_LANES_EXPECTED} 道 × {MIX_TEST_LEVELS_EXPECTED} 难度行**"
            f"（世界 {size[0]:g} m(X) × {size[1] * MIX_TEST_LANES_EXPECTED:g} m(Y)），难度由 difficulty_range="
            f"({difficulty_range[0]}, {difficulty_range[1]}) 精确固定 ⇒ `--num_envs ≤ "
            f"{MIX_TEST_LANES_EXPECTED}` 时环境 i → 第 i 道")

    # ------------------------------------- 2026-10-04（第五批）：瓦片放大到 20 m ＋ 训练侧不得被改
    say(f"\n  ── 第五批检查：单块瓦片 `terrain_generator.size`（评测 {size[0]:g}×{size[1]:g} m；"
        f"训练 {train_size[0]:g}×{train_size[1]:g} m）──")
    if (round(size[0], 9), round(size[1], 9)) != MIX_TEST_TILE_SIZE_EXPECTED:
        say(f"  ❌ 评测 cfg 的 `terrain_generator.size` 应为 {MIX_TEST_TILE_SIZE_EXPECTED}"
            f"（第五批：单块瓦片 X 8 m → 20 m），实测 {size}")
        problems += 1
    else:
        say(f"  ✅ 评测 cfg 的 `terrain_generator.size` = {size[0]:g}×{size[1]:g} m"
            f"（第五批：X 由 8 m 放大到 20 m；Y 仍 4 m）")
    if (round(train_size[0], 9), round(train_size[1], 9)) != TRAIN_TILE_SIZE_EXPECTED:
        say(f"  ❌ **训练/play 侧**的 `terrain_generator.size` 必须仍是 {TRAIN_TILE_SIZE_EXPECTED}"
            f"（`size` 是共享字段，只有评测 cfg 能改），实测 {train_size}")
        problems += 1
    else:
        say(f"  ✅ **训练/play 侧**的 `terrain_generator.size` 仍是 {train_size[0]:g}×{train_size[1]:g} m"
            f"（共享字段未被评测场景带偏）")

    # ------------------------------- 2026-10-04（第四/五批）：图案占满整条道 ＋ 单局时长
    mix_kwargs = (info.get("sub_terrain_calls") or {}).get("mix", {})  # type: ignore[union-attr]
    scale = module_constants().get("MIX_TEST_PATTERN_SPACING_SCALE")
    say(f"\n  ── 第四/五批检查：`mix` 图案占满整条道（瓦片 size[0] = {size[0]:g} m）──")
    if scale is None:
        say("  ❌ 解析不到常量 `MIX_TEST_PATTERN_SPACING_SCALE`")
        problems += 1
    elif abs(float(scale) - MIX_TEST_PATTERN_SPACING_SCALE_EXPECTED) > 1.0e-9:  # type: ignore[arg-type]
        say(f"  ❌ `MIX_TEST_PATTERN_SPACING_SCALE` 应为 {MIX_TEST_PATTERN_SPACING_SCALE_EXPECTED}"
            f"（＝「刚好占满整条道」的反算值），实测 {scale}")
        problems += 1
    if mix_kwargs.get("fill_stretched_gaps") is not True:
        say(f"  ❌ mix 调用必须显式 `fill_stretched_gaps=True`（把拉开的空档铺成 height=0 可走面），"
            f"实测 {mix_kwargs.get('fill_stretched_gaps')!r}")
        problems += 1
    offset = mix_kwargs.get("pattern_start_x")
    x_unit = mix_kwargs.get("x_unit")
    if not isinstance(offset, (int, float)) or not isinstance(x_unit, (int, float)):
        say(f"  ❌ 解析不到 mix 的 `pattern_start_x` / `x_unit`（实测 {offset!r} / {x_unit!r}）"
            "⇒ 无法核算图案长度")
        problems += 1
    elif isinstance(scale, (int, float)):
        end = float(offset) + MIX_PATTERN_END_UNITS * float(x_unit) * float(scale)
        tail = size[0] - end
        upper = (size[0] - float(offset)) / (MIX_PATTERN_END_UNITS * float(x_unit))
        if end > size[0] + 1.0e-9:
            say(f"  ❌ 图案末端 {end:.4f} m 超出瓦片长 {size[0]:g} m（`track_mix_terrain` 的溢出保护"
                f"会在运行期直接 raise；上限 = {upper:.5f}）")
            problems += 1
        elif end < MIX_TEST_PATTERN_END_MIN_X:
            say(f"  ❌ 图案末端 {end:.4f} m < 目标 {MIX_TEST_PATTERN_END_MIN_X:g} m（未占满整条道）")
            problems += 1
        else:
            say(f"  ✅ `pattern_spacing_scale` = {scale}（反算值：({size[0]:g} − {offset} − "
                f"{tail:.2f})/(160×{x_unit})）＋ `fill_stretched_gaps=True` ⇒ 图案末端 = "
                f"{offset} + 160×{x_unit}×{scale} = {end:.2f} m（占 {100.0 * end / size[0]:.2f} % 的 "
                f"{size[0]:g} m 道），尾部平地 {tail:.2f} m；乘子上限 "
                f"({size[0]:g}−{offset})/(160×{x_unit}) = {upper:.5f}（溢出保护**不放宽**：超限直接 raise）")

    # ---------------------------- 2026-10-04（第五批）：单局时长必须够走完 20 m（1.0 m/s ⇒ 20 s）
    episode_length = episode_length_s(MIX_TEST_CLASS)
    train_episode_length = episode_length_s(TRAIN_CLASS)
    needed = size[0] / MIX_TEST_FORWARD_SPEED
    say(f"\n  ── 第五批检查：单局时长（道 {size[0]:g} m ÷ 恒定 {MIX_TEST_FORWARD_SPEED:g} m/s = "
        f"{needed:.1f} s）──")
    if episode_length is None:
        say(f"  ❌ 评测 cfg 未覆盖 `episode_length_s`（继承来的默认是 20 s，正好等于走完 20 m 的时间、"
            f"没有余量）⇒ 必须显式设为 ≥ {MIX_TEST_EPISODE_LENGTH_MIN_S:g} s")
        problems += 1
    elif episode_length < MIX_TEST_EPISODE_LENGTH_MIN_S - 1.0e-9:
        say(f"  ❌ 评测 cfg 的 `episode_length_s` = {episode_length:g} s < 下界 "
            f"{MIX_TEST_EPISODE_LENGTH_MIN_S:g} s（走完 {size[0]:g} m 要 {needed:.1f} s，余量不足）")
        problems += 1
    else:
        say(f"  ✅ 评测 cfg 的 `episode_length_s` = {episode_length:g} s ≥ 下界 "
            f"{MIX_TEST_EPISODE_LENGTH_MIN_S:g} s（走完 {size[0]:g} m 要 {needed:.1f} s，余量 "
            f"{episode_length - needed:.1f} s）")
    if train_episode_length is None:
        say("  ℹ️ 训练/play 链**未**覆盖 `episode_length_s`（沿用父类 20 s）—— 本批只改评测 cfg，"
            "训练时长一字未动")
    else:
        say(f"  ℹ️ 训练/play 链的 `episode_length_s` = {train_episode_length:g} s（与评测侧各自独立）")

    absent = sorted({n for refs in MASKED_NAMES.values() for n in refs} - set(names))
    say("  ⚠️ 本场景**只有 mix**；掩码引用但本场景**不存在**的地形名（原任务判据**不放宽**，"
        "这些项在本场景恒为 0 属**预期**）：")
    if not absent:
        say("    （无）")
    for key, refs in MASKED_NAMES.items():
        for name in refs:
            if name in absent:
                say(f"    - {key} → {name}：本场景无该地形 ⇒ 该项本场景恒为 0（预期，不是配置错误）")

    inherited = False
    fwd = forward_only_names(MIX_TEST_CLASS)
    if fwd is None:
        inherited = True
        fwd = forward_only_names(TRAIN_CLASS)
    if fwd is None:
        say("  ❌ 继承链里找不到 `forward_only_terrain_names`（掩码/命令会静默退化）")
        problems += 1
    elif "mix" not in fwd:
        say(f"  ❌ `forward_only_terrain_names` 未覆盖唯一的 mix：{fwd}")
        problems += 1
    else:
        tag = "（继承父类）" if inherited else ""
        extra = [n for n in fwd if n not in names]
        note = ""
        if extra:
            note = (f"；另含 {len(extra)} 个本场景不存在的地形名（{', '.join(extra)}）—— 命令项"
                    " `MixTestVelocityCommand` 根本不读这张表，且 `is_env_assigned_to_terrain` 对未登记"
                    " 的名字返回全 False ⇒ 在这里是**惰性**的，属预期")
        say(f"  ✅ `forward_only_terrain_names`{tag} 覆盖 mix{note}")

    say("\n结论：" + ("test 任务只有 mix 一种地形；掩码引用的其它地形名在本场景恒为 0，属**预期** ✅"
                      if problems == 0 else f"test 任务有 {problems} 处问题 ❌"))
    return problems, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cols", nargs="*", type=int, default=None,
                        help="要算的 num_cols（默认＝CMoE_env_cfg 里配置的值，2026-09-28 起为 40）")
    parser.add_argument("--task", default="cmoe-rough",
                        help="要检查的任务：cmoe-rough（默认，训练/play 的 11 类按比例）或 "
                             f"mix-test（＝{MIX_TEST_TASK}，只有 mix）")
    args = parser.parse_args(argv)

    if args.task not in TASK_ALIASES:
        parser.error(f"未知任务 {args.task!r}；可选：{', '.join(sorted(TASK_ALIASES))}")
    task = TASK_ALIASES[args.task]

    if task == MIX_TEST_TASK:
        if args.cols:
            print(f"提示：`--task {args.task}` 的列数由 cfg 决定（现为 {MIX_TEST_LANES_EXPECTED} 条 mix 道），"
                  "`--cols` 被忽略")
        return 1 if mix_test_report()[0] else 0

    merged, info = merged_scene(TRAIN_CLASS)
    col_over = {k: v for k, v in {"num_cols": info["num_cols"]}.items() if v is not None}

    print("CMoE rough 的 sub_terrains 比例（基类 + 任务侧覆盖）：")
    props = info["props"]
    for name, value in merged:
        mark = "  ←覆盖" if name in props else ""  # type: ignore[operator]
        print(f"  {name:22s} {value:.4f}{mark}")

    default_cols = [int(col_over["num_cols"])] if col_over.get("num_cols") else [40]
    cols_to_check = args.cols if args.cols else default_cols
    counts_by_cols = {}
    for num_cols in cols_to_check:
        label = "训练／play 当前值" if num_cols in default_cols else f"num_cols={num_cols}（手动指定）"
        counts_by_cols[num_cols] = report(merged, num_cols, label)

    problems = 0
    for num_cols, counts in counts_by_cols.items():
        problems += check_mask_columns(counts, num_cols)
    if col_over:
        print(f"\n提示：配置里显式覆盖过 num_cols = {col_over['num_cols']}（训练与 play 都是）")

    # 2026-09-24 用户决定「所有场景都只给超前的速度」⇒ `forward_only_terrain_names` 必须覆盖
    # **全部** sub_terrains（漏一项，那一列就会静默退回全向命令）。
    names = [n for n, _ in merged]
    fwd = forward_only_names(TRAIN_CLASS)
    if fwd is None:
        print("\n（配置未设置 forward_only_terrain_names，跳过覆盖率校验）")
    else:
        missing = [n for n in names if n not in fwd]
        unknown = [n for n in fwd if n not in names]
        if missing or unknown:
            print(f"\n❌ forward_only_terrain_names 未覆盖全部地形：缺 {missing}／多余 {unknown}")
            problems += 1
        else:
            print(f"\n✅ forward_only_terrain_names 覆盖全部 {len(names)} 类地形（全场景前向命令）")

    print("\n结论：" + ("全部掩码引用的地形名都有 ≥1 列 ✅" if problems == 0 else f"有 {problems} 处问题 ❌"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
