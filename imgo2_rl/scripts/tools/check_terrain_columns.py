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

    python3 imgo2_rl/scripts/tools/check_terrain_columns.py            # 训练/play 两套列数
    python3 imgo2_rl/scripts/tools/check_terrain_columns.py --cols 20 40

比例来源：基类顺序与默认比例读**已安装的 Isaac Lab** `isaaclab/terrains/config/rough.py`
（`ROUGH_TERRAINS_CFG`）；读不到时退回下面记录的默认值。任务侧的覆盖从
`CMoE_env_cfg.py` 的 `sub_terrains[...]` 赋值里解析（含新增键，新增键按赋值顺序追加到末尾）。
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
}
# 注意：`lin_pos_y` / `yaw_abs` 的 `terrain_names=()` 表示**全局生效**（2026-09-24 用户决定
# "所有场景都给脱离中心的惩罚"），因此不在上面这张"必须 ≥1 列"的名单里。


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


def cmoe_overrides() -> tuple[dict[str, float], dict[str, float]]:
    """返回 (任务侧 proportion 覆盖, num_cols 覆盖)。"""
    tree = ast.parse(CMOE_CFG.read_text(encoding="utf-8-sig"))
    props: dict[str, float] = {}
    cols: dict[str, float] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = ast.unparse(node.targets[0])
        # 注意：`ast.unparse` 用单引号，正则必须同时接受两种引号
        m = re.fullmatch(r"sub_terrains\[['\"]([a-z_0-9]+)['\"]\]", target)
        if m and isinstance(node.value, ast.Call):
            for kw in node.value.keywords:
                if kw.arg == "proportion":
                    props[m.group(1)] = float(ast.literal_eval(kw.value))
            continue
        m = re.fullmatch(r"sub_terrains\[['\"]([a-z_0-9]+)['\"]\]\.proportion", target)
        if m:
            props[m.group(1)] = float(ast.literal_eval(node.value))
            continue
        m = re.fullmatch(r"self\.scene\.terrain\.terrain_generator\.num_cols", target)
        if m:
            cols["num_cols"] = int(ast.literal_eval(node.value))
    return props, cols


def forward_only_names() -> tuple[str, ...] | None:
    """读 `CMoE_env_cfg.py` 里 `self.commands.base_velocity.forward_only_terrain_names`（没写则 None）。"""
    tree = ast.parse(CMOE_CFG.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
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
    cols = allocate(sub_terrains, num_cols)
    counts = {name: cols.count(name) for name, _ in sub_terrains}
    print(f"\n=== {label}（num_cols={num_cols}，共 {len(cols)} 列）===")
    for name, _ in sub_terrains:
        print(f"  {name:22s} {counts[name]:>3d} 列")
    zero = [name for name, c in counts.items() if c == 0]
    if zero:
        print(f"  ⚠️ 0 列（引用它的掩码会静默失效）：{', '.join(zero)}")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cols", nargs="*", type=int, default=None,
                        help="要算的 num_cols（默认 20＝训练默认、10＝CMoE play）")
    args = parser.parse_args(argv)

    base = base_sub_terrains()
    props, col_over = cmoe_overrides()
    merged: list[tuple[str, float]] = []
    for name, default in base:
        merged.append((name, props.get(name, default)))
    for name, value in props.items():          # 新增键（如 gap/flat）按赋值顺序追加
        if name not in [n for n, _ in merged]:
            merged.append((name, value))

    print("CMoE rough 的 sub_terrains 比例（基类 + 任务侧覆盖）：")
    for name, value in merged:
        mark = "  ←覆盖" if name in props else ""
        print(f"  {name:22s} {value:.4f}{mark}")

    train_cols = args.cols if args.cols else [20, 10]
    counts_by_cols = {}
    for num_cols in train_cols:
        label = "训练" if num_cols == 20 else ("play" if num_cols == 10 else f"num_cols={num_cols}")
        counts_by_cols[num_cols] = report(merged, num_cols, label)

    problems = 0
    for key, names in MASKED_NAMES.items():
        for num_cols, counts in counts_by_cols.items():
            for name in names:
                if name not in counts:
                    print(f"\n❌ 掩码 {key} 引用的地形名 '{name}' 不在 sub_terrains 里 ⇒ 静默失效")
                    problems += 1
                elif counts[name] == 0:
                    print(f"\n❌ 掩码 {key} 引用的 '{name}' 在 num_cols={num_cols} 时只有 0 列 ⇒ 静默失效")
                    problems += 1
    if col_over:
        print(f"\n提示：配置里显式覆盖过 num_cols = {col_over['num_cols']}（play 用）")

    # 2026-09-24 用户决定「所有场景都只给超前的速度」⇒ `forward_only_terrain_names` 必须覆盖
    # **全部** sub_terrains（漏一项，那一列就会静默退回全向命令）。
    names = [n for n, _ in merged]
    fwd = forward_only_names()
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
