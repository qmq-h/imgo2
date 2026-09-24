#!/usr/bin/env python3
"""离线核对奖励项的**最终生效值**与赋值顺序（只用标准库）。

为什么需要它：本仓的奖励权重是"基类默认 0 → 各任务的 `__post_init__` 逐条赋值"这一模式，
**后赋值的赢**。这个模式已经静默坑过两次：

* 2026-09-24：把 `joint_mirror.weight = -1.0` 写在 `6220e43` 的零权重块**之前**，被随后的
  `joint_mirror.weight = 0.0` 覆盖 ⇒ mirror **静默失效**，训练照跑、日志无异常。
* 2026-09-24：`joint_mirror.params["mirror_joints"]` 被误删 hip（推理本身也错了，见 docs §29.9.3）。

`py_compile` 查不出这类问题，而真正的 env 构造需要一个跑起来的 Isaac Sim。本脚本用 AST
**按源文件顺序**（＝Python 实际执行顺序）重放赋值，给出最终生效表并标出可疑模式，
在没有任何 Isaac Lab / GPU 的机器上也能跑。

用法::

    python3 imgo2_rl/scripts/tools/check_reward_overrides.py            # 打印全部链
    python3 imgo2_rl/scripts/tools/check_reward_overrides.py cmoe       # 只看一条

只解析 `self.rewards.<term>.<attr> = <字面量>` 形式；`.params[...]` 之类的赋值不参与
（它们不改变"哪些项生效"，只改项内部参数）。**本脚本不替代真实运行**：它证明不了掩码在
正确的地形列上生效、也证明不了量级合适。

「⚠️ 被清零」的提示**包含有意为之的归零**（例如刻意关掉 `lin_vel_z_l2`）——它的用途是让每
一处归零都变成一次**有意识的选择**，而不是像 2026-09-24 的 `joint_mirror` 那样被晚赋值顺手抹掉。
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity"
BASE_MOVE = PKG / "base_move"

TARGET_RE = re.compile(r"^self\.rewards\.(\w+)\.(\w+)$")

# 链＝(显示名, [(文件, 类名), ...])，按"基类在前、子类在后"的执行顺序排列。
# 子类的 `__post_init__` 先调 `super().__post_init__()`，所以基类条目先生效。
CHAINS = {
    "ppo-rough": [
        ("ppo rough", [(PKG / "velocity_env_cfg.py", "RewardsCfg"),
                       (BASE_MOVE / "rough_env_cfg.py", "Imgo2RoughEnvCfg")]),
    ],
    "ppo-flat": [
        ("ppo flat", [(PKG / "velocity_env_cfg.py", "RewardsCfg"),
                      (BASE_MOVE / "rough_env_cfg.py", "Imgo2RoughEnvCfg"),
                      (BASE_MOVE / "flat_env_cfg.py", "Imgo2FlatEnvCfg")]),
    ],
    "cmoe": [
        ("cmoe rough", [(PKG / "velocity_env_cfg.py", "RewardsCfg"),
                        (BASE_MOVE / "CMoE_env_cfg.py", "CMoERewardsCfg"),
                        (BASE_MOVE / "rough_env_cfg.py", "Imgo2RoughEnvCfg"),
                        (BASE_MOVE / "CMoE_env_cfg.py", "Imgo2CMoERoughEnvCfg")]),
    ],
}


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def term_defaults(path: Path, class_name: str) -> tuple[dict[str, object], list[str]]:
    """读 `class ...RewardsCfg` 体内 `name = RewTerm(..., weight=<字面量>, ...)` 的默认权重。"""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        return {}, []
    weights: dict[str, object] = {}
    for node in cls.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
            continue
        for kw in node.value.keywords:
            if kw.arg == "weight":
                weights[target.id] = _literal(kw.value)
    return weights, []


def post_init_assignments(path: Path, class_name: str) -> list[tuple[int, str, str, object, str]]:
    """读 `__post_init__` 里 `self.rewards.<term>.<attr> = <字面量>` 的赋值，按源文件顺序。"""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        return []
    fn = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"), None)
    if fn is None:
        return []
    rows = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        m = TARGET_RE.match(ast.unparse(node.targets[0]))
        if m:
            rows.append((node.lineno, m.group(1), m.group(2), _literal(node.value),
                         ast.unparse(node.value)))
    return sorted(rows)


def run_chain(steps) -> tuple[dict[str, object], dict[str, str], list[str]]:
    weights: dict[str, object] = {}
    funcs: dict[str, str] = {}
    notes: list[str] = []
    for path, class_name in steps:
        defaults, _ = term_defaults(path, class_name)
        for term, value in defaults.items():
            weights[term] = value
        rows = post_init_assignments(path, class_name)
        seen: dict[str, list[int]] = {}
        for lineno, term, attr, value, raw in rows:
            seen.setdefault(term, []).append(lineno)
            if attr == "weight":
                prev = weights.get(term)
                if isinstance(prev, (int, float)) and prev != 0 and value == 0:
                    notes.append(f"⚠️ {term}: 原为 {prev!r}，在 `{path.name}:{lineno}` 被**清零**"
                                 f" ⇒ 静默失效（这正是 2026-09-24 `joint_mirror` 失效的模式）")
                weights[term] = value
            elif attr == "func":
                funcs[term] = raw
        for term, linenos in seen.items():
            weight_lines = [ln for ln, t, a, _, _ in rows if t == term and a == "weight"]
            if len(weight_lines) > 1:
                notes.append(f"ℹ️ {term}: 同一个 `__post_init__` 里 weight 被赋值 {len(weight_lines)} 次"
                             f"（L{weight_lines}）⇒ 只有**最后一次**生效")
            del linenos
    return weights, funcs, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("chains", nargs="*", default=None,
                        help=f"要核对的链，可选：{', '.join(CHAINS)}（默认全部）")
    args = parser.parse_args(argv)

    names = args.chains or list(CHAINS)
    unknown = [n for n in names if n not in CHAINS]
    if unknown:
        parser.error(f"未知的链：{unknown}；可选：{', '.join(CHAINS)}")

    for name in names:
        for label, steps in CHAINS[name]:
            weights, funcs, notes = run_chain(steps)
            effective = {t: v for t, v in weights.items() if isinstance(v, (int, float)) and v != 0}
            dropped = sorted(t for t, v in weights.items() if isinstance(v, (int, float)) and v == 0)
            print(f"\n===== [{name}] {label}：生效 {len(effective)} 项 =====")
            for term in sorted(effective, key=lambda t: (-abs(float(effective[t])), t)):
                tag = f"  ← func={funcs[term].split('.')[-1]}" if term in funcs else ""
                print(f"  {term:26s} {float(effective[term]):>12.6g}{tag}")
            print(f"  权重 0 ⇒ 被 disable_zero_weight_rewards() 移除：{', '.join(dropped) if dropped else '（无）'}")
            dead = sorted(t for t in funcs if t not in effective)
            if dead:
                notes.append(f"⚠️ {dead} 的 func 被换成了自定义类但最终权重为 0 ⇒ 该赋值是死代码")
            for note in notes:
                print(f"  {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
