"""Wiring guard: cfg kwargs and package exports must match the declared classes.

2026-10-17 训练机实跑报错：「`TowingDecoderCfg.__init__() got an unexpected keyword
argument 'mass_coef'`」——契约 v4 删掉 `mass_coef` 时，**训练用的 agent cfg**
（`tasks/.../towing/agents/upper_ppo_cfg.py`）漏改，而离线测试没有 import 它（它经
`isaaclab.utils.configclass`，无 Isaac Lab 的解释器 import 不了），所以 619 项全绿却挡不住。

同一次还发现 `rl_lab/modules/__init__.py` 仍在导出已删除的 `mass_supervision_weight`
（`from .towing_decoder import mass_supervision_weight`）——一 import 包就崩。

本文件用**纯标准库 AST** 补这两类守卫（不需要 isaaclab / torch）：

1. `rl_lab/config/*.py` 里每个 `*Cfg` 声明的字段（含同包内的基类继承）当成白名单，
   扫 `towing/agents/*.py` 里对它们的构造调用，任何未知 kwarg（例如旧的 `mass_coef`）即失败；
2. `rl_lab/modules/__init__.py` 从 `towing_decoder` 导入的名字，必须在 `towing_decoder.py`
   顶层真实存在。
"""
import ast
from pathlib import Path
import unittest

RL = Path(__file__).resolve().parents[1]
CONFIG_DIR = RL / "scripts/rl_lab/rl_lab/config"
MODULES_DIR = RL / "scripts/rl_lab/rl_lab/modules"
AGENTS_DIR = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/agents"
TASKS_DIR = RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based"


def _parse(path):
    # `utf-8-sig`：仓里个别既有脚本（如 rl_lab/ppo/train.py）带 UTF-8 BOM，Python 的 import
    # 能接受，但 `ast.parse(str)` 不接受，读的时候统一先剥掉。
    return ast.parse(Path(path).read_text(encoding="utf-8-sig"), filename=str(path))


def _annotated_fields(class_node):
    """类体内 `x: T = ...` / `x: T` 声明的字段名。"""
    fields = set()
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            fields.add(node.target.id)
    return fields


def _base_names(class_node):
    return [base.id for base in class_node.bases if isinstance(base, ast.Name)]


def _config_registry():
    """{类名: 字段集合}，字段沿同包内的基类递归合并。"""
    classes = {}
    for path in sorted(CONFIG_DIR.glob("*.py")):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = (_annotated_fields(node), _base_names(node))
    resolved = {}

    def fields_of(name, seen=()):
        if name in resolved:
            return resolved[name]
        if name not in classes or name in seen:
            return set()
        own, bases = classes[name]
        merged = set(own)
        for base in bases:
            merged |= fields_of(base, seen + (name,))
        resolved[name] = merged
        return merged

    return {name: fields_of(name) for name in classes}


class ConfigKwargTests(unittest.TestCase):
    def test_agent_cfgs_only_pass_declared_fields(self):
        registry = _config_registry()
        self.assertIn("TowingDecoderCfg", registry)
        self.assertIn("TowingOnPolicyRunnerCfg", registry)
        # 契约 v4 的 decoder 字段里不得再有质量系数（内存里/源码里都不该有）
        self.assertNotIn("mass_coef", registry["TowingDecoderCfg"])
        problems = []
        # 扫**所有任务**与训练脚本：不只 towing/agents（本次事故就在那里），
        # 任何地方构造 rl_lab 的 *Cfg 都必须只用声明过的字段。同名类只在 rl_lab/config 里
        # 定义，所以按类名匹配是安全的。
        for root in (TASKS_DIR, RL / "scripts/rl_lab"):
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                for node in ast.walk(_parse(path)):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                        continue
                    name = node.func.id
                    if name not in registry:
                        continue
                    for keyword in node.keywords:
                        if keyword.arg is not None and keyword.arg not in registry[name]:
                            problems.append(
                                f"{path.relative_to(RL)}:{node.lineno} {name}(..., "
                                f"{keyword.arg}=...) 不是该配置类声明的字段")
        self.assertEqual(problems, [], "配置类构造里出现了未声明的字段：\n" + "\n".join(problems))

    def test_no_removed_quality_supervision_kwargs_anywhere(self):
        """v4 删掉的字段不许在训练/测试台/工具里回潮（**只看代码，不看注释/文档字符串**）。

        文档里写「这里没有 mass_coef」是应该保留的说明，所以不能做纯文本扫描：这里只抓
        关键字参数、属性访问、赋值目标与 import 名。
        """
        removed = {"mass_coef", "mass_supervision_weight", "decoder_mass_weight", "weight_rollout"}
        offenders = []
        for root in (RL / "source/imgo2_rl/imgo2_rl/tasks/manager_based/towing",
                     RL / "scripts/rl_lab",
                     RL / "scripts/towing",
                     RL / "scripts/tools"):
            for path in root.rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                for node in ast.walk(_parse(path)):
                    hit = None
                    if isinstance(node, ast.Call):
                        for keyword in node.keywords:
                            if keyword.arg in removed:
                                hit = f"{keyword.arg}=（关键字参数）"
                    elif isinstance(node, ast.Attribute) and node.attr in removed:
                        hit = f".{node.attr}（属性访问）"
                    elif isinstance(node, ast.Name) and node.id in removed \
                            and isinstance(node.ctx, (ast.Store, ast.Load)):
                        hit = f"{node.id}（名字引用）"
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            if alias.name in removed:
                                hit = f"import {alias.name}"
                    if hit:
                        offenders.append(f"{path.relative_to(RL)}:{node.lineno} {hit}")
        self.assertEqual(offenders, [], "已删除的质量监督字段又被引用：\n" + "\n".join(offenders))


class PackageExportTests(unittest.TestCase):
    def test_modules_package_only_exports_existing_symbols(self):
        decoder = MODULES_DIR / "towing_decoder.py"
        defined = set()
        for node in _parse(decoder).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                defined |= {target.id for target in node.targets if isinstance(target, ast.Name)}
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        problems = []
        init = MODULES_DIR / "__init__.py"
        for node in ast.walk(_parse(init)):
            if isinstance(node, ast.ImportFrom) and node.module == "towing_decoder":
                for alias in node.names:
                    if alias.name not in defined:
                        problems.append(f"modules/__init__.py 导出了 towing_decoder 里不存在的 "
                                        f"`{alias.name}`（第 {node.lineno} 行）")
        self.assertEqual(problems, [], "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
