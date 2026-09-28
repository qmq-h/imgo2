"""离线核对「把 45 维运动先验移植进 CMoE 专家」是否成立。

**为什么要先跑这个**：CMoE 的专家吃 157 维（45 本体 + 3 explicit + 16 state latent
+ 77 地形 + 16 terrain latent），而现成的 PPO / AMP 策略只吃 45 维。想把先验塞进专家，
唯一的前提是**前 45 维与前 45 个输入在顺序、尺度、动作定义上逐项一致**；只要有一项不一致，
零填充移植就会把"先验能走"变成"喂错尺度所以不走"，而这一点在训练曲线里很难看出来
（表现为"先验没用"，而不是报错）。所以本脚本把两件事分开查：

1. **契约层（纯标准库）**：从**在跑 run 自带的 `params/env.yaml`**、训练侧源码
   （`rough_env_cfg.py` / `amp_env_cfg.py` / `assets/imgo2.py` / `cmoe_actor_critic.py`）
   与**部署侧 `imgo2_deploy/policy/imgo2/*/config.yaml`** 里读出实际声明的观测顺序/尺度、
   动作 scale/clip、默认姿态、kp/kd、关节顺序，逐项对比并推导维数（45/77/125/157/527）。
   数值一律从来源文件里读，不在这里复制一份预期值 —— 否则来源改了脚本还会"通过"。
2. **数值层（需要 torch）**：用**真实的** `CMoEExpertActorCritic` / `CMoEActorCritic`
   建模型，把先验零填充装进去，检查：
   * 专家输出与先验逐位相同（含"后 112 维塞随机非零输入仍不变"）；
   * 新增列权重恰为 0，但**梯度确实流到这些列**（证明是"接进来、初始权重为零"，
     而不是"输入被丢掉"）；
   * 5 个专家都装同一个先验后，**混合输出恒等于先验、与门控权重无关**
     （这是"第 0 步就是先验步态"的依据）。

**不做什么**：不改任何配置、不训练、不实例化 Isaac Lab。本脚本退出码 0 只代表这些离线判据通过，
不代表先验在 Isaac 里能走（那需要 step-0 教师回放）。

用法（在 imgo2_rl 目录下）：

    python scripts/tools/check_cmoe_expert_init.py                     # 自动挑最新 run 与先验
    python scripts/tools/check_cmoe_expert_init.py --skip-torch        # 只查契约层
    python scripts/tools/check_cmoe_expert_init.py --prior <model.pt> [--prior <policy.pt>]
    python scripts/tools/check_cmoe_expert_init.py --run-yaml <run>/params/env.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]  # imgo2_rl/
REPO_ROOT = ROOT.parent  # <repo>/
RL_LAB_ROOT = ROOT / "scripts" / "rl_lab"

CMAOE_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/CMoE_env_cfg.py"
ROUGH_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/rough_env_cfg.py"
AMP_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/amp_env_cfg.py"
ASSET_CFG = ROOT / "source/imgo2_rl/imgo2_rl/assets/imgo2.py"
CMOE_AC = RL_LAB_ROOT / "rl_lab/modules/cmoe_actor_critic.py"
DEPLOY_CFGS = {
    "amp": REPO_ROOT / "imgo2_deploy/policy/imgo2/amp/config.yaml",
    "ppo": REPO_ROOT / "imgo2_deploy/policy/imgo2/ppo/config.yaml",
}
LOG_TREES = (ROOT / "logs", REPO_ROOT / "logs")

# 结构事实：每条 obs 项的维数（12 个关节；三轴向量）。项名与"是否存在/尺度"一律从配置文件读。
TERM_DIMS = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "joint_pos": 12,
    "joint_vel": 12,
    "actions": 12,
}
POLICY_TERMS = ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"]
POLICY_SCALES = {"base_ang_vel": 0.25, "projected_gravity": 1.0, "velocity_commands": 1.0,
                 "joint_pos": 1.0, "joint_vel": 0.05, "actions": 1.0}
CRITIC_TERMS = ["base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                "joint_pos", "joint_vel", "actions", "height_scan"]
TERRAIN_TERMS = ["height_scan"]
# 部署 config.yaml 的观测名 ↔ 训练侧项名
DEPLOY_OBS_NAMES = {
    "base_ang_vel": "ang_vel",
    "projected_gravity": "gravity_vec",
    "velocity_commands": "commands",
    "joint_pos": "dof_pos",
    "joint_vel": "dof_vel",
    "actions": "actions",
}
JOINT_ORDER = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "shank")]
# 组级开关（缩进 4、行内标量或 null），不是 obs 项
GROUP_FLAGS = {"concatenate_terms", "concatenate_dim", "enable_corruption", "history_length", "flatten_history_dim"}


@dataclass
class Check:
    ok: bool
    label: str
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, ok: bool, label: str, detail: str = "") -> None:
        check = Check(bool(ok), label, detail)
        self.checks.append(check)
        mark = "✅" if check.ok else "❌"
        line = f"  {mark} {label}"
        if check.detail:
            line += f"\n       {check.detail}"
        print(line)

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


# --------------------------------------------------------------------------------------
# 读取 helpers（yaml 只支持本项目这些 dump 形态；解析失败一律报 ❌，不静默跳过）
# --------------------------------------------------------------------------------------
def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def block_range(lines: list[str], key: str, indent: int, start: int = 0) -> tuple[int, int] | None:
    """返回 `key:`（恰好缩进 indent）所在块的 [start, end) 行号范围。"""
    pattern = re.compile(rf"^{' ' * indent}{re.escape(key)}:\s*$")
    for index in range(start, len(lines)):
        if pattern.match(lines[index]):
            end = index + 1
            while end < len(lines):
                if lines[end].strip() and indent_of(lines[end]) <= indent:
                    break
                end += 1
            return index, end
    return None


def scalar_in_block(lines: list[str], key: str, start: int, end: int, indent: int) -> str | None:
    pattern = re.compile(rf"^{' ' * indent}{re.escape(key)}:\s*(.*)$")
    for index in range(start, end):
        match = pattern.match(lines[index])
        if match:
            return match.group(1).strip()
    return None


def parse_obs_groups(text: str) -> dict[str, list[tuple[str, str | None, bool]]]:
    """`observations:` → {group: [(term, scale, is_null), ...]}。

    组在缩进 2、项在缩进 4、项内键在缩进 6；`key: value` 是组级开关（如 concatenate_terms），
    只有 `key:`（无值）才是项块，`key: null` 是显式置空的项。
    """
    lines = text.splitlines()
    groups: dict[str, list[tuple[str, str | None, bool]]] = {}
    obs = block_range(lines, "observations", 0)
    if obs is None:
        return groups
    obs_start, obs_end = obs
    index = obs_start + 1
    while index < obs_end:
        line = lines[index]
        if line.strip() and indent_of(line) == 2 and line.rstrip().endswith(":"):
            group = line.strip()[:-1]
            group_start, group_end = index, index + 1
            while group_end < obs_end:
                if lines[group_end].strip() and indent_of(lines[group_end]) <= 2:
                    break
                group_end += 1
            terms: list[tuple[str, str | None, bool]] = []
            for row in range(group_start + 1, group_end):
                candidate = lines[row]
                if not candidate.strip() or indent_of(candidate) != 4:
                    continue
                null_match = re.match(r"^\s{4}([\w]+):\s*null\s*$", candidate)
                if null_match:
                    if null_match.group(1) not in GROUP_FLAGS:
                        terms.append((null_match.group(1), None, True))
                    continue
                if not candidate.rstrip().endswith(":"):
                    continue  # 组级开关，如 enable_corruption: true
                term = candidate.strip()[:-1]
                if term in GROUP_FLAGS:
                    continue
                term_start, term_end = row, row + 1
                while term_end < group_end:
                    if lines[term_end].strip() and indent_of(lines[term_end]) <= 4:
                        break
                    term_end += 1
                terms.append((term, scalar_in_block(lines, "scale", term_start, term_end, 6), False))
            groups[group] = terms
            index = group_end
        else:
            index += 1
    return groups


def parse_scanner_geometry(text: str, scanner: str = "height_scanner") -> tuple[float, tuple[float, float]] | None:
    lines = text.splitlines()
    scene = block_range(lines, "scene", 0)
    if scene is None:
        return None
    entry = block_range(lines, scanner, 2, start=scene[0])
    if entry is None:
        return None
    start, end = entry
    pattern = block_range(lines, "pattern_cfg", 4, start=start)
    if pattern is None or pattern[0] >= end:
        return None
    p_start, p_end = pattern
    resolution = scalar_in_block(lines, "resolution", p_start, p_end, 6)
    size_start = None
    for row in range(p_start, p_end):
        if re.match(r"^\s{6}size:\s*", lines[row]):
            size_start = row
            break
    if resolution is None or size_start is None:
        return None
    values: list[float] = []
    for row in range(size_start + 1, p_end):
        match = re.match(r"^\s*-\s*([-0-9.eE]+)\s*$", lines[row])
        if match:
            values.append(float(match.group(1)))
        elif lines[row].strip() and indent_of(lines[row]) <= 6:
            break
    if len(values) != 2:
        return None
    return float(resolution), (values[0], values[1])


def ray_count(resolution: float, size: tuple[float, float]) -> tuple[int, int, int]:
    """镜像 `grid_pattern` 的 `arange(start, end + 1e-9, step)` 计数（同 CMoE_env_cfg 的写法）。"""
    rows = math.floor(size[0] / resolution + 1.0e-9) + 1
    cols = math.floor(size[1] / resolution + 1.0e-9) + 1
    return rows, cols, rows * cols


def parse_simple_yaml(text: str) -> dict[str, str]:
    """部署 config.yaml 的子集解析：标量 + 方括号列表（可跨行），去掉行内注释。

    只认 ``key: value`` 形态的叶子键（缩进不敏感），`imgo2/ppo:` 这类带 ``/`` 的父键自然被跳过。
    """
    out: dict[str, str] = {}
    lines = [re.sub(r"#.*$", "", line) for line in text.splitlines()]
    index = 0
    while index < len(lines):
        line = lines[index]
        match = re.match(r"^\s*([A-Za-z_][\w]*):\s*(.*)$", line)
        if not match:
            index += 1
            continue
        key, value = match.group(1), match.group(2).strip()
        if value.startswith("[") and value.count("[") > value.count("]"):
            while index + 1 < len(lines) and value.count("[") > value.count("]"):
                index += 1
                value += " " + lines[index].strip()
        out[key] = re.sub(r"\s+", " ", value).strip()
        index += 1
    return out


def as_floats(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(v) for v in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", value)]


def as_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return re.findall(r'"([^"]*)"', value)


def as_float(value: str | None) -> float | None:
    numbers = as_floats(value)
    return numbers[0] if numbers else None


def find_scalar(text: str, pattern: str, label: str, report: Report) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        report.add(False, f"源码里找不到 {label}", f"pattern={pattern}")
        return None
    return match.group(1).strip()


OFFLINE_PKG = "_rl_lab_offline"


def import_offline(relative: str):
    """在**不引导 Isaac Sim**的前提下导入 `rl_lab` 的子模块。

    `rl_lab/modules/__init__.py` → `actor_critic_recurrent` → `rl_lab.utils` →
    `export_deploy_cfg` → `isaaclab` → `omni.log`，而 `omni.log` 只在 Isaac Sim 启动后
    才存在；`rl_lab/utils/__init__.py` 同理。离线核对只需要 `modules/cmoe_actor_critic.py`
    与 `utils/pretrained_prior.py`，所以把 `rl_lab/` 目录挂到**私有包名** `_rl_lab_offline`
    下导入：相对 import（`from .actor_critic import ...`）照常解析，而真实的
    `rl_lab.*` 命名空间一个字节都不动（否则会污染同进程里其它测试的导入）。

    Args:
        relative: 相对 `rl_lab/` 的模块路径，如 ``"modules.cmoe_actor_critic"``。
    """
    import importlib
    import types

    if OFFLINE_PKG not in sys.modules:
        root = types.ModuleType(OFFLINE_PKG)
        root.__path__ = [str(RL_LAB_ROOT / "rl_lab")]
        sys.modules[OFFLINE_PKG] = root
        # 给**每个**子包都塞 stub（不只是需要的两个）：这样按名导入时不会连锁执行
        # `modules/__init__.py` → `rl_lab.utils` → `export_deploy_cfg` → `isaaclab` 那条链。
        subpackages = sorted(
            path.name for path in (RL_LAB_ROOT / "rl_lab").iterdir()
            if path.is_dir() and (path / "__init__.py").is_file() and not path.name.startswith("__")
        )
        for sub in subpackages:
            stub = types.ModuleType(f"{OFFLINE_PKG}.{sub}")
            stub.__path__ = [str(RL_LAB_ROOT / "rl_lab" / sub)]
            sys.modules[stub.__name__] = stub
            setattr(root, sub, stub)
    return importlib.import_module(f"{OFFLINE_PKG}.{relative}")


def newest_env_yaml() -> Path | None:
    candidates: list[Path] = []
    for tree in LOG_TREES:
        candidates.extend(tree.glob("cmoe/*/*/params/env.yaml"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def discover_priors() -> list[Path]:
    """默认先验：最新轮数的 AMP checkpoint（优先 gait 记录里那份）+ 部署 PPO 导出件。"""
    priors: list[Path] = []
    recorded: Path | None = None
    gait_json = REPO_ROOT / "docs/gait_eval_amp_24500.json"
    if gait_json.is_file():
        try:
            recorded = Path(json.loads(gait_json.read_text(encoding="utf-8"))["checkpoint"])
        except Exception:  # noqa: BLE001 - 记录文件坏了就退回扫描
            recorded = None
    if recorded is not None and recorded.is_file():
        priors.append(recorded)
    else:
        best: tuple[int, Path] | None = None
        for tree in LOG_TREES:
            for path in tree.glob("amp_rsl_rl/*/*/model_*.pt"):
                match = re.search(r"model_(\d+)\.pt$", path.name)
                if match and (best is None or int(match.group(1)) > best[0]):
                    best = (int(match.group(1)), path)
        if best is not None:
            priors.append(best[1])
    ppo = REPO_ROOT / "imgo2_deploy/policy/imgo2/ppo/policy.pt"
    if ppo.is_file():
        priors.append(ppo)
    return priors


# --------------------------------------------------------------------------------------
# 1. 契约层
# --------------------------------------------------------------------------------------
def check_run_yaml(path: Path, report: Report) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    report.section("1. 在跑 run 自带的配置（契约的真实来源）")
    print(f"  run env.yaml : {path}")

    groups = parse_obs_groups(text)

    policy = [(name, scale) for name, scale, is_null in groups.get("policy", []) if not is_null]
    policy_names = [name for name, _ in policy]
    report.add(policy_names == POLICY_TERMS, "policy 组项与顺序",
               f"实际={policy_names}")
    report.add(policy_names == POLICY_TERMS and any(
        name in {"base_lin_vel", "height_scan"} for name, _, _ in groups.get("policy", [])),
        "policy 组不含特权项（base_lin_vel / height_scan 显式 null）",
        f"实际组内项={[n for n, _, _ in groups.get('policy', [])]}")
    scales_ok = {}
    for name, scale in policy:
        try:
            scales_ok[name] = abs(float(scale) - POLICY_SCALES[name]) < 1e-9  # type: ignore[arg-type]
        except (TypeError, ValueError):
            scales_ok[name] = False
    report.add(all(scales_ok.values()), "policy 组尺度",
               f"实际={dict(policy)}；期望={POLICY_SCALES}")

    terrain = [(name, scale) for name, scale, is_null in groups.get("terrain", []) if not is_null]
    report.add([n for n, _ in terrain] == TERRAIN_TERMS, "terrain 组项",
               f"实际={terrain}")

    critic = [(name, scale) for name, scale, is_null in groups.get("critic", []) if not is_null]
    critic_names = [name for name, _ in critic]
    critic_scales_ok = all(abs(float(s) - 1.0) < 1e-9 for _, s in critic if s is not None)
    report.add(critic_names == CRITIC_TERMS and critic_scales_ok, "critic 组项/顺序/尺度（全 1.0）",
               f"实际项={critic_names}，尺度={[s for _, s in critic]}")

    geometry = parse_scanner_geometry(text)
    terrain_dim = 0
    if geometry is None:
        report.add(False, "height_scanner 扫描几何", "params/env.yaml 里没解析到 pattern_cfg(resolution/size)")
    else:
        resolution, size = geometry
        rows, cols, terrain_dim = ray_count(resolution, size)
        report.add(terrain_dim == 77, "height_scanner 射线数 = 77",
                   f"resolution={resolution}, size={size} ⇒ {rows}x{cols} = {terrain_dim}")
    report.add(len(terrain) == 1 and terrain_dim > 0, "terrain 组维数 == 射线数",
               f"terrain 项={[n for n, _ in terrain]}，射线={terrain_dim}")

    one_step = sum(TERM_DIMS[name] for name in policy_names if name in TERM_DIMS)
    critic_dim = sum(TERM_DIMS[name] for name in critic_names if name in TERM_DIMS) + terrain_dim
    report.add(one_step == 45, "num_one_step_obs（当前帧本体感知）", f"={one_step}")
    report.add(critic_dim == 125, "critic 维数", f"= 3 + {one_step} + {terrain_dim} = {critic_dim}")

    agent = path.parent / "agent.yaml"
    history_steps = num_experts = None
    explicit_dim = state_latent = terrain_latent = None
    actor_hidden: list[int] = []
    if agent.is_file():
        agent_text = agent.read_text(encoding="utf-8", errors="replace")
        history_steps = find_scalar(agent_text, r"^history_steps:\s*(\d+)", "agent.yaml history_steps", report)
        num_experts = find_scalar(agent_text, r"^\s+num_experts:\s*(\d+)", "agent.yaml num_experts", report)
        explicit_dim = find_scalar(agent_text, r"^\s+explicit_dim:\s*(\d+)", "agent.yaml explicit_dim", report)
        state_latent = find_scalar(agent_text, r"^\s+state_latent_dim:\s*(\d+)", "agent.yaml state_latent_dim", report)
        terrain_latent = find_scalar(agent_text, r"^\s+terrain_latent_dim:\s*(\d+)", "agent.yaml terrain_latent_dim", report)
        hidden_block = re.search(r"actor_hidden_dims:\n((?:\s*-\s*\d+\n)+)", agent_text)
        if hidden_block:
            actor_hidden = [int(v) for v in re.findall(r"-\s*(\d+)", hidden_block.group(1))]
        report.add(bool(actor_hidden), "agent.yaml actor_hidden_dims", f"={actor_hidden}")
    else:
        report.add(False, "run 自带的 agent.yaml", f"缺失：{agent}")

    expert_dim = None
    actor_dim = None
    if None not in (history_steps, explicit_dim, state_latent, terrain_latent):
        expert_dim = one_step + int(explicit_dim) + int(state_latent) + terrain_dim + int(terrain_latent)
        actor_dim = int(history_steps) * one_step + terrain_dim
        report.add(expert_dim == 157, "expert / gate 输入维数",
                   f"= {one_step} + {explicit_dim}(explicit) + {state_latent}(state) + {terrain_dim}(terrain) "
                   f"+ {terrain_latent}(terrain latent) = {expert_dim}")
        report.add(actor_dim == 527, "actor 总输入维数",
                   f"= {history_steps} * {one_step} + {terrain_dim} = {actor_dim}")

    print(f"  派生契约     : one_step={one_step}, terrain={terrain_dim}, critic={critic_dim}, "
          f"expert={expert_dim}, actor={actor_dim}, history={history_steps}, experts={num_experts}, "
          f"hidden={actor_hidden}")
    return {
        "one_step": one_step,
        "terrain": terrain_dim,
        "critic": critic_dim,
        "expert": expert_dim or 0,
        "actor": actor_dim or 0,
        "history": int(history_steps or 0),
        "experts": int(num_experts or 0),
        "hidden": actor_hidden,
    }


def check_source_formula(report: Report) -> None:
    report.section("2. 源码里的结构（切片与维数公式）")
    text = CMOE_AC.read_text(encoding="utf-8")
    block = re.search(r"self\.actor_input_dim\s*=\s*\(([^)]*)\)", text)
    if block is None:
        report.add(False, "cmoe_actor_critic.py 的 actor_input_dim 公式", "没找到赋值")
    else:
        terms = re.findall(r"\b(num_one_step_obs|explicit_dim|state_latent_dim|terrain_obs_dim|terrain_latent_dim)\b",
                           block.group(1))
        report.add(terms == ["num_one_step_obs", "explicit_dim", "state_latent_dim", "terrain_obs_dim",
                             "terrain_latent_dim"],
                   "expert 输入 = 本体 + explicit + state latent + terrain + terrain latent",
                   f"实际公式项={terms}")
    report.add(bool(re.search(r"self\.terrain_obs_start\s*=\s*self\.history_obs_dim", text)),
               "地形通道从 history 之后开始（terrain_obs_start = history_obs_dim）")
    report.add(bool(re.search(r"observations\[:,\s*:\s*self\.num_one_step_obs\]", text)),
               "专家拿的是**当前帧**本体感知（observations[:, :num_one_step_obs]）")
    env_cfg_text = CMAOE_CFG.read_text(encoding="utf-8")
    report.add(bool(re.search(r"math\.floor\([^)]*1\.0e-9", env_cfg_text)),
               "77 条射线的 arange 容差写法（math.floor(... + 1.0e-9)）在 CMoE_env_cfg 里，本脚本镜像同一语义")


def check_contract_vs_deploy(report: Report) -> dict[str, object]:
    report.section("3. 训练侧契约 vs 部署 config.yaml")
    rough = ROUGH_CFG.read_text(encoding="utf-8")
    amp = AMP_CFG.read_text(encoding="utf-8")
    asset = ASSET_CFG.read_text(encoding="utf-8")

    training = {
        "ang_vel_scale": find_scalar(rough, r"observations\.policy\.base_ang_vel\.scale\s*=\s*([-0-9.]+)",
                                     "policy.base_ang_vel.scale", report),
        "dof_pos_scale": find_scalar(rough, r"observations\.policy\.joint_pos\.scale\s*=\s*([-0-9.]+)",
                                     "policy.joint_pos.scale", report),
        "dof_vel_scale": find_scalar(rough, r"observations\.policy\.joint_vel\.scale\s*=\s*([-0-9.]+)",
                                     "policy.joint_vel.scale", report),
    }
    report.add(bool(re.search(r"observations\.policy\.base_lin_vel\s*=\s*None", rough)),
               "训练侧 policy 不含 base_lin_vel（rough_env_cfg.py）")
    report.add(bool(re.search(r"observations\.policy\.height_scan\s*=\s*None", rough)),
               "训练侧 policy 不含 height_scan（rough_env_cfg.py）")
    report.add(bool(re.search(r"observations\.critic\.height_scan\s*=\s*None", amp)),
               "AMP critic 不含 height_scan（amp_env_cfg.py）⇒ 48 维正是 CMoE critic 的前 48 维")

    action_block = re.search(r"actions\.joint_pos\.scale\s*=\s*\{([^}]*)\}", rough)
    hip_scale = other_scale = None
    if action_block:
        hip = re.search(r'"\.\*_hip_joint":\s*([-0-9.]+)', action_block.group(1))
        other = re.search(r'"\^\(\?!\.\*_hip_joint\)\.\*":\s*([-0-9.]+)', action_block.group(1))
        hip_scale = hip.group(1) if hip else None
        other_scale = other.group(1) if other else None
    report.add(hip_scale == "0.125" and other_scale == "0.25", "动作 scale：髋 0.125 / 其余 0.25",
               f"实际 hip={hip_scale}, other={other_scale}")
    clip = re.search(r"actions\.joint_pos\.clip\s*=\s*\{\"\.\*\":\s*\(([-0-9.]+),\s*([-0-9.]+)\)\}", rough)
    clip_ok = clip is not None and float(clip.group(1)) == -3.0 and float(clip.group(2)) == 3.0
    report.add(clip_ok, "动作 clip ±3", f"实际={clip.groups() if clip else None}")

    default_pose = {}
    for part in ("hip", "thigh", "shank"):
        value = find_scalar(asset, rf'"\.\*_{part}_joint":\s*([-0-9.]+)', f"asset {part} 默认角", report)
        default_pose[part] = value
    report.add(default_pose == {"hip": "0.0", "thigh": "0.87", "shank": "-1.82"},
               "默认姿态 0 / 0.87 / −1.82（assets/imgo2.py）", f"实际={default_pose}")
    stiffness = find_scalar(asset, r"^\s*stiffness=([-0-9.]+)", "asset stiffness", report)
    damping = find_scalar(asset, r"^\s*damping=([-0-9.]+)", "asset damping", report)
    report.add(stiffness == "25.0" and damping == "0.5", "执行器增益 kp 25 / kd 0.5",
               f"实际 stiffness={stiffness}, damping={damping}")

    joint_names_block = re.search(r"joint_names\s*=\s*\[([^\]]*)\]", rough, re.DOTALL)
    joint_names = re.findall(r'"([^"]+)"', joint_names_block.group(1)) if joint_names_block else []
    report.add(joint_names == JOINT_ORDER, "关节顺序 FL/FR/RL/RR × hip/thigh/shank",
               f"实际={joint_names}")

    deploy_reports: dict[str, object] = {}
    for name, path in DEPLOY_CFGS.items():
        if not path.is_file():
            report.add(True, f"部署 {name} config.yaml 不存在（跳过）", f"{path}")
            continue
        config = parse_simple_yaml(path.read_text(encoding="utf-8"))
        obs = as_strings(config.get("observations"))
        expected_obs = [DEPLOY_OBS_NAMES[t] for t in POLICY_TERMS]
        report.add(obs == expected_obs, f"[{name}] 部署观测顺序 == 训练侧 policy 组",
                   f"部署={obs}；期望={expected_obs}")
        num_obs = as_float(config.get("num_observations"))
        report.add(num_obs == 45.0, f"[{name}] num_observations = 45", f"={num_obs}")
        for key, train_key in (("ang_vel_scale", "ang_vel_scale"), ("dof_pos_scale", "dof_pos_scale"),
                               ("dof_vel_scale", "dof_vel_scale")):
            deploy_value = as_float(config.get(key))
            train_value = float(training[train_key]) if training[train_key] else None
            report.add(deploy_value is not None and train_value is not None
                       and abs(deploy_value - train_value) < 1e-9,
                       f"[{name}] {key} 与训练侧一致",
                       f"部署={deploy_value}，训练={train_value}")
        commands_scale = as_floats(config.get("commands_scale"))
        report.add(commands_scale == [1.0, 1.0, 1.0], f"[{name}] commands_scale = [1,1,1]",
                   f"={commands_scale}")
        report.add(as_float(config.get("lin_vel_scale")) is not None,
                   f"[{name}] lin_vel_scale 有声明（actor 不含 base_lin_vel ⇒ 不参与比对）",
                   f"={config.get('lin_vel_scale')}")

        action_scale = as_floats(config.get("action_scale"))
        pattern_ok = bool(action_scale) and len(action_scale) == 12 and all(
            abs(v - 0.125) < 1e-9 if i % 3 == 0 else abs(v - 0.25) < 1e-9 for i, v in enumerate(action_scale))
        report.add(pattern_ok, f"[{name}] action_scale 髋 0.125 / 其余 0.25", f"={action_scale}")
        lower = as_floats(config.get("clip_actions_lower"))
        upper = as_floats(config.get("clip_actions_upper"))
        report.add(lower == [-3.0] * 12 and upper == [3.0] * 12, f"[{name}] 动作 clip ±3", f"={lower} / {upper}")
        pose = as_floats(config.get("default_dof_pos"))
        pose_ok = bool(pose) and len(pose) == 12 and all(
            abs(pose[i] - [0.0, 0.87, -1.82][i % 3]) < 1e-6 for i in range(12))
        report.add(pose_ok, f"[{name}] default_dof_pos == 训练侧默认姿态", f"={pose}")
        kp = as_floats(config.get("rl_kp"))
        kd = as_floats(config.get("rl_kd"))
        report.add(kp == [25.0] * 12 and kd == [0.5] * 12, f"[{name}] rl_kp/rl_kd = 25/0.5",
                   f"kp={kp} kd={kd}")
        mapping = as_floats(config.get("joint_mapping"))
        report.add(mapping == [float(i) for i in range(12)], f"[{name}] joint_mapping 恒等",
                   f"={mapping}")
        report.add(as_float(config.get("num_of_dofs")) == 12.0, f"[{name}] num_of_dofs = 12",
                   f"={config.get('num_of_dofs')}")
        deploy_reports[name] = {"config": config, "obs": obs}
    return {"training": training, "deploy": deploy_reports}


# --------------------------------------------------------------------------------------
# 2. 数值层（torch）
# --------------------------------------------------------------------------------------
def check_torch(priors: list[Path], contract: dict[str, int], mode: str, report: Report) -> None:
    report.section("4. 数值层（真实 CMoE 模块 + 零填充移植）")
    try:
        import torch
    except ModuleNotFoundError as error:
        report.add(False, "torch 不可用", f"{error}（用训练环境的解释器跑，或加 --skip-torch）")
        return
    CMoEActorCritic = import_offline("modules.cmoe_actor_critic").CMoEActorCritic
    prior_module = import_offline("utils.pretrained_prior")
    install_prior = prior_module.install_prior
    load_prior = prior_module.load_prior
    teacher_actor = prior_module.teacher_actor
    teacher_critic = prior_module.teacher_critic

    one_step = contract["one_step"]
    terrain_dim = contract["terrain"]
    critic_dim = contract["critic"]
    expert_dim = contract["expert"]
    actor_dim = contract["actor"]
    history_steps = contract["history"]
    num_experts = contract["experts"]
    hidden = contract["hidden"] or [512, 256, 128]
    action_dim = TERM_DIMS["actions"]

    torch.manual_seed(0)
    observations = torch.randn(8, actor_dim)
    observations[:, :one_step] = observations[:, :one_step] * 0.5  # 尺度贴近真实观测

    installed: list[tuple[Path, object, object]] = []
    for path in priors:
        prior = load_prior(path)
        print(f"  先验 : {path}")
        print(f"         {prior.summary()}" + (f"；{'; '.join(prior.notes)}" if prior.notes else ""))
        if prior.actor_obs_dim != one_step:
            report.add(False, f"[{path.name}] 先验 actor 输入维 == 45",
                       f"实际={prior.actor_obs_dim}")
            continue
        report.add(prior.action_dim == action_dim, f"[{path.name}] 先验输出维 == 12",
                   f"实际={prior.action_dim}")

        teacher = teacher_actor(prior)
        with torch.no_grad():
            reference = teacher(observations[:, :one_step])

        fixture = CMoEActorCritic(
            num_actor_obs=actor_dim,
            num_critic_obs=critic_dim,
            num_one_step_obs=one_step,
            num_actions=action_dim,
            history_steps=history_steps,
            terrain_obs_dim=terrain_dim,
            num_experts=num_experts,
            actor_hidden_dims=list(hidden),
            critic_hidden_dims=list(hidden),
            activation="elu",
            init_noise_std=1.0,
        )
        info = install_prior(fixture, prior, mode=mode)
        entry = info["experts"][0]["actor"]
        report.add(info["installed_experts"] == (num_experts if mode == "all" else 1),
                   f"[{path.name}] 已装进 {info['installed_experts']}/{info['total_experts']} 个专家")
        report.add(entry["first_layer"] == (hidden[0], expert_dim),
                   f"[{path.name}] 第一层零填充形状",
                   f"(512,45) -> {entry['first_layer']}，置零列={entry['zeroed_columns']}")

        first_weight = fixture.experts[0].actor[0].weight.detach()
        report.add(float(first_weight[:, one_step:].abs().max().item()) == 0.0,
                   f"[{path.name}] 新增 {expert_dim - one_step} 列权重恰为 0",
                   f"max|w[:, {one_step}:]| = {float(first_weight[:, one_step:].abs().max().item())}")

        with torch.no_grad():
            # 真实路径：估计器算出的 explicit/latent + 77 维地形一起喂给专家
            actor_input = fixture.build_actor_input(observations)
            # 对照路径：把后 112 维换成随机非零值（证明"零权重"而不是"输入被丢掉"）
            random_tail = torch.cat([actor_input[:, :one_step], torch.randn_like(actor_input[:, one_step:])], dim=-1)
            output_real = fixture.experts[0].act(actor_input)
            output_rand = fixture.experts[0].act(random_tail)
        diff_real = float((output_real - reference).abs().max().item())
        diff_rand = float((output_rand - reference).abs().max().item())
        report.add(diff_real == 0.0, f"[{path.name}] 专家输出 == 先验输出（真实估计器/地形尾输入）",
                   f"max|diff| = {diff_real}")
        report.add(diff_rand == 0.0, f"[{path.name}] 后 {expert_dim - one_step} 维填随机非零后输出仍不变",
                   f"max|diff| = {diff_rand}")

        fixture.zero_grad(set_to_none=True)
        fixture.experts[0].act(random_tail).sum().backward()
        grad_tail = fixture.experts[0].actor[0].weight.grad.detach()[:, one_step:].abs().max()
        report.add(float(grad_tail.item()) > 0.0,
                   f"[{path.name}] 梯度确实流到新增列（通路接上了，不是输入被丢弃）",
                   f"max|grad[:, {one_step}:]| = {float(grad_tail.item()):.3e}")

        if prior.has_critic:
            teacher_value = teacher_critic(prior)
            critic_obs = torch.randn(8, critic_dim)
            fixture.zero_grad(set_to_none=True)
            with torch.no_grad():
                critic_out = fixture.experts[0].evaluate(critic_obs)
                critic_ref = teacher_value(critic_obs[:, : int(prior.critic_obs_dim)])
            critic_diff = float((critic_out - critic_ref).abs().max().item())
            report.add(critic_diff == 0.0,
                       f"[{path.name}] critic 零填充等价（{prior.critic_obs_dim} -> {critic_dim}）",
                       f"max|diff| = {critic_diff}")

        if prior.std is not None:
            report.add(abs(float(fixture.std.mean().item()) - float(prior.std.mean().item())) < 1e-6,
                       f"[{path.name}] 噪声 std 已随先验装入",
                       f"模型 std 均值={float(fixture.std.mean().item()):.4f}，"
                       f"先验={float(prior.std.mean().item()):.4f}（默认 init_noise_std=1.0）")
        else:
            report.add(True, f"[{path.name}] 无 std 可拷（TorchScript 导出件不含噪声）",
                       f"模型 std 均值={float(fixture.std.mean().item()):.4f}（仍是 init_noise_std）")
        installed.append((path, prior, fixture))

    # 全专家初始化 ⇒ 混合恒等于先验、与门控无关
    if mode != "all":
        report.add(True, "跳过「混合恒等于先验」检查：--mode=first 时其余专家仍是随机初始化（这是预期）")
        return
    for path, prior, fixture in installed:
        teacher = teacher_actor(prior)
        with torch.no_grad():
            reference = teacher(observations[:, :one_step])
        outputs = []
        for _ in range(2):
            with torch.no_grad():
                for parameter in fixture.gating_network.parameters():
                    parameter.data.normal_()
                outputs.append(fixture.act_inference(observations).clone())
        diff_a = float((outputs[0] - reference).abs().max().item())
        diff_b = float((outputs[1] - reference).abs().max().item())
        gate_diff = float((outputs[0] - outputs[1]).abs().max().item())
        # 单专家级别是完全逐位相等（上面已断言 0.0）；这里是 5 个相同专家的**加权和**，
        # softmax 权重之和在 float32 下是 1 ± 1e-7，所以允许 1e-5 的舍入余量。
        tolerance = 1.0e-5
        report.add(max(diff_a, diff_b, gate_diff) <= tolerance,
                   f"[{path.name}] 全专家同一先验 ⇒ 混合恒等于先验、与门控无关",
                   f"两次随机门控 max|diff vs 先验| = {diff_a:.3e} / {diff_b:.3e}，"
                   f"两次门控之间 max|diff| = {gate_diff:.3e}（float32 softmax 权重和 = 1 ± 1e-7）")


# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="离线核对 45 维先验能否移植进 CMoE 专家。")
    parser.add_argument("--run-yaml", type=Path, default=None,
                        help="某个 run 的 params/env.yaml；默认取两个 logs 树里最新的一个。")
    parser.add_argument("--prior", type=Path, action="append", default=None,
                        help="先验权重（可重复）：AMP checkpoint 或 play.py 导出的 TorchScript。"
                             "默认自动挑最新 AMP checkpoint + 部署 ppo/policy.pt。")
    parser.add_argument("--mode", choices=("all", "first"), default="all",
                        help="先验装进全部专家（默认）还是只装第 0 个。")
    parser.add_argument("--skip-torch", action="store_true", help="只查契约层，不导入 torch。")
    args = parser.parse_args()

    report = Report()
    print("CMoE 先验移植核对（离线）")
    print(f"  imgo2_rl : {ROOT}")
    print(f"  repo     : {REPO_ROOT}")

    run_yaml = args.run_yaml or newest_env_yaml()
    if run_yaml is None or not Path(run_yaml).is_file():
        report.section("0. run 配置")
        report.add(False, "找不到 run 的 params/env.yaml",
                   f"给定={args.run_yaml}，两个 logs 树里也没扫到 cmoe/*/*/params/env.yaml")
        contract = {"one_step": 45, "terrain": 77, "critic": 125, "expert": 157, "actor": 527,
                    "history": 10, "experts": 5, "hidden": [512, 256, 128]}
    else:
        contract = check_run_yaml(Path(run_yaml), report)

    check_source_formula(report)
    check_contract_vs_deploy(report)

    priors = args.prior or discover_priors()
    if not priors:
        report.section("3.5 先验权重")
        report.add(False, "没找到任何先验权重", "用 --prior 指定 AMP checkpoint 或 TorchScript 导出件")
    elif args.skip_torch:
        report.section("4. 数值层（已用 --skip-torch 跳过）")
        print("  先验 : " + ", ".join(str(p) for p in priors))
    else:
        check_torch(priors, contract, args.mode, report)

    print("\n移植配方：第一层 (hidden, 45) → (hidden, expert_dim)，列 45..expert_dim-1 置 0；"
          "critic (hidden, 48) → (hidden, 125)，列 48..124 置 0；std (12,) 直拷。")

    print(f"\n=== 汇总：{len(report.checks) - len(report.failures)}/{len(report.checks)} 项通过 ===")
    for check in report.failures:
        print(f"  ❌ {check.label}")
        if check.detail:
            print(f"     {check.detail}")
    if not report.failures:
        print("  ✅ 全部通过：先验与 CMoE 专家的前 45 维契约一致，零填充移植逐位等价。")
        print("     注意：这只说明「装得进去」，不代表先验在 Isaac 里能走（需要 step-0 教师回放）。")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
