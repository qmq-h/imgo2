#!/usr/bin/env python3
"""Static PPO rough weight/selector audit; does not import Isaac or run simulation."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity"


def tree(path):
    result = ast.parse(path.read_text(encoding="utf-8-sig"))
    compile(result, str(path), "exec")
    return result


def saved_weights(path):
    """Read just the known YAML rewards/weight indentation, without object loading."""
    result, inside, key = {}, False, None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line == "rewards:":
            inside = True
            continue
        if inside and line and not line.startswith(" "):
            break
        if not inside:
            continue
        match = re.fullmatch(r"  (\w+):(?:\s+(?:null|None))?\s*", line)
        if match:
            key = match.group(1)
            result[key] = 0.0
        match = re.fullmatch(r"    weight: (.+)", line)
        if match:
            result[key] = float(match.group(1))
    if not result:
        raise ValueError("No rewards section found in saved env.yaml")
    return result


def saved_height_parameters(path):
    """Read scalar reward target_height/threshold fields, ignoring YAML objects."""
    result, inside, key = {}, False, None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line == "rewards:":
            inside = True
            continue
        if inside and line and not line.startswith(" "):
            break
        if not inside:
            continue
        match = re.fullmatch(r"  (\w+):(?:\s+(?:null|None))?\s*", line)
        if match:
            key = match.group(1)
        match = re.fullmatch(r"      (target_height|threshold): (.+)", line)
        if match:
            result.setdefault(key, {})[match.group(1)] = float(match.group(2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-env", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    source = BASE / "base_move/rough_env_cfg.py"
    common = BASE / "velocity_env_cfg.py"
    weights, lines = {}, {}
    for cls in tree(common).body:
        if isinstance(cls, ast.ClassDef) and cls.name == "RewardsCfg":
            for assignment in cls.body:
                if isinstance(assignment, ast.Assign) and isinstance(assignment.value, ast.Call):
                    for kw in assignment.value.keywords:
                        if kw.arg == "weight":
                            weights[assignment.targets[0].id] = ast.literal_eval(kw.value)
    source_tree = tree(source)
    cls = next(n for n in source_tree.body if isinstance(n, ast.ClassDef) and n.name == "Imgo2RoughEnvCfg")
    attributes = {a.targets[0].id: ast.literal_eval(a.value) for a in cls.body
                  if isinstance(a, ast.Assign) and isinstance(a.targets[0], ast.Name)}
    hip_patterns, mirror_pairs, foot_patterns = None, None, None
    height_parameters = {}
    for node in ast.walk(cls):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = ast.unparse(target)
                if re.fullmatch(r"self.rewards.\w+.weight", name):
                    key = name.split(".")[2]
                    weights[key] = ast.literal_eval(node.value)
                    lines[key] = node.lineno
                if (isinstance(target, ast.Subscript)
                        and re.fullmatch(r"self.rewards.\w+.params", ast.unparse(target.value))):
                    field = ast.literal_eval(target.slice)
                    if field in ("target_height", "threshold"):
                        height_parameters.setdefault(name.split(".")[2], {})[field] = ast.literal_eval(node.value)
                if name.startswith("self.rewards.joint_deviation_l1.params[") and name.endswith(".joint_names"):
                    hip_patterns = ast.literal_eval(node.value)
                if (isinstance(target, ast.Subscript) and name.startswith("self.rewards.joint_mirror.params[")
                        and ast.literal_eval(target.slice) == "mirror_joints"):
                    mirror_pairs = ast.literal_eval(node.value)
                if name.startswith("self.rewards.feet_height_body.params[") and name.endswith(".body_names"):
                    foot_patterns = [attributes[n.attr] if isinstance(n, ast.Attribute)
                                     else ast.literal_eval(n) for n in node.value.elts]
    if hip_patterns is None or mirror_pairs is None or foot_patterns is None:
        raise ValueError("Required selector declarations not found")

    asset = ROOT / "imgo2_rl/source/imgo2_rl/imgo2_rl/assets/imgo2.py"
    def asset_path(node):
        if isinstance(node, ast.Name) and node.id == "_REPO_ROOT":
            return ROOT
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return asset_path(node.left) / ast.literal_eval(node.right)
        raise ValueError("Unsupported default URDF path expression")
    model = None
    for node in tree(asset).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_DEFAULT_URDF_PATH"
            for target in node.targets
        ):
            model = asset_path(node.value)
    if model is None:
        raise ValueError("No default URDF path declaration")
    xml = ET.parse(model).getroot()
    joints = [n.attrib["name"] for n in xml.findall("joint") if n.attrib["type"] != "fixed"]
    patterns = hip_patterns + [p for pair in mirror_pairs for p in pair]
    matches = {p: [n for n in joints if re.fullmatch(p, n)] for p in patterns}
    feet = [n.attrib["name"] for n in xml.findall("link")
            if any(re.fullmatch(p, n.attrib["name"]) for p in foot_patterns)]
    selected_hips = {n for p in hip_patterns for n in matches[p]}
    checks = {"syntax": True, "hip_count_4": len(selected_hips) == 4,
              "feet_count_4": len(feet) == 4,
              "mirror_each_two": all(len(matches[p]) == 2 for pair in mirror_pairs for p in pair),
              "declared_action_joint_set": set(attributes["joint_names"]) == set(joints)}
    files = [source, common, BASE / "mdp/rewards.py", BASE / "mdp/commands.py", asset, model]
    report = {"scope": "Static declared weights/selectors and analytic examples only; no Isaac instantiation, CLI overrides or rollout.",
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "active_weights": {k: v for k, v in weights.items() if v != 0},
              "disabled_zero_weights": [k for k, v in weights.items() if v == 0],
              "source_lines": lines, "selector_matches": matches, "feet": feet, "checks": checks}
    report["declared_height_parameters"] = height_parameters
    if args.saved_env:
        saved = saved_weights(args.saved_env)
        report["saved_env"] = str(args.saved_env)
        report["saved_env_sha256"] = hashlib.sha256(args.saved_env.read_bytes()).hexdigest()
        report["saved_weight_differences"] = {
            k: {"current_source": weights.get(k, 0), "saved": saved.get(k, 0)}
            for k in sorted(set(weights) | set(saved)) if weights.get(k, 0) != saved.get(k, 0)
        }
        parameters = saved_height_parameters(args.saved_env)
        report["saved_parameter_differences"] = {
            key + "." + field: {"current_source": value, "saved": parameters.get(key, {}).get(field)}
            for key, fields in height_parameters.items() for field, value in fields.items()
            if (weights.get(key, 0) or saved.get(key, 0)) and value != parameters.get(key, {}).get(field)
        }
    report["analytic_examples"] = []
    for height in (0.30, 0.24):
        raw = (0.043 - height + 0.20) ** 2 * math.tanh(2 * 0.30)
        report["analytic_examples"].append({"base_height": height, "stance_foot_link_world_z": 0.043,
            "body_z": 0.043 - height, "relative_speed": 0.30, "raw_single_foot": raw,
            "historical_weight_minus6_step_reward": -6 * raw * 0.02,
            "current_step_reward": weights["feet_height_body"] * raw * 0.02})
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.json:
        args.json.write_text(output)
        print(f"checks={checks}; active={len(report['active_weights'])}; output={args.json}")
    else:
        print(output, end="")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
