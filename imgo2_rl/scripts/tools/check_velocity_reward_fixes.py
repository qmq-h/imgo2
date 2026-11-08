#!/usr/bin/env python3
"""Check reward syntax and adjacent rollback blocks without importing Isaac/Torch.

Optionally compare with a Git revision from before the changes:
    python3 imgo2_rl/scripts/tools/check_velocity_reward_fixes.py --baseline-ref HEAD
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
RELATIVE = "imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/mdp/rewards.py"
NAMES = ("feet_slide", "base_height_l2", "feet_height_body")


def verify(source, baseline=None):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    baseline_tree = ast.parse(baseline) if baseline is not None else None
    result = {}
    replacements = []
    for name in NAMES:
        live = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(live) != 1:
            raise ValueError(f"Expected exactly one active {name}")
        start = [i for i, line in enumerate(lines) if line.startswith(f"# BEGIN LEGACY {name} --")]
        end = [i for i, line in enumerate(lines) if line.rstrip() == f"# END LEGACY {name}"]
        if len(start) != 1 or len(end) != 1 or start[0] >= end[0]:
            raise ValueError(f"Missing/ambiguous legacy block: {name}")
        restored = []
        for line in lines[start[0] + 1:end[0]]:
            if line.startswith("# "):
                restored.append(line[2:])
            elif line.rstrip() == "#":
                restored.append("\n")
            else:
                raise ValueError(f"Non-comment line in legacy block: {name}")
        old_source = "".join(restored)
        old_tree = ast.parse(old_source)
        if len(old_tree.body) != 1 or old_tree.body[0].name != name:
            raise ValueError(f"Legacy block does not restore one {name} function")
        if "".join(lines[end[0] + 1:live[0].lineno - 1]).strip():
            raise ValueError(f"Legacy block is not adjacent to {name}")
        entry = {"active_line": live[0].lineno, "legacy_line": start[0] + 1,
                 "legacy_sha256": hashlib.sha256(old_source.encode()).hexdigest()}
        if baseline_tree is not None:
            old = next(n for n in baseline_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            expected = ast.get_source_segment(baseline, old)
            if old_source.rstrip("\n") != expected:
                raise ValueError(f"Legacy text differs from baseline: {name}")
            entry["legacy_matches_baseline"] = True
        result[name] = entry
        replacements.append((start[0], live[0].end_lineno, old_source))
    rollback_lines = lines[:]
    for start, end, old_source in sorted(replacements, reverse=True):
        rollback_lines[start:end] = [old_source]
    rollback_tree = ast.parse("".join(rollback_lines))
    if baseline_tree is not None and ast.dump(rollback_tree) != ast.dump(baseline_tree):
        raise ValueError("Rollback AST differs from baseline outside the archived changes")
    return {"scope": "Standard-library syntax/rollback checks; no Isaac or dynamics",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "syntax": True, "rollback_syntax": True, "functions": result,
            "rollback_matches_baseline_ast": True if baseline is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", help="Git revision before the modifications")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    baseline = None
    if args.baseline_ref:
        baseline = subprocess.run(
            ["git", "show", f"{args.baseline_ref}:{RELATIVE}"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout
    report = verify((ROOT / RELATIVE).read_text(encoding="utf-8-sig"), baseline)
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
