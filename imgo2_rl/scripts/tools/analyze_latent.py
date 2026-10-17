#!/usr/bin/env python3
"""离线检查上层 decoder：显式估计误差 + latent 是否按工况聚类（纯标准库）。

为什么需要它：`play_towing_test.py --dump-latent` 会把 decoder 的 16 维 latent 与 6 维显式估计
**按 case/阶段汇总**写进 `report.json` 的 `metrics["latent"]` / `metrics["estimator"]`。
latent 无法从 `tow.csv` 事后重建（policy 帧里的 `last_action`/`last_loco_action`/`joint_vel`
都没有记录），所以这个检查只能吃带 `--dump-latent` 的产物。

用法：

    python3 scripts/tools/analyze_latent.py --report imgo2_rl/logs/towing/play_test/<run>/report.json
    python3 scripts/tools/analyze_latent.py --report .../report.json --phase coast --pcs 3

它回答两件事：

1. **decoder 估计得准不准**（6 维输出里可比真值的 4 项：机体系 vx / vy、小车质量、‖F‖ 对绳张力）；
2. **latent 有没有按工况分开**：对每个分组（连接类型 / 质量档 / 坡度量级 / 速度档）给出
   η²（组间方差占比）与「留一最近质心」分类准确率（对照多数类基线）。
   注意 latent 坐标**没有物理含义**（decoder 的 docstring 明说），所以这里只看"分组能不能被分开"，
   不解释单个坐标。

linalg 全部手写（幂迭代求协方差的主特征向量），不依赖 numpy/torch —— 与仓库其它离线工具一致。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

#: 默认看牵引段（负载最明确、且 offset 头参与最少）
DEFAULT_PHASE = "tow"
#: 分组维度：`case` 字典里的键 → 显示名
GROUPINGS = (
    ("connection", "连接类型"),
    ("cart_mass_kg", "质量档"),
    ("grade_deg", "坡度量级"),
    ("velocity_mps", "速度档"),
)
#: `metrics["estimator"]["abs_err"]` 的四个量
#: 2026-10-16 契约 v4 起 decoder 不再输出质量 ⇒ 估算项只留这 3 个
ESTIMATOR_ITEMS = ("vx_mps", "vy_mps", "force_norm_n")


# ------------------------------------------------------------------ 线性代数（手写）
def _standardize(rows):
    """按列标准化；常数列（方差 0）保留为 0。返回 (标准化矩阵, 每列均值, 每列标准差)。"""
    n = len(rows)
    width = len(rows[0])
    means = [sum(row[j] for row in rows) / n for j in range(width)]
    scales = []
    for j in range(width):
        var = sum((row[j] - means[j]) ** 2 for row in rows) / n
        scales.append(math.sqrt(var) if var > 1e-12 else 0.0)
    matrix = [[((row[j] - means[j]) / scales[j]) if scales[j] else 0.0 for j in range(width)]
              for row in rows]
    return matrix, means, scales


def _covariance(matrix):
    n = len(matrix)
    width = len(matrix[0])
    cov = [[0.0] * width for _ in range(width)]
    for row in matrix:
        for a in range(width):
            if row[a] == 0.0:
                continue
            for b in range(a, width):
                cov[a][b] += row[a] * row[b]
    for a in range(width):
        for b in range(a, width):
            cov[a][b] /= n
            cov[b][a] = cov[a][b]
    return cov


def _matvec(cov, vector):
    return [sum(cov[i][j] * vector[j] for j in range(len(vector))) for i in range(len(cov))]


def _norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def principal_components(matrix, count, *, iterations=200, tol=1e-10):
    """幂迭代 + 收缩求前 `count` 个主成分。返回 [(特征值, 特征向量), ...]（降序）。"""
    width = len(matrix[0])
    cov = _covariance(matrix)
    total = sum(cov[i][i] for i in range(width))
    components = []
    deflated = [row[:] for row in cov]
    for index in range(min(count, width)):
        vector = [1.0 / math.sqrt(width)] * width
        if index:  # 后续成分换个起点，避开与前一个完全正交的退化方向
            vector = [(-1.0 if i % 2 else 1.0) for i in range(width)]
            norm = _norm(vector)
            vector = [value / norm for value in vector]
        eigenvalue = 0.0
        for _ in range(iterations):
            product = _matvec(deflated, vector)
            length = _norm(product)
            if length <= 1e-15:
                break
            candidate = [value / length for value in product]
            if _norm([candidate[i] - vector[i] for i in range(width)]) < tol:
                vector = candidate
                break
            vector = candidate
        eigenvalue = sum(vector[i] * _matvec(deflated, vector)[i] for i in range(width))
        components.append((eigenvalue, vector))
        # 收缩：去掉这个方向
        for a in range(width):
            for b in range(width):
                deflated[a][b] -= eigenvalue * vector[a] * vector[b]
    return components, total


def eta_squared(values, labels):
    """组间方差 / 总方差（0 = 分组无区分度，1 = 完全分开）。低于 1e-12 的总方差返回 0。"""
    n = len(values)
    if n == 0:
        return 0.0
    grand = sum(values) / n
    total = sum((value - grand) ** 2 for value in values)
    if total <= 1e-12:
        return 0.0
    groups = {}
    for value, label in zip(values, labels):
        groups.setdefault(label, []).append(value)
    between = sum(len(items) * (sum(items) / len(items) - grand) ** 2 for items in groups.values())
    return between / total


def nearest_centroid_loo(points, labels):
    """留一最近质心分类：返回 (准确率, 多数类基线, 样本数)。"""
    n = len(points)
    if n == 0:
        return 0.0, 0.0, 0
    unique = sorted(set(labels), key=lambda value: str(value))
    counts = {label: labels.count(label) for label in unique}
    majority = max(counts.values()) / n
    # 每个（留一后的）组质心 = (总和 − 该点) / (计数 − 1)
    sums = {label: [0.0] * len(points[0]) for label in unique}
    for point, label in zip(points, labels):
        for j, value in enumerate(point):
            sums[label][j] += value
    correct = 0
    for index, (point, label) in enumerate(zip(points, labels)):
        best, best_label = None, None
        for candidate in unique:
            count = counts[candidate] - (1 if candidate == label else 0)
            if count <= 0:
                continue
            centroid = [(sums[candidate][j] - (point[j] if candidate == label else 0.0)) / count
                        for j in range(len(point))]
            distance = sum((point[j] - centroid[j]) ** 2 for j in range(len(point)))
            if best is None or distance < best:
                best, best_label = distance, candidate
        correct += int(best_label == label)
    return correct / n, majority, n


# ------------------------------------------------------------------ 读取与汇总
def _phase_block(case_metrics, phase):
    latent = case_metrics.get("latent") or {}
    return latent.get(phase)


def collect(entries, phase):
    """返回 (latent 矩阵, case 字典列表)；缺 latent 的 case 直接跳过。

    `entries` 是 `report.json["cases"]` 的每一项：`case` 既可能在 `metrics` 里，也可能在
    条目顶层（本测试台两处都写），所以两处都找一次。
    """
    matrix, cases = [], []
    for entry in entries:
        metrics = entry.get("metrics", entry)
        vector = _phase_block(metrics, phase)
        if not vector:
            continue
        case = metrics.get("case") or entry.get("case") or {}
        matrix.append([float(value) for value in vector])
        cases.append(case)
    return matrix, cases


def summarise_estimator(metrics_list, phase):
    rows = {}
    for item in ESTIMATOR_ITEMS:
        errors = []
        for metrics in metrics_list:
            block = ((metrics.get("estimator") or {}).get("abs_err") or {}).get(phase)
            if block and block.get(item) is not None:
                errors.append(float(block[item]))
        rows[item] = (statistics.median(errors), len(errors)) if errors else (None, 0)
    gt = {}
    #: 契约 v4：gt_mean 只有 [vx, vy, 张力] 三项（质量不再进 decoder 口径）
    for index, name in enumerate(("vx_mps", "vy_mps", "rope_tension_n")):
        values = []
        for metrics in metrics_list:
            block = ((metrics.get("estimator") or {}).get("gt_mean") or {}).get(phase)
            if block and block[index] is not None:
                values.append(float(block[index]))
        gt[name] = statistics.median(values) if values else None
    return rows, gt


def main(argv=None):
    parser = argparse.ArgumentParser(description="上层 decoder：估计误差与 latent 聚类（纯标准库）")
    parser.add_argument("--report", type=Path, required=True, help="带 --dump-latent 的 report.json")
    parser.add_argument("--phase", choices=("station", "tow", "coast"), default=DEFAULT_PHASE,
                        help="用哪个阶段的 latent 均值做聚类（默认牵引段）")
    parser.add_argument("--pcs", type=int, default=3, help="取前几个主成分做分组检验")
    args = parser.parse_args(argv)

    if not args.report.is_file():
        parser.error(f"--report 指向的文件不存在：{args.report}")
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    entries = list(payload.get("cases", []))
    metrics_list = [entry.get("metrics", entry) for entry in entries]
    if not metrics_list:
        parser.error(f"{args.report} 里没有 cases")

    matrix, cases = collect(entries, args.phase)
    print(f"# {args.report}")
    print(f"# cases={len(metrics_list)}  阶段={args.phase}  有 latent 的 case={len(matrix)}")
    if not matrix:
        print("\n[没有 latent 数据] 这一轮的 report.json 里没有 metrics['latent']。")
        print("  ⇒ 需要用 `play_towing_test.py --upper-checkpoint <ckpt> --dump-latent` 重跑一轮；")
        print("    latent 无法从 tow.csv 事后重建（帧里的 last_action/last_loco_action/joint_vel 没记录）。")
        return 1

    rows, gt = summarise_estimator(metrics_list, args.phase)
    print("\n## decoder 显式估计（中位绝对误差；真值中位作尺度参照）")
    print(f"  {'量':>12s} {'中位 |误差|':>12s} {'真值中位':>10s} {'样本':>6s}")
    for item in ESTIMATOR_ITEMS:
        value, count = rows[item]
        reference = gt.get("rope_tension_n" if item == "force_norm_n" else item)
        cell = "—" if value is None else f"{value:.4f}"
        ref = "—" if reference is None else f"{reference:.4f}"
        print(f"  {item:>12s} {cell:>12s} {ref:>10s} {count:>6d}")

    standard, _, _ = _standardize(matrix)
    components, total = principal_components(standard, max(1, args.pcs))
    print(f"\n## latent 主成分（{len(matrix)} 个 case × {len(matrix[0])} 维，标准化后）")
    for index, (eigenvalue, vector) in enumerate(components):
        share = eigenvalue / total * 100 if total else 0.0
        top = sorted(range(len(vector)), key=lambda j: -abs(vector[j]))[:4]
        loadings = " ".join(f"z{j}:{vector[j]:+.2f}" for j in top)
        print(f"  PC{index + 1}: 方差占比 {share:5.1f}%   主要载荷 {loadings}")
    scores = [[sum(row[j] * vector[j] for j in range(len(vector))) for _, vector in components]
              for row in standard]

    print("\n## latent 是否按工况聚类")
    print("   η² = 组间/总方差（0 = 分不开，1 = 完全分开）；LOO = 留一最近质心准确率（对照多数类基线）")
    print(f"  {'分组':>10s} {'组数':>5s} {'η²max(维)':>10s} {'η²(PC1)':>8s} {'LOO(全维)':>9s} "
          f"{'LOO(最优单维)':>13s} {'多数类':>7s}")
    for key, label in GROUPINGS:
        labels = [case.get(key) for case in cases]
        if any(value is None for value in labels):
            continue
        unique = sorted(set(labels), key=lambda value: str(value))
        if len(unique) < 2:
            continue
        per_dim = [eta_squared([row[j] for row in standard], labels)
                   for j in range(len(standard[0]))]
        best_dim = max(range(len(per_dim)), key=lambda j: per_dim[j])
        eta_pc1 = eta_squared([score[0] for score in scores], labels)
        accuracy, majority, n = nearest_centroid_loo(standard, labels)
        single = [[row[best_dim]] for row in standard]
        accuracy1, _, _ = nearest_centroid_loo(single, labels)
        print(f"  {label:>10s} {len(unique):>5d} {per_dim[best_dim]:>7.3f}(z{best_dim:<2d}) "
              f"{eta_pc1:>8.3f} {accuracy:>9.3f} {accuracy1:>13.3f} {majority:>7.3f}  (n={n})")
    print("\n读法：η²max / LOO 接近多数类基线 ⇒ 该分组没有进入这个瓶颈（latent 里分不开）；"
          "明显更高 ⇒ 分组（近似线性）可分。latent 坐标本身没有物理含义，只看可分性，不解释单个坐标。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
