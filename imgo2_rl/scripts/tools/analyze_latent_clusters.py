#!/usr/bin/env python3
"""离线检验 towing latent 的聚类结构：连接类型能否分开？连接内部质量档能否分开？

为什么还需要它（与 `analyze_latent.py`、`analyze_latent_tsne.py` 的分工）：

* `analyze_latent.py` 只给「逐维 η² + 留一最近质心」一组数字；
* `analyze_latent_tsne.py` 只给二维可视化（t-SNE 不产生簇标签）；
* 判断「不同连接能不能被**无监督**分出来」还需要：
  ① 真的跑聚类（k-means）并与标签比 **ARI / NMI / 纯度**（看簇是否落在标签上）；
  ② 给这些指标配**置换检验**，避免把 800 点的偶然结构当成可分；
  ③ 把**长度/坡度/速度**这些混淆项按住（`--rows` 取绳与杆同长度的行带）再比一次；
  ④ 在**每个连接内部**单独看质量档（连接混在一起时，质量的可分性会被连接带偏）。

核心指标（标准化、PCA、k-means、ARI/NMI/纯度、轮廓系数、kNN 留一、置换检验）全部只用
标准库，和仓库其它离线检查一样能在没有 Isaac Lab/numpy 的机器上复现；`--plots` 才需要
matplotlib，且只画图、不参与结论。

用法：

    python3 imgo2_rl/scripts/tools/analyze_latent_clusters.py \\
        --report imgo2_rl/logs/towing/play_test/<run>/report.json \\
        --phase tow --output <run>/latent_clusters

受控子集（只保留绳/杆同长度的 row 0–9、只看质量档）：

    ... --rows 0 9 --filter cart_mass_kg=10

读法：ARI≈0 / NMI≈0 / 纯度≈多数类 ⇒ 无监督聚类分不出该标签；ARI 明显 >0 且置换 p 很小
才说明标签在 latent 里有结构。latent 坐标没有物理含义，只看可分性，不解释单个坐标。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

# 复用 `analyze_latent.py` 里已经验证过的线性代数与分类器，避免两处各写一份。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_latent import (  # noqa: E402
    eta_squared,
    nearest_centroid_loo,
    principal_components,
)

#: 分组维度：`case` 键 → 显示名（与 analyze_latent.py 保持一致）
GROUPINGS = (
    ("connection", "连接类型"),
    ("cart_mass_kg", "质量档"),
    ("grade_deg", "坡度量级"),
    ("velocity_mps", "速度档"),
)
_MASK64 = (1 << 64) - 1


# ------------------------------------------------------------------ 基础工具
class Rng:
    """固定种子的 64 位 LCG；不碰全局 random，保证同一 seed 逐位可复现。"""

    def __init__(self, seed: int):
        self.state = (int(seed) * 6364136223846793005 + 1442695040888963407) & _MASK64

    def random(self) -> float:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & _MASK64
        return (self.state >> 11) / float(1 << 53)

    def randint(self, bound: int) -> int:
        return int(self.random() * bound) % bound if bound > 0 else 0


def standardize(rows):
    """按列标准化；常数列置 0。返回 (矩阵, 均值, 标准差)。"""
    n, width = len(rows), len(rows[0])
    means = [sum(row[j] for row in rows) / n for j in range(width)]
    scales = []
    for j in range(width):
        var = sum((row[j] - means[j]) ** 2 for row in rows) / n
        scales.append(math.sqrt(var) if var > 1e-12 else 0.0)
    matrix = [[((row[j] - means[j]) / scales[j]) if scales[j] else 0.0 for j in range(width)]
              for row in rows]
    return matrix, means, scales


def dist2(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


# ------------------------------------------------------------------ 聚类
def _kmeans_plusplus(points, k, rng):
    centroids = [list(points[rng.randint(len(points))])]
    while len(centroids) < k:
        weights = [min(dist2(point, c) for c in centroids) for point in points]
        total = sum(weights)
        if total <= 1e-18:
            centroids.append(list(points[rng.randint(len(points))]))
            continue
        target = rng.random() * total
        acc = 0.0
        for point, weight in zip(points, weights):
            acc += weight
            if acc >= target:
                centroids.append(list(point))
                break
        else:
            centroids.append(list(points[-1]))
    return centroids


def kmeans(points, k, seed, *, n_init=5, max_iter=100, tol=1e-7):
    """k-means++ 初始化 + Lloyd 迭代；返回 (assign, centroids, inertia)。多初始化取最优。"""
    if len(points) < k:
        raise ValueError(f"need >= {k} points, got {len(points)}")
    best = None
    for attempt in range(n_init):
        rng = Rng(seed * 7919 + attempt)
        centroids = _kmeans_plusplus(points, k, rng)
        assign = [0] * len(points)
        for _ in range(max_iter):
            assign = [min(range(k), key=lambda c: dist2(point, centroids[c])) for point in points]
            totals = [[0.0] * len(points[0]) for _ in range(k)]
            counts = [0] * k
            for point, cluster in zip(points, assign):
                counts[cluster] += 1
                for j, value in enumerate(point):
                    totals[cluster][j] += value
            shifted = []
            for cluster in range(k):
                if counts[cluster]:
                    shifted.append([value / counts[cluster] for value in totals[cluster]])
                else:  # 空簇：留在原地，避免除零
                    shifted.append(list(centroids[cluster]))
            movement = sum(dist2(a, b) for a, b in zip(centroids, shifted))
            centroids = shifted
            if movement <= tol:
                break
        inertia = sum(dist2(point, centroids[cluster]) for point, cluster in zip(points, assign))
        if best is None or inertia < best[0]:
            best = (inertia, assign, centroids)
    return best[1], best[2], best[0]


def silhouette(points, labels):
    """平均轮廓系数（欧氏、O(n²)）。单元素簇记 0。"""
    n = len(points)
    if n < 2:
        return 0.0
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    scores = []
    for i, point in enumerate(points):
        own = groups[labels[i]]
        if len(own) <= 1:
            scores.append(0.0)
            continue
        intra = sum(math.sqrt(dist2(point, points[j])) for j in own if j != i) / (len(own) - 1)
        nearest = min(
            sum(math.sqrt(dist2(point, points[j])) for j in members) / len(members)
            for label, members in groups.items() if label != labels[i])
        scores.append((nearest - intra) / max(intra, nearest) if max(intra, nearest) > 0 else 0.0)
    return sum(scores) / n


def contingency(labels, clusters):
    """返回 (cluster → label → 计数) 的表，键排序后输出。"""
    table = {}
    for label, cluster in zip(labels, clusters):
        table.setdefault(cluster, {})
        table[cluster][label] = table[cluster].get(label, 0) + 1
    return table


def purity(labels, clusters):
    table = contingency(labels, clusters)
    return sum(max(row.values()) for row in table.values()) / len(labels)


def _entropy(counts, total):
    return -sum(c / total * math.log(c / total) for c in counts if c > 0)


def normalized_mutual_information(a, b):
    """NMI（算术平均归一化；含退化分支，与 scikit-learn 默认口径一致）。"""
    n = len(a)
    ca, cb, cab = {}, {}, {}
    for x, y in zip(a, b):
        ca[x] = ca.get(x, 0) + 1
        cb[y] = cb.get(y, 0) + 1
        cab[(x, y)] = cab.get((x, y), 0) + 1
    ha, hb = _entropy(ca.values(), n), _entropy(cb.values(), n)
    if ha <= 0 and hb <= 0:  # 两边都是单一簇：sklearn 记 1.0
        return 1.0
    if ha <= 0 or hb <= 0:   # 只有一边是单一簇：互信息必然为 0
        return 0.0
    mutual = 0.0
    for (x, y), count in cab.items():
        mutual += count / n * math.log(n * count / (ca[x] * cb[y]))
    return 2 * mutual / (ha + hb)


def adjusted_rand_index(a, b):
    """ARI；`最大索引 == 期望索引`的退化情形按 scikit-learn 记 1.0。"""
    n = len(a)
    ca, cb, cab = {}, {}, {}
    for x, y in zip(a, b):
        ca[x] = ca.get(x, 0) + 1
        cb[y] = cb.get(y, 0) + 1
        cab[(x, y)] = cab.get((x, y), 0) + 1
    comb2 = lambda value: value * (value - 1) / 2  # noqa: E731
    index = sum(comb2(c) for c in cab.values())
    expected = sum(comb2(c) for c in ca.values()) * sum(comb2(c) for c in cb.values()) / comb2(n)
    maximum = (sum(comb2(c) for c in ca.values()) + sum(comb2(c) for c in cb.values())) / 2
    if maximum == expected:
        return 1.0
    return (index - expected) / (maximum - expected)


def knn_loo(points, labels, k=5):
    """留一 k 近邻多数投票；返回 (准确率, 多数类基线)。"""
    n = len(points)
    unique = sorted(set(labels), key=str)
    counts = {label: labels.count(label) for label in unique}
    correct = 0
    for i, point in enumerate(points):
        neighbours = sorted((j for j in range(n) if j != i),
                            key=lambda j: (dist2(point, points[j]), j))[:k]
        votes = {}
        for j in neighbours:
            votes[labels[j]] = votes.get(labels[j], 0) + 1
        best = max(votes.items(), key=lambda item: (item[1], -unique.index(item[0])))[0]
        correct += int(best == labels[i])
    return correct / n, max(counts.values()) / n


# ------------------------------------------------------------------ 可分性统计
def multivariate_eta2(points, labels):
    """全维「组间散度迹 / 总散度迹」（多变量 η²）。"""
    n = len(points)
    grand = [sum(row[j] for row in points) / n for j in range(len(points[0]))]
    total = sum(sum((row[j] - grand[j]) ** 2 for j in range(len(row))) for row in points)
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    between = 0.0
    for members in groups.values():
        centroid = [sum(points[i][j] for i in members) / len(members) for j in range(len(grand))]
        between += len(members) * sum((centroid[j] - grand[j]) ** 2 for j in range(len(grand)))
    return between / total if total > 1e-12 else 0.0


def permutation_pvalue(points, labels, statistic, *, permutations, seed):
    """标签置换检验：返回 (观测值, 置换均值, p)。p = (1+#{perm >= obs}) / (1+B)。"""
    observed = statistic(points, labels)
    if permutations <= 0:
        return observed, float("nan"), float("nan")
    rng = Rng(seed)
    shuffled = list(labels)
    hits, total = 0, 0.0
    for _ in range(permutations):
        for index in range(len(shuffled) - 1, 0, -1):  # Fisher-Yates
            swap = rng.randint(index + 1)
            shuffled[index], shuffled[swap] = shuffled[swap], shuffled[index]
        value = statistic(points, shuffled)
        total += value
        hits += int(value >= observed - 1e-12)
    return observed, total / permutations, (1 + hits) / (1 + permutations)


def _centroid_statistic(points, labels):
    return nearest_centroid_loo(points, labels)[0]


def centroid_loo_confusion(points, labels):
    """留一最近质心的混淆矩阵（行 = 真实标签，列 = 预测标签）；回答「哪两类互相混」。"""
    unique = sorted(set(labels), key=str)
    position = {label: index for index, label in enumerate(unique)}
    counts = {label: labels.count(label) for label in unique}
    sums = {label: [0.0] * len(points[0]) for label in unique}
    for point, label in zip(points, labels):
        for j, value in enumerate(point):
            sums[label][j] += value
    matrix = [[0] * len(unique) for _ in unique]
    for point, label in zip(points, labels):
        best, best_label = None, None
        for candidate in unique:
            count = counts[candidate] - (1 if candidate == label else 0)
            if count <= 0:
                continue
            centroid = [(sums[candidate][j] - (point[j] if candidate == label else 0.0)) / count
                        for j in range(len(point))]
            distance = dist2(point, centroid)
            if best is None or distance < best:
                best, best_label = distance, candidate
        matrix[position[label]][position[best_label]] += 1
    return unique, matrix


def label_report(points, labels, *, permutations, seed, include_silhouette=True):
    unique = sorted(set(labels), key=str)
    centroid_accuracy, majority, _ = nearest_centroid_loo(points, labels)
    knn_accuracy, _ = knn_loo(points, labels, k=min(5, len(points) - 1))
    eta_all, eta_mean, p_eta = permutation_pvalue(
        points, labels, multivariate_eta2, permutations=permutations, seed=seed)
    # 分类器也要有零假设：随机打乱标签后，留一最近质心自己就能拿到一个不低的准确率。
    _, centroid_mean, p_centroid = permutation_pvalue(
        points, labels, _centroid_statistic, permutations=permutations, seed=seed + 1)
    # 混淆矩阵：最高 4 类才打，避免刷屏
    confusion = {}
    if len(unique) <= 4:
        confusion_labels, matrix = centroid_loo_confusion(points, labels)
        confusion = {"rows": "truth", "cols": "predicted", "labels": confusion_labels,
                     "matrix": matrix}
    report = {
        "groups": len(unique),
        "counts": {str(label): labels.count(label) for label in unique},
        "eta2_multivariate": eta_all,
        "eta2_permutation_mean": eta_mean,
        "eta2_permutation_p": p_eta,
        "centroid_loo": centroid_accuracy,
        "centroid_loo_permutation_mean": centroid_mean,
        "centroid_loo_permutation_p": p_centroid,
        "knn5_loo": knn_accuracy,
        "majority_baseline": majority,
        "permutations": permutations,
        "centroid_loo_confusion": confusion,
    }
    if include_silhouette:
        report["silhouette"] = silhouette(points, labels)
    # 逐维 η²（标准化后每维方差≈1，所以可直接比较）
    per_dim = [eta_squared([row[j] for row in points], labels) for j in range(len(points[0]))]
    best = max(range(len(per_dim)), key=lambda j: per_dim[j])
    report["eta2_best_dim"] = {"dim": best, "value": per_dim[best]}
    return report


# ------------------------------------------------------------------ 读取
def _phase_block(metrics, phase):
    return (metrics.get("latent") or {}).get(phase)


def load_cases(payload, phase, *, include_nocart, filters, rows):
    """返回 (raw 向量, case, mass 误差列表, 统计)。按 filters/rows 过滤。"""
    vectors, cases, errors = [], [], []
    skipped = {"missing_latent": 0, "invalid_latent": 0, "nocart": 0, "filtered": 0}
    for entry in payload.get("cases", []):
        metrics = entry.get("metrics", entry)
        case = metrics.get("case") or entry.get("case") or {}
        if case.get("cart_present") is False and not include_nocart:
            skipped["nocart"] += 1
            continue
        if rows is not None and not (rows[0] <= case.get("row", -1) <= rows[1]):
            skipped["filtered"] += 1
            continue
        if any(case.get(key) != value for key, value in filters.items()):
            skipped["filtered"] += 1
            continue
        vector = _phase_block(metrics, phase)
        if not vector:
            skipped["missing_latent"] += 1
            continue
        try:
            vector = [float(value) for value in vector]
            valid = len(vector) == 16 and all(math.isfinite(value) for value in vector)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            skipped["invalid_latent"] += 1
            continue
        error = ((metrics.get("estimator") or {}).get("abs_err") or {}).get(phase) or {}
        vectors.append(vector)
        cases.append(case)
        errors.append(error.get("mass_kg"))
    return vectors, cases, errors, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True, help="带 --dump-latent 的 report.json")
    parser.add_argument("--phase", choices=("station", "tow", "coast"), default="tow")
    parser.add_argument("--output", type=Path, help="写 metrics.json / points.csv（可选）")
    parser.add_argument("--plots", action="store_true", help="额外写 PCA 图（需要 matplotlib）")
    parser.add_argument("--include-nocart", action="store_true", help="保留 cart_present=false 的样本")
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=JSON_VALUE")
    parser.add_argument("--rows", type=int, nargs=2, metavar=("MIN", "MAX"),
                        help="只保留该 row 区间（0–9 是绳/杆同长度带）")
    parser.add_argument("--cluster-k", type=int, nargs="+", default=[3, 4, 12])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--pcs", type=int, default=3)
    parser.add_argument("--connection-binary", action="store_true",
                        help="额外把 compliant/inextensible 并成 rope，单独评一次 rope vs rigid")
    args = parser.parse_args(argv)

    if not args.report.is_file():
        parser.error(f"--report 不存在：{args.report}")
    if args.permutations < 0:
        parser.error("--permutations 必须 >= 0")
    filters = {}
    for item in args.filter:
        try:
            key, value = item.split("=", 1)
            filters[key] = json.loads(value)
        except ValueError as exc:
            parser.error(f"--filter 需要 KEY=JSON_VALUE：{item!r}（{exc}）")
    rows = tuple(args.rows) if args.rows else None
    if rows is not None and rows[0] > rows[1]:
        parser.error("--rows MIN 不能大于 MAX")

    payload = json.loads(args.report.read_text(encoding="utf-8-sig"))
    raw, cases, errors, skipped = load_cases(
        payload, args.phase, include_nocart=args.include_nocart, filters=filters, rows=rows)
    print(f"# {args.report}\n# 阶段={args.phase} 选中={len(raw)} 过滤/跳过={skipped} "
          f"filters={filters or '{}'} rows={rows or '全部'}")
    if len(raw) < 8:
        print("样本不足 8 个，不输出聚类结论（需要报告带 --dump-latent、或放宽 --filter/--rows）。")
        return 1

    matrix, means, scales = standardize(raw)
    components, total_variance = principal_components(matrix, max(1, args.pcs))
    scores = [[sum(row[j] * vector[j] for j in range(len(vector))) for _, vector in components]
              for row in matrix]
    variance_share = [eigenvalue / total_variance * 100 if total_variance else 0.0
                      for eigenvalue, _ in components]
    print("\n## latent 主成分（标准化后）")
    for index, (eigenvalue, vector) in enumerate(components):
        top = sorted(range(len(vector)), key=lambda j: -abs(vector[j]))[:4]
        loadings = " ".join(f"z{j}:{vector[j]:+.2f}" for j in top)
        print(f"  PC{index + 1}: {variance_share[index]:5.1f}%   主要载荷 {loadings}")

    labels_by_key = {key: [case.get(key) for case in cases] for key, _ in GROUPINGS}
    if args.connection_binary and all(labels_by_key["connection"]):
        labels_by_key["connection_binary"] = [
            "rigid" if value == "rigid" else "rope" for value in labels_by_key["connection"]]
    print("\n## 标签可分性（16 维标准化 latent；置换检验打乱标签）")
    print(f"  {'分组':>8s} {'组数':>4s} {'多变量η²':>9s} {'η²置换p':>8s} {'轮廓':>7s} "
          f"{'质心LOO':>8s} {'LOO置换p':>8s} {'kNN5 LOO':>9s} {'多数类':>7s} {'最优单维η²':>12s}")
    label_rows = {}
    for key, display in GROUPINGS + (("connection_binary", "绳vs杆"),):
        labels = labels_by_key.get(key)
        if not labels or any(value is None for value in labels):
            continue
        if len(set(labels)) < 2:
            continue
        report = label_report(matrix, labels, permutations=args.permutations, seed=args.seed)
        label_rows[key] = report
        print(f"  {display:>8s} {report['groups']:>4d} {report['eta2_multivariate']:>9.3f} "
              f"{report['eta2_permutation_p']:>8.3f} {report.get('silhouette', float('nan')):>7.3f} "
              f"{report['centroid_loo']:>8.3f} {report['centroid_loo_permutation_p']:>8.3f} "
              f"{report['knn5_loo']:>9.3f} {report['majority_baseline']:>7.3f} "
              f"{report['eta2_best_dim']['value']:>7.3f}(z{report['eta2_best_dim']['dim']})")

    print("\n  留一最近质心混淆矩阵（行=真实标签，列=预测标签）")
    for key, display in GROUPINGS + (("connection_binary", "绳vs杆"),):
        report = label_rows.get(key)
        if not report:
            continue
        confusion = report["centroid_loo_confusion"]
        print(f"    [{display}] {'':>10s}" + "".join(f"{str(label):>14s}" for label in confusion["labels"]))
        for label, row in zip(confusion["labels"], confusion["matrix"]):
            print(f"    {'':>8s}{str(label):>14s}" + "".join(f"{value:>14d}" for value in row))

    print("\n## 无监督聚类 vs 标签（k-means，多初始化取最优）")
    print(f"  {'对照':>16s} {'k':>3s} {'ARI':>7s} {'NMI':>7s} {'纯度':>7s} {'多数类纯度':>10s}")
    cluster_runs = []
    targets = [("connection", "连接类型", 3), ("cart_mass_kg", "质量档×连接", None)]
    if "connection_binary" in labels_by_key:
        targets.append(("connection_binary", "绳vs杆", 2))
    for key, display, target_k in targets:
        labels = labels_by_key.get(key)
        if not labels or any(value is None for value in labels):
            continue
        k = target_k or len(set(labels))
        if k < 2:  # 受控子集里该标签只剩一类（例如固定质量后），没有可分性问题
            continue
        assign, _, inertia = kmeans(matrix, k, args.seed)
        ari = adjusted_rand_index(labels, assign)
        nmi = normalized_mutual_information(labels, assign)
        pur = purity(labels, assign)
        majority_purity = max(labels.count(v) for v in set(labels)) / len(labels)
        print(f"  {display:>16s} {k:>3d} {ari:>7.3f} {nmi:>7.3f} {pur:>7.3f} {majority_purity:>10.3f}")
        cluster_runs.append({"target": key, "k": k, "ari": ari, "nmi": nmi, "purity": pur,
                             "majority_purity": majority_purity, "inertia": inertia,
                             "contingency": {str(c): {str(l): int(v) for l, v in row.items()}
                                             for c, row in contingency(labels, assign).items()}})
    for k in sorted(set(args.cluster_k)):
        if k < 2 or k >= len(matrix):
            continue
        assign, _, inertia = kmeans(matrix, k, args.seed)
        print(f"  {'(无标签)':>16s} {k:>3d}  inertia={inertia:9.1f}  轮廓={silhouette(matrix, assign):.3f}")
        cluster_runs.append({"target": None, "k": k, "inertia": inertia,
                             "silhouette": silhouette(matrix, assign)})

    # 连接 × 质量 的联合标签（12 类）单独评一次：k-means k=12 是否落在设计格上
    if not args.include_nocart and all(labels_by_key["connection"]) and all(labels_by_key["cart_mass_kg"]):
        joint = [f"{c}|{m:g}" for c, m in zip(labels_by_key["connection"], labels_by_key["cart_mass_kg"])]
        if len(set(joint)) >= 2:
            k = len(set(joint))
            assign, _, inertia = kmeans(matrix, k, args.seed)
            print(f"  {'连接×质量(12类)':>16s} {k:>3d} {adjusted_rand_index(joint, assign):>7.3f} "
                  f"{normalized_mutual_information(joint, assign):>7.3f} "
                  f"{purity(joint, assign):>7.3f} "
                  f"{max(joint.count(v) for v in set(joint)) / len(joint):>10.3f}")
            cluster_runs.append({"target": "connection|mass", "k": k,
                                 "ari": adjusted_rand_index(joint, assign),
                                 "nmi": normalized_mutual_information(joint, assign),
                                 "purity": purity(joint, assign), "inertia": inertia})

    print("\n## 每个连接内部：质量档能不能分开（只看有车样本）")
    print("  「latent 半径」= 组内逐维方差的平方根之和（未降维的距离尺度；≈0 说明该组 latent 近似常数）")
    print(f"  {'连接':>12s} {'n':>4s} {'latent半径':>9s} {'多变量η²':>9s} {'置换p':>7s} {'质心LOO':>8s} "
          f"{'kNN5 LOO':>9s} {'多数类':>7s} {'k=4 ARI':>8s} {'质量误差中位':>11s}")
    per_connection = {}
    for connection in sorted(set(labels_by_key["connection"]), key=str):
        index = [i for i, value in enumerate(labels_by_key["connection"]) if value == connection]
        if len(index) < 8:
            continue
        subset = [matrix[i] for i in index]
        masses = [labels_by_key["cart_mass_kg"][i] for i in index]
        subset_errors = [errors[i] for i in index if errors[i] is not None]
        report = label_report(subset, masses, permutations=args.permutations,
                              seed=args.seed, include_silhouette=False)
        k = len(set(masses))
        assign, _, _ = kmeans(subset, k, args.seed) if k > 1 else ([0] * len(subset), None, 0.0)
        ari = adjusted_rand_index(masses, assign) if k > 1 else float("nan")
        spread = math.sqrt(sum(statistics.pvariance([row[j] for row in subset])
                               for j in range(len(subset[0]))))
        report.update({"n": len(index), "mass_ari_k4": ari, "latent_rms_spread": spread,
                       "mass_error_median": statistics.median(subset_errors) if subset_errors else None,
                       "error_samples": len(subset_errors)})
        per_connection[connection] = report
        error_text = "—" if report["mass_error_median"] is None else f"{report['mass_error_median']:.3f}"
        print(f"  {connection:>12s} {len(index):>4d} {spread:>9.3f} {report['eta2_multivariate']:>9.3f} "
              f"{report['eta2_permutation_p']:>7.3f} {report['centroid_loo']:>8.3f} "
              f"{report['knn5_loo']:>9.3f} {report['majority_baseline']:>7.3f} {ari:>8.3f} "
              f"{error_text:>11s}")

    print("\n读法：ARI/NMI/纯度接近 0/多数类 ⇒ 聚类没落在该标签上；置换 p 是「同规模标签随机"
          "打乱仍能达到该 η²」的概率。轮廓系数越大越成团，但大簇+小簇工况下会偏低。")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "points.csv").open("w", newline="", encoding="utf-8") as handle:
            # 显式 LF：仓库 `.gitattributes` 是 `* text=auto eol=lf`，csv 模块默认写 CRLF，
            # 否则每次 add 都会被警告「CRLF 将被 LF 替换」，工作区与入库内容也不一致。
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["source_index", *[f"pc{i + 1}" for i in range(len(components))],
                             "connection", "cart_mass_kg", "grade_deg", "velocity_mps",
                             "connection_length_m", "row", "column", "cart_present"])
            for i, (case, score) in enumerate(zip(cases, scores)):
                writer.writerow([i, *score, case.get("connection"), case.get("cart_mass_kg"),
                                 case.get("grade_deg"), case.get("velocity_mps"),
                                 case.get("connection_length_m"), case.get("row"),
                                 case.get("column"), case.get("cart_present")])
        metadata = {
            "source": str(args.report.resolve()), "phase": args.phase,
            "selected": len(raw), "skipped": skipped, "filters": filters, "rows": rows,
            "include_nocart": args.include_nocart, "seed": args.seed,
            "permutations": args.permutations,
            "standardization_mean": means, "standardization_scale": scales,
            "pca_variance_share_percent": variance_share,
            "labels": label_rows, "clustering": cluster_runs,
            "per_connection_mass": per_connection,
            "caution": "k-means/ARI 只看标签在 latent 里是否有结构；latent 坐标没有物理含义，"
                       "不等于物理参数辨识。置换检验用同一批标准化坐标，p 小只说明结构不像是"
                       "随机标签造成的。",
        }
        (args.output / "metrics.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写 {args.output}/metrics.json 与 points.csv")

        if args.plots:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError as exc:  # 只影响画图，不影响上面的结论
                print(f"[跳过画图] 缺少 matplotlib：{exc}")
            else:
                _configure_cjk_font(plt)
                _write_plots(args.output, scores, labels_by_key, cluster_runs, matrix,
                             args.seed, variance_share, plt)
                print(f"已写 {args.output}/pca_*.png")
    return 0


def _configure_cjk_font(plt):
    """图里要写中文，优先挑一个系统里真实存在的 CJK 字体；没有就退回默认（只影响标签显示）。"""
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC",
                      "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei"):
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    print("[提示] 系统里没有可用的中文字体，图中的中文可能显示为方框（不影响数值结论）。")
    return None


def _scatter_by_label(ax, scores, values, title, *, marker=None, size=14):
    """按标签上色；每类换一个 marker，避免重叠时后画的类把前面的完全盖住。"""
    unique = sorted(set(values), key=lambda v: (0, v) if isinstance(v, (int, float)) else (1, str(v)))
    cmap = plt_cmap(len(unique))
    markers = (marker,) if marker else ("o", "^", "s", "D", "v", "P", "X", "*")
    for i, label in enumerate(unique):
        mask = [v == label for v in values]
        ax.scatter([s[0] for s, keep in zip(scores, mask) if keep],
                   [s[1] for s, keep in zip(scores, mask) if keep],
                   s=size, alpha=.75, marker=markers[i % len(markers)], color=cmap(i),
                   label=f"{label if label is not None else 'missing'} (n={sum(mask)})")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7)


def plt_cmap(count):
    import matplotlib.pyplot as plt
    return plt.get_cmap("tab10" if count <= 10 else "tab20")


def _write_plots(output, scores, labels_by_key, cluster_runs, matrix, seed, variance_share, plt):
    connections = labels_by_key["connection"]
    masses = labels_by_key["cart_mass_kg"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _scatter_by_label(axes[0], scores, connections, "按连接类型")
    _scatter_by_label(axes[1], scores, masses, "按质量档（kg）")
    fig.suptitle(f"latent PCA（PC1 {variance_share[0]:.0f}% / PC2 "
                 f"{variance_share[1] if len(variance_share) > 1 else 0:.0f}%）")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output / "pca_labels.png", dpi=200)
    plt.close(fig)

    cluster_run = next((run for run in cluster_runs if run.get("target") == "connection"), None)
    if cluster_run:
        assign, _, _ = kmeans(matrix, cluster_run["k"], seed)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        _scatter_by_label(axes[0], scores, [f"C{c}" for c in assign],
                          f"k-means 簇（k={cluster_run['k']}）")
        _scatter_by_label(axes[1], scores, connections, "同一坐标按连接类型")
        fig.suptitle(f"k-means vs 连接类型：ARI={cluster_run['ari']:.3f} "
                     f"NMI={cluster_run['nmi']:.3f} 纯度={cluster_run['purity']:.3f}")
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(output / "pca_kmeans_connection.png", dpi=200)
        plt.close(fig)

    present = sorted({c for c, m in zip(connections, masses) if c}, key=str)
    fig, axes = plt.subplots(1, len(present), figsize=(5.5 * len(present), 5.2), squeeze=False)
    for ax, connection in zip(axes.flat, present):
        index = [i for i, value in enumerate(connections) if value == connection]
        _scatter_by_label(ax, [scores[i] for i in index], [masses[i] for i in index],
                          f"{connection}（n={len(index)}）", size=16)
        for mass in sorted({masses[i] for i in index}, key=float):
            members = [i for i in index if masses[i] == mass]
            ax.plot(sum(scores[i][0] for i in members) / len(members),
                    sum(scores[i][1] for i in members) / len(members),
                    marker="x", markersize=13, markeredgewidth=2.5, color="black")
    fig.suptitle("每个连接内部按质量档着色（黑 × = 该档质心）")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output / "pca_mass_by_connection.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
