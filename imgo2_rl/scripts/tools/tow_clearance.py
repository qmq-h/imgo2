"""由 URDF 几何 + 记录的关节角，离线算「小车车头 ↔ 机器人后腿表面」的真实间隙。

## 为什么需要它

`tow.csv` 里的 `rope_distance_m` 是**两个挂点**（机器人 base 后 0.16 m、小车 base 前
0.25 m）之间的三维距离，**不是**车头与机器人的距离。两者的差是一个随机器人姿态变化的量：

    间隙 = d_挂点 − (|x_min_腿| − 0.16 m) + ε        ε = 0（撞车斗前表面）
                                                       或 0.01 m（撞前轮外缘）

（车斗前表面与小车挂点同在 base 前 0.25 m，两项相消，所以主项只剩腿的后伸量。）
`|x_min_腿|` 只由机器人腿的位形决定，实测范围 **0.27~0.59 m**，于是「碰撞发生时的挂点
间距」在 **0.10~0.43 m** 之间浮动（见 `docs/towing_p4_tow_drag_2026-09-20.md` §5.17）。
因此

  * 不能用挂点间距判断「小车是否追到机器人」——四个真实撞上机器人的 run 里，
    该值分别是 0.3631 / 0.3744 / 0.3649 / 0.3457 m，全都看不出碰撞；同一个 0.3639 m 在
    默认站姿下是「还差 0.126 m」、在后腿伸到最远时是「已经压进去 0.067 m」；
  * 也不能用它当「最终停车距离」，它是挂点距离而不是车头距离。

判「是否真的接触」以**接触力**为准（`cart_deck_fx_n` 非零、`cart_wheel_fx_n` 突跳），
那是直接测力、不依赖任何几何假设；本工具算的是「间隙」这个连续量，用来给出停车距离，
并可在接触力报警的同一时刻验证间隙是否刚好过零（自洽性交叉验证）。

## 算法与近似（都要写清楚，别让数字看起来比它实际更硬）

1. FK 用 `urdf/imgo2.urdf` 的链式变换，关节角取自记录（策略/契约顺序，由 config.json 的
   `policy_joint_names` 给出）；关节名与 URDF 名逐一核对，对不上直接报错。
2. 碰撞几何用**表面采样点**表示：长方体取 8 个角 + 棱上按 `SAMPLE_M` 加密；圆柱取两端
   圆周 16 等分 + 轴端；球取中心±半径。因此单个刚体的 x 极值最多偏 `r(1−cos(π/16))`
   = 0.019r（车轮 r=0.08 ⇒ 1.5 mm）。
3. 「间隙」定义为**面向面之间的纵向差**：先把两团点各自取朝向对方的那一侧
   （离极值 `FACE_BAND_M` 以内的点），再在**横向/竖向都在 `FACING_TOL_M` 以内**的点对里
   取 `min(x_robot − x_cart)`。所以它是「车头到机器人后表面」的纵向距离，不是三维最短距离；
   真正的三维最短距离会给得更大（两侧错开时）。极端错位下它偏保守（偏小），
   这方向对「有没有撞上」是安全的，对「停车距离」是保守估计。
4. 忽略车轮自转：车轮是绕自身轴的圆柱，自转不改变它在世界系的 x 范围。
"""

from __future__ import annotations

import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
SAMPLE_M = 0.02          # 长方体棱上的加密间距
CIRCLE_SAMPLES = 16      # 圆柱端面圆周等分数
FACE_BAND_M = 0.05       # 取「面向对方的一侧」的带宽
FACING_TOL_M = 0.05      # 判为「面对面」所需的横向/竖向接近度


# ------------------------------------------------------------------ 基础线性代数
def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rotate(r, v):
    return [sum(r[i][j] * v[j] for j in range(3)) for i in range(3)]


def axis_rotation(axis, angle):
    norm = math.sqrt(sum(x * x for x in axis))
    if norm == 0.0:
        return IDENTITY
    x, y, z = [a / norm for a in axis]
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]


def rpy_rotation(rpy):
    return matmul(axis_rotation([0, 0, 1], rpy[2]),
                  matmul(axis_rotation([0, 1, 0], rpy[1]), axis_rotation([1, 0, 0], rpy[0])))


def quaternion_rotation(q_xyzw):
    """由 (x, y, z, w) 得旋转矩阵（记录里四元数按 x/y/z/w 分列存）。"""
    x, y, z, w = q_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("零四元数")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def _vec(element, key, default):
    if element is None:
        return list(default)
    raw = element.get(key)
    return [float(v) for v in raw.split()] if raw else list(default)


# ------------------------------------------------------------------ 模型解析
def read_chain(urdf, tip, root):
    """返回从 root 到 tip 的关节链 [(kind, xyz, R_origin, axis), ...]。"""
    by_child = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    chain, node = [], tip
    while node != root:
        joint = by_child[node]
        origin = joint.find("origin")
        axis_element = joint.find("axis")
        axis = _vec(axis_element, "xyz", (1.0, 0.0, 0.0)) if axis_element is not None else None
        kind = joint.get("type")
        if kind not in ("revolute", "continuous", "fixed"):
            raise ValueError(f"不支持的关节类型 {kind!r}（{joint.get('name')}）")
        chain.append((kind, _vec(origin, "xyz", (0, 0, 0)), rpy_rotation(_vec(origin, "rpy", (0, 0, 0))), axis,
                      joint.get("name")))
        node = joint.find("parent").get("link")
    return chain[::-1]


def read_primitives(urdf, link_name):
    """某 link 的碰撞几何 [(原点偏移, 原点旋转, 形状)]；形状是 ('box', half) / ('cyl', r, half) / ('sph', r)。"""
    for link in urdf.findall("link"):
        if link.get("name") != link_name:
            continue
        out = []
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            geometry = collision.find("geometry")
            box, cylinder, sphere = geometry.find("box"), geometry.find("cylinder"), geometry.find("sphere")
            if box is not None:
                shape = ("box", [float(v) / 2 for v in box.get("size").split()])
            elif cylinder is not None:
                shape = ("cyl", float(cylinder.get("radius")), float(cylinder.get("length")) / 2)
            elif sphere is not None:
                shape = ("sph", float(sphere.get("radius")))
            else:
                continue
            out.append((_vec(origin, "xyz", (0, 0, 0)), rpy_rotation(_vec(origin, "rpy", (0, 0, 0))), shape))
        return out
    raise KeyError(f"URDF 里没有 link {link_name!r}")


def shape_points(shape):
    """形状的局部表面采样点（含极值点）。"""
    kind = shape[0]
    if kind == "box":
        half = shape[1]
        points = []
        for sign_x in (-1, 1):
            for sign_y in (-1, 1):
                steps = max(1, int(math.ceil(2 * half[2] / SAMPLE_M)))
                for i in range(steps + 1):
                    z = -half[2] + 2 * half[2] * i / steps
                    points.append([sign_x * half[0], sign_y * half[1], z])
        for sign_x in (-1, 1):
            for sign_z in (-1, 1):
                steps = max(1, int(math.ceil(2 * half[1] / SAMPLE_M)))
                for i in range(steps + 1):
                    y = -half[1] + 2 * half[1] * i / steps
                    points.append([sign_x * half[0], y, sign_z * half[2]])
        for sign_y in (-1, 1):
            for sign_z in (-1, 1):
                steps = max(1, int(math.ceil(2 * half[0] / SAMPLE_M)))
                for i in range(steps + 1):
                    x = -half[0] + 2 * half[0] * i / steps
                    points.append([x, sign_y * half[1], sign_z * half[2]])
        return points
    if kind == "cyl":
        radius, half = shape[1], shape[2]
        points = []
        for sign in (-1, 1):
            for i in range(CIRCLE_SAMPLES):
                angle = 2 * math.pi * i / CIRCLE_SAMPLES
                points.append([radius * math.cos(angle), sign * half, radius * math.sin(angle)])
        points += [[0.0, -half, 0.0], [0.0, half, 0.0]]
        return points
    radius = shape[1]
    return [[radius, 0.0, 0.0], [-radius, 0.0, 0.0], [0.0, radius, 0.0],
            [0.0, -radius, 0.0], [0.0, 0.0, radius], [0.0, 0.0, -radius]]


def urdf_total_mass(urdf) -> float:
    """URDF 全部 link 的质量之和。

    用途不在间隙，而在弹性诊断（§5.18）：绳的折合质量 `μ = ((1/m_eff)+(1/m_robot))⁻¹`
    需要机器人质量，而旧 run 的 `config.json` 里没有记它——从 URDF 求和就能让这些 run
    **不用重跑**也把伸长/过冲比算出来。
    """
    total = 0.0
    for link in urdf.findall("link"):
        inertial = link.find("inertial")
        mass = inertial.find("mass") if inertial is not None else None
        if mass is not None:
            total += float(mass.get("value"))
    return total


class Body:
    """一个刚体：若干 (链, 碰撞几何) 部件，能把采样点送到世界系。"""

    def __init__(self, name, parts, root_link, joints):
        self.name = name
        self.parts = parts              # [(chain, [ (offset, R, shape), ... ]), ...]
        self.root_link = root_link
        self.joints = joints            # 本条链用到的关节名集合（用于核对记录是否齐全）

    def points(self, joint_angles, position, rotation):
        """joint_angles: {关节名: 角度}；position/rotation: 根刚体在世界系的位姿。"""
        out = []
        for chain, primitives in self.parts:
            p, r = list(position), rotation
            index = 0
            for kind, xyz, origin_r, axis, name in chain:
                p = [a + b for a, b in zip(p, rotate(r, xyz))]
                r = matmul(r, origin_r)
                if kind != "fixed":
                    angle = joint_angles.get(name)
                    if angle is None:
                        raise KeyError(f"记录里缺关节角 {name}")
                    r = matmul(r, axis_rotation(axis, angle))
                    index += 1
            for offset, origin_r, shape in primitives:
                centre = [a + b for a, b in zip(p, rotate(r, offset))]
                m = matmul(r, origin_r)
                for local in shape_points(shape):
                    out.append([a + b for a, b in zip(centre, rotate(m, local))])
        return out


# 车轮自转角没有被记录（只有 ω），但车轮是**绕自身轴**的圆柱，自转不改变它在世界系的
# x 范围；影响只是采样多边形相对真圆的 x 极值偏差，上界 r(1−cos(π/16)) = 1.5 mm。
# 所以这里显式取 0，而不是让缺失关节静默当 0（后者正是造成过假通过的坏模式）。
WHEEL_JOINT_ANGLES = {f"wheel_{leg}_joint": 0.0 for leg in ("fl", "fr", "rl", "rr")}


def build_bodies(repo: Path):
    robot_urdf = ET.parse(repo / "imgo2_description" / "urdf" / "imgo2.urdf").getroot()
    cart_urdf = ET.parse(repo / "imgo2_description" / "cart" / "cart.urdf").getroot()
    # 机器人：base（躯干）+ 四条腿。前腿正常情况够不到小车，但姿态极端时可能，
    # 所以四条都算，不用「后腿」这种会随姿态失效的假设。
    robot_parts = []
    for link in ("base",):
        robot_parts.append(([], [(offset, r, shape)
                                 for offset, r, shape in read_primitives(robot_urdf, link)]))
    for leg in ("FL", "FR", "RL", "RR"):
        for part in ("HIP", "THIGH", "SHANK", "FOOT"):
            link = f"{leg}_{part}"
            chain = read_chain(robot_urdf, link, "base")
            robot_parts.append((chain, read_primitives(robot_urdf, link)))
    robot = Body("robot", robot_parts, "base", None)

    cart_parts = []
    for link in ("base_link", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
        chain = read_chain(cart_urdf, link, "base_link")
        cart_parts.append((chain, read_primitives(cart_urdf, link)))
    cart = Body("cart", cart_parts, "base_link", None)
    return robot, cart


def robot_mass_kg(repo: Path) -> float:
    """机器人总质量（URDF 求和），供弹性诊断算折合质量。"""
    return urdf_total_mass(ET.parse(repo / "imgo2_description" / "urdf" / "imgo2.urdf").getroot())


# ------------------------------------------------------------------ 间隙计算
def clearance(robot_points, cart_points):
    """返回 (纵向间隙 m, 三维最近距离 m)。

    纵向间隙 = 只取面向对方的一侧面点、且横向/竖向足够接近的点对里 min(x_robot − x_cart)。
    三维最近距离 = 同一批点对里 min(‖Δ‖)。两者都只在「面面对」点对上取，
    因此是车头/后表面之间的量，而不是整团点云的全局最小值。
    """
    if not robot_points or not cart_points:
        raise ValueError("点集为空")
    cart_front = max(p[0] for p in cart_points)
    front_cart = [p for p in cart_points if p[0] >= cart_front - FACE_BAND_M]
    robot_rear = min(p[0] for p in robot_points)
    rear_robot = [p for p in robot_points if p[0] <= robot_rear + FACE_BAND_M]
    gap, distance = None, None
    for rp in rear_robot:
        for cp in front_cart:
            if abs(rp[1] - cp[1]) > FACING_TOL_M or abs(rp[2] - cp[2]) > FACING_TOL_M:
                continue
            scalar = rp[0] - cp[0]
            if gap is None or scalar < gap:
                gap = scalar
            norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(rp, cp)))
            if distance is None or norm < distance:
                distance = norm
    return gap, distance


def row_state(row, joint_names):
    """从一行记录取出 FK 需要的全部量。"""
    angles = {}
    for index, name in enumerate(joint_names):
        key = f"robot_jp_{index:02d}"
        if key not in row:
            raise KeyError(f"记录里缺 {key}（本 run 早于间隙记录，需重跑）")
        angles[name] = float(row[key])
    robot_pose = ([float(row["robot_x_m"]), 0.0, float(row["robot_z_m"])],
                  quaternion_rotation([float(row[f"robot_quat_{axis}"]) for axis in "xyzw"]))
    cart_pose = ([float(row["load_x_m"]), 0.0, float(row["load_z_m"])],
                 quaternion_rotation([float(row[f"load_quat_{axis}"]) for axis in "xyzw"]))
    return angles, robot_pose, cart_pose


def scan_case(case_dir: Path, repo: Path, stride: int = 5):
    """扫一个 case 目录，返回间隙随阶段的统计。"""
    config = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
    joint_names = config.get("policy_joint_names")
    if not joint_names:
        raise KeyError(f"{case_dir}/config.json 缺 policy_joint_names（本 run 早于间隙记录，需重跑）")
    robot, cart = build_bodies(repo)
    with (case_dir / "tow.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    per_phase = {}
    samples = []
    for index, row in enumerate(rows):
        # 必须采样**最后一行**：`final_gap_x_m` / `final_clearance_m` 取的就是它的值，
        # 抽稀时如果只按 `index % stride` 过滤，末行可能被跳过（第一版就漏了这一条）
        if index % stride and index != len(rows) - 1:
            continue
        angles, robot_pose, cart_pose = row_state(row, joint_names)
        gap, distance = clearance(robot.points(angles, *robot_pose), cart.points(WHEEL_JOINT_ANGLES, *cart_pose))
        samples.append((row["phase"], float(row["time_s"]), gap, distance,
                        float(row["rope_distance_m"])))
        per_phase.setdefault(row["phase"], []).append(gap)
    last = samples[-1]
    coast = [s for s in samples if s[0] == "coast"]
    result = {
        "samples": len(samples),
        "final_gap_x_m": last[2],
        "final_rope_distance_m": last[4],
        "min_gap_x_m": min(s[2] for s in samples),
        "min_gap_x_coast_m": min((s[2] for s in coast), default=None),
        "final_min_distance_m": last[3],
        "min_distance_m": min(s[3] for s in samples),
    }
    return result


def repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "imgo2_description").is_dir():
            return candidate
    raise RuntimeError("找不到仓库根：向上没有含 imgo2_description/ 的目录")


# ------------------------------------------------------------------ 自检
def selftest() -> int:
    """离线自检：几何与解析结论、以及「记录缺列必须报错」的负例。"""
    robot, cart = build_bodies(repo_root())
    nominal = {}
    for leg in ("FL", "FR", "RL", "RR"):
        nominal[f"{leg}_hip_joint"] = 0.0
        nominal[f"{leg}_thigh_joint"] = 0.87
        nominal[f"{leg}_shank_joint"] = -1.82

    base_pose = ([0.0, 0.0, 0.2735], IDENTITY)
    cart_pose = ([0.0, 0.0, 0.15], IDENTITY)
    rp = robot.points(nominal, *base_pose)
    cp = cart.points(WHEEL_JOINT_ANGLES, *cart_pose)
    rear = min(p[0] for p in rp)
    front = max(p[0] for p in cp)
    ok = True
    print(f"  默认姿态：机器人最后表面 x={rear:+.4f} m，小车最前表面 x={front:+.4f} m")
    if abs(front - 0.26) > 0.005:              # 车头 = 前轮外缘 0.18+0.08 = 0.26
        print(f"  **FAIL** 小车最前表面应为 0.26 m，实际 {front:.4f} m")
        ok = False
    if abs(rear + 0.3984) > 0.005:             # 与独立 FK 一致（文档 §5.17 的同一组数）
        print(f"  **FAIL** 机器人最后表面应为 -0.3984 m，实际 {rear:+.4f} m")
        ok = False
    # 与解析式对齐：挂点间距 0.80 m 时（跑动初始状态）间隙应为
    # 0.80 − (|−0.3984| − 0.16) − 0.25 = 0.5616 m。
    # 注意这里用的是车斗前表面 0.25，**不是**前轮外缘 0.26：前轮在 y=±0.20，
    # 而最后表面是大腿箱（y≈0.067，宽 0.032）——两者横向差 ~0.10 m 根本不重叠，
    # 所以那个角落只能撞车斗。这就是「横向分道」导致的 0.01 m 差，
    # 也说明不能用「车头 = 前轮外缘」这类单一数字去推碰撞间距。
    separated = ([0.80 + 0.41, 0.0, 0.2735], IDENTITY)   # x_R − x_L = 0.80 + 0.41
    rp2 = robot.points(nominal, *separated)
    gap2, distance2 = clearance(rp2, cp)
    print(f"  挂点间距 0.80 m 时：纵向间隙={gap2:+.4f} m（解析 0.5616），三维最近={distance2:+.4f} m")
    if abs(gap2 - 0.5616) > 0.01:
        print(f"  **FAIL** 纵向间隙应为 0.5616 m，实际 {gap2:.4f} m")
        ok = False
    # 平移不变性 / 零点定义：把小车沿 +x 移动 gap2，间隙应恰为 0
    shifted = [[p[0] + gap2, p[1], p[2]] for p in cp]
    gap3, _ = clearance(rp2, shifted)
    if abs(gap3) > 1e-9:
        print(f"  **FAIL** 车沿 +x 平移 {gap2:.4f} m 后间隙应为 0，实际 {gap3:.6f}")
        ok = False
    # 负例：缺关节列必须报错，不能静默当成 0
    try:
        row_state({"robot_x_m": 0, "robot_z_m": 0, "load_x_m": 0, "load_z_m": 0,
                   "robot_quat_x": 0, "robot_quat_y": 0, "robot_quat_z": 0, "robot_quat_w": 1,
                   "load_quat_x": 0, "load_quat_y": 0, "load_quat_z": 0, "load_quat_w": 1},
                  ["FL_hip_joint"])
    except KeyError:
        pass
    else:
        print("  **FAIL** 缺关节列时未报错")
        ok = False
    print("  selftest:", "OK" if ok else "**FAILED**")
    return 0 if ok else 1


def main(argv) -> int:
    if len(argv) == 1 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(argv) > 1 else 2
    if argv[1] == "--selftest":
        return selftest()
    cases = []
    for argument in argv[1:]:
        path = Path(argument)
        if (path / "tow.csv").is_file():
            cases.append(path)
        else:
            cases.extend(sorted(p for p in path.glob("*/case_*") if (p / "tow.csv").is_file()))
    if not cases:
        print("没有找到任何含 tow.csv 的 case 目录", file=sys.stderr)
        return 2
    repo = repo_root()
    print(f"{'case':<26}{'末间隙':>9}{'滑行最小':>10}{'全程最小':>10}{'三维最近':>10}{'末挂点距':>10}")
    for case in cases:
        result = scan_case(case, repo)
        # 没有 coast 段的 run 该列是 None，不能直接格式化（第一版在这里 TypeError 崩掉）
        coast = ("n/a" if result["min_gap_x_coast_m"] is None
                 else f"{result['min_gap_x_coast_m']:.4f}")
        print(f"{case.name:<26}{result['final_gap_x_m']:>9.4f}"
              f"{coast:>10}{result['min_gap_x_m']:>10.4f}"
              f"{result['min_distance_m']:>10.4f}{result['final_rope_distance_m']:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
