"""Small, stdlib-only record writer for reproducible cart experiments."""

import csv
import json
import math
from pathlib import Path

LEGS = ("fl", "fr", "rl", "rr")

# 机器人 12 个关节角的列名。列名生成只放这一处：入口填列、离线工具 FK、以及
# 「入口字段集 == 记录列集」这条离线契约测试都引用它，避免三处各写一遍 f-string 漂移。
ROBOT_JOINT_POSITION_FIELDS = tuple(f"robot_jp_{index:02d}" for index in range(12))


def joint_position_fields(values):
    """把策略顺序的 12 个关节角转成记录用的字典。

    键的生成放在这里（而不是入口里写推导式），于是离线测试可以直接调用**生产函数**
    核对入口写出的列，不必去解析源码里的 f-string。
    """
    values = list(values)
    if len(values) != len(ROBOT_JOINT_POSITION_FIELDS):
        raise ValueError(f"契约要求 {len(ROBOT_JOINT_POSITION_FIELDS)} 个关节角，收到 {len(values)}")
    return {name: float(value) for name, value in zip(ROBOT_JOINT_POSITION_FIELDS, values)}


FIELDS = (
    "time_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
    "roll_rad", "pitch_rad", "yaw_rad", "wx_radps", "wy_radps", "wz_radps",
    *(f"{leg}_omega_radps" for leg in LEGS),
    *(f"{leg}_tau_nm" for leg in LEGS),
    *(f"{leg}_normal_n" for leg in LEGS),
)


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


class CartRecorder:
    def __init__(self, directory: Path, config: dict):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        write_json(directory / "config.json", config)
        self._stream = (directory / "trajectory.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=FIELDS)
        self._writer.writeheader()
        self._last_time = -1.0

    def append(self, row: dict):
        if set(row) != set(FIELDS) or not all(math.isfinite(float(row[k])) for k in FIELDS):
            raise ValueError("Invalid/non-finite cart sample")
        if row["time_s"] <= self._last_time:
            raise ValueError("Cart sample times must strictly increase")
        self._writer.writerow(row)
        self._stream.flush()
        self._last_time = row["time_s"]

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# 拖曳记录（计划 P5 的字段表，P4 先只填其中不需要步态指标的部分）。
# `ref_cmd_mps` 在 P4 恒等于用户指令（还没有 command shaping），留列是为了 P6/P8。
# `phase` 标出阶段（station/站定、tow/拖曳、coast/指令归零后的滑行）：计划 P6 的
# 「t≥5 s 后 T 多快归零、机器人有无瞬态」都要按阶段取窗口，写进数据比事后猜边界可靠。
TOW_NUMERIC_FIELDS = (
    "time_s", "user_cmd_mps", "ref_cmd_mps", "robot_vx_mps", "load_vx_mps",
    "rope_tension_n", "rope_distance_m", "robot_x_m", "load_x_m",
    # 高度单列出来：机器人被拽倒/塌下去时，只看 pitch 不够直观
    "robot_z_m", "load_z_m",
    # 四个轮速单列：v/r 与 ω 的比值能区分「滚动」与「滑动」，是诊断低速段
    # 与解析预测不符（实测提前硬停）的关键量
    *(f"wheel_{leg}_omega_radps" for leg in LEGS),
    "body_pitch_rad", "body_pitch_rate_radps",
    # ---------------------------------------------------------------- 停车距离度量
    # 「小车是否追到机器人」「最终停车距离机器人的位置」不能用挂点间距判：
    # 挂点间距只是两个**挂点**的距离，而真实接触取决于车头与机器人后腿的几何。
    # 二者相差一个随姿态变化的量（实测 d_contact = 0.12~0.44 m，见
    # docs/towing_p4_tow_drag_2026-09-20.md §5.17），所以必须记录算出真实间隙所需的
    # 全部状态量：12 个关节角（决定机器人后腿伸到哪里）+ 双方完整姿态。
    # 关节角按**策略/契约顺序**存（policy_cfg.joint_names，逐腿 FL/FR/RL/RR），
    # 顺序随 config.json 的 `policy_joint_names` 一起落盘，不靠约定记忆。
    *ROBOT_JOINT_POSITION_FIELDS,
    "robot_quat_x", "robot_quat_y", "robot_quat_z", "robot_quat_w",
    "load_quat_x", "load_quat_y", "load_quat_z", "load_quat_w",
    # 撞击的独立见证（不依赖任何 FK/几何建模）：
    # `cart_deck_fx_n` 是车斗 base_link 受到的接触合力在 x 的分量 —— 车斗永远不碰地面，
    # 所以它非零**只可能**是机器人压上来了；`cart_wheel_fx_n` 是四个轮子的同类分量之和，
    # 正常滚动时只有 ~1.7 N（用来克服轮阻），撞击时会跳到几十 N。
    "cart_deck_fx_n", "cart_wheel_fx_n",
)
TOW_FIELDS = ("phase", *TOW_NUMERIC_FIELDS)


class TowRecorder:
    """逐物理步记录机器人/负载/绳状态，写进**调用方已独占创建**的实验目录。

    与 `CartRecorder` 的差别在目录归属：`CartRecorder` 服务「一个运行里多个 case 子目录」，
    所以每个实例自己建目录；拖曳运行只有一个目录，由入口用 `mkdir(exist_ok=False)`
    独占创建（保证新运行绝不落进已有目录），这里只做写入。因此本类要求目录**已存在**，
    并额外拒绝覆盖已存在的 `tow.csv`/`config.json`（写坏已有产物要提前拦住）。

    （第一版照抄了 `CartRecorder` 的「自己建目录」语义，于是入口先建目录、recorder 再建
    一次，实跑时报 `FileExistsError` 自己撞自己；离线单测当时只单独构造 recorder，没覆盖
    这条集成路径。）
    """

    def __init__(self, directory: Path, config: dict):
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"记录目录必须已存在（由调用方用 exist_ok=False 独占创建）：{directory}")
        for name in ("tow.csv", "config.json"):
            if (directory / name).exists():
                raise FileExistsError(f"拒绝覆盖已有产物：{directory / name}")
        self.directory = directory
        write_json(directory / "config.json", config)
        self._stream = (directory / "tow.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=TOW_FIELDS)
        self._writer.writeheader()
        self._last_time = -1.0

    def append(self, row: dict):
        if set(row) != set(TOW_FIELDS):
            raise ValueError("Invalid tow sample fields")
        if not all(math.isfinite(float(row[k])) for k in TOW_NUMERIC_FIELDS):
            raise ValueError("Invalid/non-finite tow sample")
        if row["phase"] not in ("station", "tow", "coast"):
            raise ValueError(f"Unknown phase: {row['phase']!r}")
        if row["time_s"] <= self._last_time:
            raise ValueError("Tow sample times must strictly increase")
        self._writer.writerow(row)
        self._stream.flush()
        self._last_time = row["time_s"]

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
