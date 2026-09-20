#!/usr/bin/env python3
"""只读 TensorBoard event 文件里的标量，用来盯训练（纯标准库，无需 GPU / torch / tensorboard）。

用途：训练在跑的时候，"当前效果"其实就在 `logs/amp_rsl_rl/<experiment>/<run>/events.out.tfevents.*`
里。Isaac Lab + rsl_rl 只写标量（`add_scalar`），所以一个几十行的解析器就够，不必装 tensorboard。

用法：

    # 默认取 logs/amp_rsl_rl 下**最新**的 event 文件
    python3 scripts/tools/read_tfevents.py

    # 指定 run 目录（取目录里最新的 event 文件）或直接给 event 文件
    python3 scripts/tools/read_tfevents.py --run logs/amp_rsl_rl/base_move_amp_rlamp
    python3 scripts/tools/read_tfevents.py logs/amp_rsl_rl/base_move_amp/2026-09-17_21-25-57/events.out.tfevents.123

    # 指定要看的迭代点（取"不超过该点的最后一个值"）与要匹配的 tag 子串
    python3 scripts/tools/read_tfevents.py --steps 0,500,1000,2000,3000 --match height,air,disc,error
    python3 scripts/tools/read_tfevents.py --tags          # 只列出全部 tag

文件格式（这就是"为什么能只用标准库"）：

    event 文件 = 若干 TFRecord 帧，每帧是
        uint64 length | uint32 masked_crc(length) | length 字节 protobuf | uint32 masked_crc(data)
    protobuf 是 `tensorflow.Event`，本工具只用到三个字段：
        field 1 (double) = wall_time
        field 2 (varint) = step
        field 5 (message) = Summary，其中 repeated field 1 = Value
            Value.field 1 (string) = tag, Value.field 2 (float) = simple_value
    CRC 只拿来跳过（不校验），所以尾部"正在写一半"的帧会被安全丢弃 —— 训练进行中也能读。

限制：只认 `simple_value` 标量（rsl_rl 正好只写这个）；histogram / image / tensor 一律跳过。

⚠️ **同一个文件里有两种 x 轴**：rsl_rl 里绝大多数 tag 用**迭代轮号**当 x，但
`Train/mean_reward/time`、`Train/mean_episode_length/time` 用的是 `self.tot_time`
（**墙钟秒**）—— 两者相差一个"每轮秒数"（本仓库约 1.25），所以"所有 tag 的最大 step"
并不是迭代数。本工具因此把表头写成 `iterations=<取自某个迭代轴 tag 的值>`，并把带
`/time` 后缀的 tag 单独点名（表格里每个 tag 仍按**它自己的** x 轴取值）。
"""

from __future__ import annotations

import argparse
import glob
import os
import struct
import sys

# 默认想看的量（"当前效果"最关心的几组）
DEFAULT_MATCH = (
    "mean_root_height", "fraction_root_height", "mean_last_air", "mean_air_time_fraction",
    "disc_expert_pred", "disc_policy_pred", "mean_amp_disc_pred", "mean_task_reward_step",
    "weighted_task_reward_step", "error_vel", "Episode_Termination", "Episode_Reward",
    "Loss/value_function", "Loss/AMP", "learning_rate", "mean_episode_length",
)


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    """读一个 protobuf varint，返回 (值, 新下标)。"""
    result = shift = 0
    while True:
        if i >= len(buf):
            raise ValueError("varint 越界（文件被截断？）")
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes):
    """迭代一个 protobuf message 的 (field_number, wire_type, value)。

    wire_type: 0 = varint（值给 int）；1 = 64-bit（值给 float）；2 = length-delimited（值给 bytes）；
    5 = 32-bit（值给 float）。其它类型直接抛错 —— 本工具要处理的字段不会用到它们。
    """
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
            yield field, wire, value
        elif wire == 1:
            yield field, wire, struct.unpack("<d", buf[i:i + 8])[0]
            i += 8
        elif wire == 2:
            length, i = _varint(buf, i)
            yield field, wire, buf[i:i + length]
            i += length
        elif wire == 5:
            yield field, wire, struct.unpack("<f", buf[i:i + 4])[0]
            i += 4
        else:
            raise ValueError(f"不支持的 protobuf wire type: {wire}")


def iter_records(path: str):
    """迭代 event 文件里的 TFRecord 载荷（丢弃写了一半的尾帧）。"""
    with open(path, "rb") as handle:
        data = handle.read()
    i = 0
    while i + 12 <= len(data):
        length = struct.unpack("<Q", data[i:i + 8])[0]
        i += 12                      # 8 字节长度 + 4 字节 masked CRC
        if length <= 0 or i + length + 4 > len(data):
            return                   # 尾部不完整（训练还在写）⇒ 正常结束
        yield data[i:i + length]
        i += length + 4              # 载荷 + 4 字节 masked CRC


def read_scalars(path: str) -> dict[str, list[tuple[int, float]]]:
    """读出 {tag: [(step, value), ...]}（按文件中出现顺序）。"""
    series: dict[str, list[tuple[int, float]]] = {}
    for payload in iter_records(path):
        step = None
        values: dict[str, float] = {}
        for field, wire, value in _fields(payload):
            if field == 2 and wire == 0:
                step = value
            elif field == 5 and wire == 2:
                for sub_field, sub_wire, sub_value in _fields(value):
                    if sub_field != 1 or sub_wire != 2:
                        continue
                    tag = simple = None
                    for leaf_field, leaf_wire, leaf_value in _fields(sub_value):
                        if leaf_field == 1 and leaf_wire == 2:
                            tag = leaf_value.decode("utf-8", "replace")
                        elif leaf_field == 2 and leaf_wire == 5:
                            simple = leaf_value
                    if tag is not None and simple is not None:
                        values[tag] = simple
        if step is None:
            continue
        for tag, val in values.items():
            series.setdefault(tag, []).append((step, val))
    return series


def resolve_event_file(run: str | None, root: str) -> str:
    """把 `--run` 解析成一个具体的 event 文件；为空时取 `root` 下最新的那个。

    `--root` 的默认值是 `logs/amp_rsl_rl`（相对当前工作目录）。仓库里训练日志有**三个可能位置**：
    从仓库根跑训练落在 `<repo>/logs/...`，从 `imgo2_rl/` 跑则落在 `<repo>/imgo2_rl/logs/...`
    （`train.py` 用的是相对路径），所以在 `imgo2_rl/` 里跑本脚本时还要能看见上一级的 `logs/`。
    因此默认依次试 `logs/amp_rsl_rl`、`imgo2_rl/logs/amp_rsl_rl`、`../logs/amp_rsl_rl`，
    并在所有命中里取**最新的那个 event 文件**——省得每次先猜 cwd。
    """
    if run:
        if os.path.isfile(run):
            return run
        hits = sorted(glob.glob(os.path.join(run, "**", "events.out.tfevents.*"), recursive=True),
                      key=os.path.getmtime)
        if not hits:
            raise SystemExit(f"{run} 下找不到 events.out.tfevents.*")
        return hits[-1]
    candidates = [root] if root != "logs/amp_rsl_rl" else [
        root, os.path.join("imgo2_rl", root), os.path.join(os.pardir, root)]
    hits = [path for base in candidates
            for path in glob.glob(os.path.join(base, "*", "*", "events.out.tfevents.*"))]
    if not hits:
        raise SystemExit(f"在 {candidates} 下找不到任何 event 文件（用 --run 指定）")
    return max(hits, key=os.path.getmtime)


CLOCK_AXIS_SUFFIX = "/time"
# 优先用来代表「迭代轮号」的 tag（rsl_rl 每轮必写、且 x 轴就是 locs['it']）
ITERATION_AXIS_TAGS = ("Train/mean_reward", "Loss/value_function")


def iteration_step(series: dict[str, list[tuple[int, float]]]) -> tuple[int, str]:
    """给出「当前跑到第几轮」以及这个数字的来源 tag。

    不能直接对全部 tag 取 max：`*/time` 系列用 `tot_time`（墙钟秒）当 x 轴，数值比轮号大
    （本仓库约 ×1.25），会把进度报高。优先用已知的迭代轴 tag，其次在非 `*/time` 的 tag 里取
    最大的末 step。
    """
    for tag in ITERATION_AXIS_TAGS:
        if series.get(tag):
            return series[tag][-1][0], tag
    candidates = [(points[-1][0], tag) for tag, points in series.items()
                  if points and not tag.endswith(CLOCK_AXIS_SUFFIX)]
    if candidates:
        step, tag = max(candidates)
        return step, tag
    if series:
        points = max(series.values(), key=lambda pts: pts[-1][0] if pts else -1)
        return points[-1][0], "(未知，可能是时钟轴)"
    return -1, "(空文件)"


def value_at(points: list[tuple[int, float]], step: int) -> float:
    """取"不超过 step 的最后一个值"；没有则返回 nan。"""
    got = [val for at, val in points if at <= step]
    return got[-1] if got else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("event_file", nargs="?", help="直接给 event 文件（也可用 --run）")
    parser.add_argument("--run", help="run 目录（或实验目录），取其中最新的 event 文件")
    parser.add_argument("--root", default="logs/amp_rsl_rl", help="不给 --run 时的搜索根（默认 logs/amp_rsl_rl）")
    parser.add_argument("--steps", default="", help="逗号分隔的迭代点，例如 0,500,1000,2000,3000")
    parser.add_argument("--match", default="", help="逗号分隔的 tag 子串（默认一组常用量；--all 看全部）")
    parser.add_argument("--tags", action="store_true", help="只列出全部 tag 名")
    parser.add_argument("--all", action="store_true", help="打印所有 tag（配合 --steps）")
    args = parser.parse_args(argv)

    path = resolve_event_file(args.run or args.event_file, args.root)
    series = read_scalars(path)
    last_step, axis_tag = iteration_step(series)
    print(f"# {path}")
    print(f"# iterations={last_step}（x 轴取自 {axis_tag}）, tags={len(series)}")
    clock_tags = sorted(tag for tag in series if tag.endswith(CLOCK_AXIS_SUFFIX))
    if clock_tags:
        print(f"# 注意：{len(clock_tags)} 个 tag 的 x 轴是墙钟秒而不是迭代轮号（不要拿它们当进度）："
              + ", ".join(clock_tags))

    if args.tags:
        for tag in sorted(series):
            print(tag)
        return 0

    steps = [int(x) for x in args.steps.split(",") if x.strip()] or [last_step]
    if args.all:
        picks = sorted(series)
    else:
        needles = tuple(x.strip() for x in args.match.split(",") if x.strip()) or DEFAULT_MATCH
        picks = [tag for tag in sorted(series) if any(n in tag for n in needles)]
    if not picks:
        print("（没有匹配的 tag，用 --tags 看全部）")
        return 0

    header = "tag".ljust(58) + "".join(f"{step:>11}" for step in steps) + f"{'last':>11}"
    print(header)
    print("-" * len(header))
    for tag in picks:
        points = series[tag]
        cells = "".join(f"{value_at(points, step):>11.4f}" for step in steps)
        print(tag.ljust(58) + cells + f"{points[-1][1]:>11.4f}")
    print(f"# last 列取的是最后一条记录（step={last_step}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
