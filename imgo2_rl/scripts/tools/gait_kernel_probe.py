"""离线反演 CMoE 的步态相位核：给定「周期 / 占空比 / 步态」，算出三态读数应该长什么样。

**为什么需要它**：`gait_metric_{trot,bound,pace}`（三个 `GaitReward`，同一套核、只换配对方式）
是判读"学到什么步态"的唯一分类器，而它们的读数**同时**取决于步态形状与周期/占空比——

* 干净 trot 在 `T=0.93 s / duty=0.5` 下应当是 `trot 1.000 / bound 0.280`（差 0.72，一眼可辨）；
* 但在 `T=0.34 s / duty=0.5` 下只剩 `1.000 / 0.676`（差 0.324）；
* **四足锁相（pronk）在 duty≈0.5 时三种配对完全相等**（本工具实测 0.676/0.676/0.676）——
  因为"反相核"比的是**一只脚的滞空时间 vs 另一只脚的触地时间**，duty≈0.5 时滞空≈触地，
  于是无论怎么配对都近似满足，只剩 2 个"同步核"在分辨。

所以看到三态读数**接近相等**时，正确的结论是"步态既非 trot 也非 bound/pace（近锁相）"，
而不是"哪个略高就是哪种步态"。run H（2026-09-24_22-08-42，1849 轮）实测
`trot 0.768 / bound 0.782 / pace 0.765` 正是这个签名。

核公式逐字取自 `mdp/rewards.py::GaitReward`（`_sync_reward_func` / `_async_reward_func`），
`std`/`max_err` **从 `CMoE_env_cfg.py` 读**（三处 `gait_metric_*` ＋ `feet_gait` 的 post_init
覆盖必须一致，本工具顺带把这条不变量检查掉）。纯标准库，无需 torch / Isaac Lab。

用法：

    python scripts/tools/gait_kernel_probe.py                       # 默认 T=0.34s, duty=0.5
    python scripts/tools/gait_kernel_probe.py --period 0.93
    python scripts/tools/gait_kernel_probe.py --observed 0.768,0.782,0.765   # 反查实测读数像哪种步态
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]  # imgo2_rl/
CMOE_CFG = ROOT / "source/imgo2_rl/imgo2_rl/tasks/manager_based/locomotion/velocity/base_move/CMoE_env_cfg.py"

FEET = ("FL", "FR", "RL", "RR")
# 三种配对方式的定义（结构性事实；具体哪只脚对哪只脚从配置里读出来核对）
PAIR_LABELS = ("trot", "bound", "pace")


def read_kernel_params(path: Path) -> tuple[float, float, list[tuple[str, tuple[tuple[str, str], tuple[str, str]]]]]:
    """从 CMoE_env_cfg 读出 (std, max_err) 与三个 `gait_metric_*` 的配对方式。

    `std`/`max_err` 必须**四处一致**（三处 metric ＋ `feet_gait` 的 post_init 覆盖），
    否则分类器读数不再代表奖励 —— 这是配置注释里的显式不变量，本函数不一致就报错。
    """
    text = path.read_text(encoding="utf-8")
    grouped = re.findall(r'"std":\s*([0-9.]+)', text)
    overridden = re.findall(r'feet_gait\.params\["std"\]\s*=\s*([0-9.]+)', text)
    max_errs = re.findall(r'"max_err":\s*([0-9.]+)', text)
    max_err_overrides = re.findall(r'feet_gait\.params\["max_err"\]\s*=\s*([0-9.]+)', text)
    values = {float(v) for v in grouped + overridden}
    caps = {float(v) for v in max_errs + max_err_overrides}
    if len(values) != 1 or len(caps) != 1:
        raise ValueError(
            f"相位核参数不一致：std={sorted(values)}（{len(grouped)} 处 metric + {len(overridden)} 处 feet_gait），"
            f"max_err={sorted(caps)}（{len(max_errs)} + {len(max_err_overrides)}）"
        )

    declarations: list[tuple[str, tuple[tuple[str, str], tuple[str, str]]]] = []
    for label in PAIR_LABELS:
        block = re.search(rf"gait_metric_{label}\s*=\s*RewTerm\((.*?)\n    \)", text, re.DOTALL)
        if block is None:
            raise ValueError(f"配置里找不到 gait_metric_{label} 的 RewTerm")
        pairs = re.search(r"synced_feet_pair_names.*?\]|\"synced_feet_pair_names\":\s*(\(.*?\)),\n", block.group(1), re.DOTALL)
        names = re.findall(r'"([A-Z]{2})_FOOT"', block.group(1))
        if pairs is None or len(names) != 4:
            raise ValueError(f"gait_metric_{label} 的 synced_feet_pair_names 解析失败（拿到 {names}）")
        declarations.append((label, ((names[0], names[1]), (names[2], names[3]))))
    return next(iter(values)), next(iter(caps)), declarations


def clip_sq(delta: float, cap: float) -> float:
    return min(delta * delta, cap)


def kernel_product(std: float, cap: float, air: dict, con: dict,
                   pairs: tuple[tuple[str, str], tuple[str, str]], kernel: str = "gauss") -> float:
    """`GaitReward.__call__` 的 6 核乘积（2 同步 × 4 反相），逐字复刻。

    * ``kernel="gauss"``（上游形式）＝ `exp(−se/std)`，地板 `exp(−2·max_err²/std)`；
    * ``kernel="hinge"``（2026-09-25 试的备选）＝ `max(0, 1 − se/max_err²)`：**只用 `max_err`**、
      值域 [0, 1]、地板天然为 0（见 docs §17 的实测对照）。
    """
    (a0, a1), (b0, b1) = pairs

    def _score(se: float) -> float:
        if kernel == "hinge":
            return max(0.0, 1.0 - se / cap)
        return pow(2.718281828459045, -se / std)

    def sync(f0: str, f1: str) -> float:
        return _score(clip_sq(air[f0] - air[f1], cap) + clip_sq(con[f0] - con[f1], cap))

    def asynch(f0: str, f1: str) -> float:
        return _score(clip_sq(air[f0] - con[f1], cap) + clip_sq(con[f0] - air[f1], cap))

    return (sync(a0, a1) * sync(b0, b1)
            * asynch(a0, b0) * asynch(a1, b1) * asynch(a0, b1) * asynch(b0, a1))


def simulate(std: float, cap: float, declarations, offsets: dict, period: float, duty: float,
             dt: float = 0.02, steps: int = 6000) -> dict[str, float]:
    """按相位模型推进四足接触状态，返回三个声明的 6 核乘积时间平均（后一半用于统计）。"""
    air = {f: 0.0 for f in FEET}
    con = {f: 0.0 for f in FEET}
    totals = {label: 0.0 for label in PAIR_LABELS}
    kept = 0
    for index in range(steps):
        time = index * dt
        for foot in FEET:
            if ((time / period + offsets[foot]) % 1.0) < duty:
                con[foot] += dt
                air[foot] = 0.0
            else:
                air[foot] += dt
                con[foot] = 0.0
        if index < steps // 2:
            continue
        for label, pairs in declarations:
            totals[label] += kernel_product(std, cap, air, con, pairs)
        kept += 1
    return {label: totals[label] / kept for label in PAIR_LABELS}


def gait_offsets(name: str) -> dict[str, float]:
    """四种候选步态的相位偏置（trot/bound/pace 各自"两对同相、两对反相"）。"""
    if name == "trot":
        return {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}
    if name == "bound":
        return {"FL": 0.0, "FR": 0.0, "RL": 0.5, "RR": 0.5}
    if name == "pace":
        return {"FL": 0.0, "RL": 0.0, "FR": 0.5, "RR": 0.5}
    if name == "lockstep":  # pronk 的理想化：四足完全同相
        return {foot: 0.0 for foot in FEET}
    raise ValueError(f"未知步态 {name}")


def shape(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    return [v / mean if mean else 0.0 for v in values]


def replay_contacts(contact, dt: float, std: float, cap: float, declarations,
                    ep_len=None, done=None, min_ep_len: int = 0,
                    kernel: str = "gauss") -> dict[str, float]:
    """按 Isaac 的 `current_air_time`/`current_contact_time` 语义**回放真实接触序列**。

    2026-09-25 新增：只看理想化步态不够 —— 用某个策略**自己的**接触序列（例如 `eval_gait.py
    --dump-npz` 产出的 `contact`，形如 `(T, N, 4)`，列序 FL/FR/RL/RR）回放，才能回答
    "这份策略在这三个系数下本来该读多少"。

    返回三个声明的读数，外加两个诊断量：`mismatch`（六对 `|Δair|+|Δcon|` 的时间平均，秒）与
    `last_air_time`（**完整**腾空时长均值，与 `diag_air_time` 同口径）。
    """
    import numpy as np  # 只有这条路径需要 numpy（模拟表仍是纯标准库）

    frames = np.asarray(contact)
    steps, num_envs, num_feet = frames.shape
    totals = {label: 0.0 for label in PAIR_LABELS}
    mismatch_total = 0.0
    air_total = 0.0
    samples = 0
    for env_id in range(num_envs):
        air = [0.0] * num_feet
        con = [0.0] * num_feet
        last_air = [0.0] * num_feet
        was_contact = [False] * num_feet
        for step in range(steps):
            if min_ep_len and ep_len is not None and int(ep_len[step, env_id]) < min_ep_len:
                continue
            if done is not None and int(done[step, env_id]):
                continue
            for foot in range(num_feet):
                in_contact = bool(frames[step, env_id, foot])
                if in_contact:
                    if not was_contact[foot]:
                        # 落地这一步：`last_air_time` 记的是**刚结束**的那段腾空（Isaac 语义）
                        last_air[foot] = air[foot]
                    con[foot] += dt
                    air[foot] = 0.0
                else:
                    air[foot] += dt
                    con[foot] = 0.0
                was_contact[foot] = in_contact
            air_map = {FEET[i]: air[i] for i in range(num_feet)}
            con_map = {FEET[i]: con[i] for i in range(num_feet)}
            for label, pairs in declarations:
                totals[label] += kernel_product(std, cap, air_map, con_map, pairs, kernel)
            mismatch_total += sum(
                abs(air[i] - air[j]) + abs(con[i] - con[j])
                for i in range(num_feet) for j in range(i + 1, num_feet)
            ) / max(num_feet * (num_feet - 1) // 2, 1)
            air_total += sum(last_air) / num_feet
            samples += 1
    if samples == 0:
        raise ValueError("回放没有有效样本（min_ep_len/done 过滤掉了全部帧？）")
    result = {label: totals[label] / samples for label in PAIR_LABELS}
    result["mismatch"] = mismatch_total / samples
    result["last_air_time"] = air_total / samples
    result["samples"] = float(samples)
    return result


def verdict(values: dict[str, float]) -> str:
    """按三态读数给一句判读（领先量 > 0.05＝真领先；三态差 < 0.02＝锁相/乱走）。"""
    spread = max(values[label] for label in PAIR_LABELS) - min(values[label] for label in PAIR_LABELS)
    margin = values["trot"] - values["bound"]
    if spread < 0.02:
        return "三态相等 ⇒ 锁相/乱走（既不是 trot 也不是 bound/pace）"
    if margin > 0.05:
        return f"trot 领先 +{margin:.3f} ⇒ 与 trot 类先验一致"
    if margin < -0.05:
        return f"bound 领先 {-margin:.3f} ⇒ 偏向 bound"
    return f"三者接近、trot−bound = {margin:+.3f} ⇒ 介于 trot 与其它之间"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线反演 CMoE 步态相位核的三态读数。")
    parser.add_argument("--period", type=float, default=0.34, help="步周期（秒）。默认 0.34（run H 空时 0.169 s、duty≈0.5 反推）")
    parser.add_argument("--duty", type=float, default=0.5, help="触地占空比。默认 0.5")
    parser.add_argument("--dt", type=float, default=0.02, help="仿真步长（训练侧 0.02 s）")
    parser.add_argument("--observed", type=str, default=None,
                        help="实测三态读数 `trot,bound,pace`（按这个顺序），用来反查最接近哪种步态（按形状归一化比较）")
    parser.add_argument("--cfg", type=Path, default=CMOE_CFG)
    parser.add_argument("--npz", type=Path, default=None,
                        help="用真实接触序列回放（如 eval_gait.py --dump-npz 的产物，需含 `contact` (T,N,4)"
                             " 与 `ep_len`/`done`）：给出**这份策略自己**在这三个系数下的读数与六对时间差。")
    parser.add_argument("--min-ep-len", type=int, default=50,
                        help="--npz 回放时丢掉回合前 N 步（warmup/重置瞬态）。默认 50。")
    parser.add_argument("--kernel", choices=("gauss", "hinge"), default="gauss",
                        help="回放用的核：gauss＝上游形式（std 决定分辨率、max_err 决定地板）；"
                             "hinge＝只用 max_err 的归一化二次核（地板天然 0，见 docs §17）。")
    parser.add_argument("--sweep", action="store_true",
                        help="配 --npz：扫描若干 (std, max_err, kernel) 组合，给出各组合下**这份策略自己的**"
                             "三态读数与 trot−bound 溢价（weight=1.0 时溢价就是 /s）。")
    args = parser.parse_args(argv)

    try:
        std, max_err, declarations = read_kernel_params(args.cfg)
    except (ValueError, FileNotFoundError) as error:
        print(f"❌ 读配置失败：{error}")
        return 1
    cap = max_err ** 2  # `clip(Δ², max=max_err²)`：传给核的是**平方上界**

    print(f"配置来源 : {args.cfg.relative_to(ROOT.parent)}")
    print(f"相位核   : std={std}, max_err={max_err}（四处一致：3 个 gait_metric_* + feet_gait）")
    for label, pairs in declarations:
        print(f"  {label:5s} 配对 = {pairs[0][0]}↔{pairs[0][1]}, {pairs[1][0]}↔{pairs[1][1]}")
    print(f"\n模拟     : period={args.period} s, duty={args.duty}, dt={args.dt} s")

    results: dict[str, dict[str, float]] = {}
    print(f"\n{'步态':<10}{'trot':>8}{'bound':>8}{'pace':>8}   形状判定")
    for name in ("trot", "bound", "pace", "lockstep"):
        row = simulate(std, cap, declarations, gait_offsets(name), args.period, args.duty, args.dt)
        results[name] = row
        spread = max(row.values()) - min(row.values())
        judgement = "三态相等（不可分辨）" if spread < 0.02 else f"领先 {max(row, key=row.get)} +{spread:.3f}"
        print(f"{name:<10}{row['trot']:>8.3f}{row['bound']:>8.3f}{row['pace']:>8.3f}   {judgement}")

    trot_row = results["trot"]
    advantage = trot_row["trot"] - max(trot_row["bound"], trot_row["pace"])
    print(f"\n判据：干净 trot 下 `trot` 声明的领先量 = {advantage:.3f}/s（权重 1.0 ⇒ 约 {advantage:.2f}/s 的边际奖励）")
    if advantage < 0.1:
        print("  ⇒ 该周期/占空比下相位项几乎无法分辨步态，继续调核参数的收益有限。")

    if args.observed:
        try:
            observed = [float(v) for v in args.observed.split(",")]
        except ValueError:
            print(f"\n❌ --observed 需要三个数，例如 0.768,0.782,0.765；收到 {args.observed}")
            return 1
        if len(observed) != 3:
            print(f"\n❌ --observed 需要三个数（trot,bound,pace），收到 {len(observed)} 个")
            return 1
        target = shape(observed)
        print(f"\n实测     : trot={observed[0]:.3f} bound={observed[1]:.3f} pace={observed[2]:.3f}"
              f"（形状归一化后 {target[0]:.3f}/{target[1]:.3f}/{target[2]:.3f}）")
        ranking = sorted(
            results.items(),
            key=lambda item: sum((a - b) ** 2 for a, b in zip(shape(list(item[1].values())), target)),
        )
        for name, row in ranking:
            distance = sum((a - b) ** 2 for a, b in zip(shape(list(row.values())), target)) ** 0.5
            print(f"  {name:<10} 形状距离 {distance:.4f}")
        print(f"  ⇒ 最接近：**{ranking[0][0]}**")

    if args.npz:
        try:
            import numpy as np
        except ModuleNotFoundError:
            print("\n❌ --npz 需要 numpy（模拟表仍可无 numpy 使用）")
            return 1
        if not args.npz.is_file():
            print(f"\n❌ 找不到 {args.npz}")
            return 1
        data = np.load(args.npz, allow_pickle=True)
        if "contact" not in data.files:
            print(f"\n❌ {args.npz} 里没有 `contact`（有 {data.files}）")
            return 1
        replay = replay_contacts(
            data["contact"], args.dt, std, cap, declarations,
            ep_len=data["ep_len"] if "ep_len" in data.files else None,
            done=data["done"] if "done" in data.files else None,
            min_ep_len=args.min_ep_len, kernel=args.kernel,
        )
        print(f"\n实测回放 : {args.npz}（{int(replay['samples'])} 帧有效样本，丢前 {args.min_ep_len} 步）")
        print(f"  三态读数   trot={replay['trot']:.3f}  bound={replay['bound']:.3f}  pace={replay['pace']:.3f}"
              f"   领先量 trot−bound = {replay['trot'] - replay['bound']:+.3f}")
        print(f"  诊断量     六对 |Δair|+|Δcon| 均值 = {replay['mismatch']:.4f} s"
              f"（这就是 `max_err` 该取的量级；当前 max_err={max_err}）")
        print(f"             完整腾空时长均值 = {replay['last_air_time']:.4f} s（`diag_air_time` 同口径）")
        print(f"  判读       {verdict(replay)}")

        if args.sweep:
            grid = [("gauss", 0.2, 0.5, "现参数"), ("gauss", 0.2, 0.1, ""), ("gauss", 0.01, 0.1, ""),
                    ("gauss", 0.005, 0.1, ""), ("gauss", 0.02, 0.3, ""),
                    ("hinge", 0.0, 0.15, "无 std"), ("hinge", 0.0, 0.30, "无 std")]
            print(f"\n--sweep：同一份接触序列在不同核参数下的读数（weight=1.0 ⇒ 溢价＝奖励 /s）")
            print(f"  {'核':<22}{'trot':>7}{'bound':>7}{'pace':>7}{'溢价':>8}{'地板':>8}")
            for kind, kernel_std, max_err, note in grid:
                row = replay_contacts(
                    data["contact"], args.dt, kernel_std, max_err ** 2, declarations,
                    ep_len=data["ep_len"] if "ep_len" in data.files else None,
                    done=data["done"] if "done" in data.files else None,
                    min_ep_len=args.min_ep_len, kernel=kind,
                )
                floor = 0.0 if kind == "hinge" else pow(2.718281828459045, -2 * max_err ** 2 / kernel_std)
                tag = f"{kind}(std={kernel_std}, max_err={max_err})" + (f" {note}" if note else "")
                print(f"  {tag:<22}{row['trot']:>7.3f}{row['bound']:>7.3f}{row['pace']:>7.3f}"
                      f"{row['trot'] - row['bound']:>+8.3f}{floor:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
