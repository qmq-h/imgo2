"""Longitudinal CMoE obstacle courses scaled for the Imgo2 quadruped.

2026-09-28（用户：「我看了一下 cmoe 的地形设置，我们四足这个地形还是差点，感觉可以对齐一下」）：
本文件的结构与参考实现对齐。参考＝`Hoshi-No-Ai/CMoE`（ICRA 2026）@ `4575d6ae`：

* 比例／行列数／初始等级：`legged_gym/legged_gym/envs/g1/g1_cmoe_config.py` 的 `class terrain`；
* 每种地形的**难度律**：`legged_gym/legged_gym/utils/humanoid_terrain.py::Terrain.make_terrain`；
* 纵向障碍（gap／hurdle／step／narrow stairs／mix）的图案：
  `legged_gym/legged_gym/utils/parkour_terrain_utils.py`。

参考是骨盆高 0.75 m 的 G1，我们是站高 0.30 m 的四足 ⇒ **只对齐结构**（地形种类、比例、行列数、
难度律形式、初始等级），已有类型的米制难度区间沿用我们已验证的四足值；参考里没有现成四足对应值
的三个新增类型（`hurdle`／`mix`／`narrow_stairs`）按参考米制 × :data:`REFERENCE_SCALE` 落地。
逐项对应、已知偏离与验证方式见 `docs/cmoe_terrain_alignment_2026-09-28.md`。
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


_BASE_DEPTH = 1.0

# 站高比：参考 G1 的 base_height_target = 0.75 m（`g1_cmoe_config.py::rewards.base_height_target`），
# 我们 Imgo2 站立高度 0.30 m ⇒ 0.30/0.75 = 0.4。**只用于参考里没有四足对应值的新增类型**
# （hurdle／mix／narrow_stairs）；已有类型的难度区间是我们自己验证过的四足值，不参与缩放。
REFERENCE_SCALE = 0.4


def _platform(x0: float, x1: float, width: float, top_height: float = 0.0) -> trimesh.Trimesh:
    """Create one full-width platform segment whose top is at ``top_height``."""
    length = x1 - x0
    height = _BASE_DEPTH + top_height
    center = (0.5 * (x0 + x1), 0.5 * width, 0.5 * (top_height - _BASE_DEPTH))
    return trimesh.creation.box(
        (length, width, height), trimesh.transformations.translation_matrix(center)
    )


def _corridor(
    x0: float, x1: float, tile_width: float, corridor_width: float, top_height: float = 0.0
) -> trimesh.Trimesh:
    """Create one **partial-width** platform segment centred in ``y``（窄走廊／独木桥用）。

    参考 `parkour_terrain_utils.mix_obstacles_terrain` / `narrow_stairs_terrain` 的做法是
    「走廊内抬升、走廊外下沉」：本函数负责抬升的那一半，下沉由调用方铺一块 ``top_height<0``
    的整宽平台当坑底。
    """
    length = x1 - x0
    height = _BASE_DEPTH + top_height
    center = (0.5 * (x0 + x1), 0.5 * tile_width, 0.5 * (top_height - _BASE_DEPTH))
    return trimesh.creation.box(
        (length, corridor_width, height), trimesh.transformations.translation_matrix(center)
    )


def track_gap_terrain(difficulty: float, cfg: CMoETrackGapTerrainCfg):
    """Generate the original CMoE-style +x course with several transverse gaps."""
    gap_width = cfg.gap_width_range[0] + difficulty * (
        cfg.gap_width_range[1] - cfg.gap_width_range[0]
    )
    spacing = cfg.platform_length_range[1] - difficulty * (
        cfg.platform_length_range[1] - cfg.platform_length_range[0]
    )

    meshes: list[trimesh.Trimesh] = []
    cursor = 0.0
    platform_end = cfg.first_gap_x
    for _ in range(cfg.num_gaps):
        meshes.append(_platform(cursor, platform_end, cfg.size[1]))
        cursor = platform_end + gap_width
        platform_end = cursor + spacing
    if cursor < cfg.size[0]:
        meshes.append(_platform(cursor, cfg.size[0], cfg.size[1]))

    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], 0.0])
    return meshes, origin


@configclass
class CMoETrackGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a longitudinal course containing repeated gaps."""

    function = track_gap_terrain
    gap_width_range: tuple[float, float] = (0.08, 0.16)
    platform_length_range: tuple[float, float] = (0.65, 0.95)
    first_gap_x: float = 1.8
    num_gaps: int = 4
    spawn_x: float = 0.75


def track_step_terrain(difficulty: float, cfg: CMoETrackStepTerrainCfg):
    """Generate separated full-width steps along +x on an otherwise flat track."""
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    step_length = cfg.step_length_range[0] + difficulty * (
        cfg.step_length_range[1] - cfg.step_length_range[0]
    )

    meshes = [_platform(0.0, cfg.size[0], cfg.size[1])]
    x = cfg.first_step_x
    for _ in range(cfg.num_steps):
        if x + step_length >= cfg.size[0]:
            break
        meshes.append(_platform(x, x + step_length, cfg.size[1], step_height))
        x += step_length + cfg.step_spacing

    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], 0.0])
    return meshes, origin


@configclass
class CMoETrackStepTerrainCfg(SubTerrainBaseCfg):
    """Configuration for separated step obstacles on a longitudinal track."""

    function = track_step_terrain
    step_height_range: tuple[float, float] = (0.04, 0.12)
    step_length_range: tuple[float, float] = (0.18, 0.30)
    step_spacing: float = 0.85
    first_step_x: float = 1.8
    num_steps: int = 4
    spawn_x: float = 0.75


def track_stairs_terrain(difficulty: float, cfg: CMoETrackStairsTerrainCfg):
    """Generate one compact multi-step staircase across a +x course."""
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    total_height = cfg.num_steps * step_height
    meshes: list[trimesh.Trimesh] = []

    if cfg.ascending:
        meshes.append(_platform(0.0, cfg.stairs_start_x, cfg.size[1]))
        for index in range(cfg.num_steps):
            x0 = cfg.stairs_start_x + index * cfg.step_depth
            x1 = x0 + cfg.step_depth
            meshes.append(_platform(x0, x1, cfg.size[1], (index + 1) * step_height))
        end_x = cfg.stairs_start_x + cfg.num_steps * cfg.step_depth
        meshes.append(_platform(end_x, cfg.size[0], cfg.size[1], total_height))
        spawn_height = 0.0
    else:
        meshes.append(_platform(0.0, cfg.stairs_start_x, cfg.size[1], total_height))
        for index in range(cfg.num_steps):
            x0 = cfg.stairs_start_x + index * cfg.step_depth
            x1 = x0 + cfg.step_depth
            meshes.append(_platform(x0, x1, cfg.size[1], (cfg.num_steps - index - 1) * step_height))
        end_x = cfg.stairs_start_x + cfg.num_steps * cfg.step_depth
        meshes.append(_platform(end_x, cfg.size[0], cfg.size[1]))
        spawn_height = total_height

    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], spawn_height])
    return meshes, origin


@configclass
class CMoETrackStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a single multi-level staircase obstacle."""

    function = track_stairs_terrain
    step_height_range: tuple[float, float] = (0.025, 0.08)
    step_depth: float = 0.30
    num_steps: int = 4
    stairs_start_x: float = 2.0
    spawn_x: float = 0.75
    ascending: bool = True


# ======================================================================================
# 2026-09-28 结构对齐新增：参考 CMoE 有、我们缺的三类障碍地形。
# 参考是 `Hoshi-No-Ai/CMoE` @ `4575d6ae`，逐项来源见文件头 docstring；米制 = 参考 × REFERENCE_SCALE。
# 比例（`g1_cmoe_config.py::terrain.terrain_dict`）里这三类各占 0.1，与 `gap`(0.3) 一起构成
# 参考的 parkour 列（`non_parkour_terrain = 0.5` 之后的部分）。
# ======================================================================================


def track_hurdle_terrain(difficulty: float, cfg: CMoETrackHurdleTerrainCfg):
    """一串**薄而高**的整宽横栏（参考 `parkour_hurdle_terrain` 的缩比版）。

    参考调用（`humanoid_terrain.py:230`）::

        parkour_hurdle_terrain(num_stones=4, stone_len=0.1+0.2*d,
                               hurdle_height_range=[0.2*d, 0.15+0.25*d],
                               x_range=[1.2, 2], half_valid_width=[4, 4.5])

    ``half_valid_width`` 取 4–4.5 m 远大于瓦片半宽 ⇒ 参考里横栏也是**整宽**的（没有侧向空隙）。
    ×0.4 后：厚 0.04–0.12 m、栏高 ``[0.08*d, 0.06+0.10*d]`` m、间距 0.48–0.80 m。
    与我们已有的 `boxes`（0.18–0.30 m 厚、0.08–0.30 m 高的矮块）区别正是"更薄更高" ⇒ 跨栏而非踩台阶。
    """
    hurdle_len = cfg.stone_len_range[0] + difficulty * (cfg.stone_len_range[1] - cfg.stone_len_range[0])
    height_min = cfg.hurdle_height_min_slope * difficulty
    height_max = cfg.hurdle_height_max_base + cfg.hurdle_height_max_slope * difficulty

    meshes: list[trimesh.Trimesh] = [_platform(0.0, cfg.size[0], cfg.size[1])]
    cursor = cfg.platform_length
    for _ in range(cfg.num_hurdles):
        cursor += float(np.random.uniform(*cfg.spacing_range))
        x0, x1 = cursor - 0.5 * hurdle_len, cursor + 0.5 * hurdle_len
        if x1 >= cfg.size[0]:
            break
        meshes.append(
            _platform(x0, x1, cfg.size[1], float(np.random.uniform(height_min, height_max)))
        )

    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], 0.0])
    return meshes, origin


@configclass
class CMoETrackHurdleTerrainCfg(SubTerrainBaseCfg):
    """整宽薄横栏（跨栏）。米制常量＝参考 `parkour_hurdle_terrain` × ``REFERENCE_SCALE``。"""

    function = track_hurdle_terrain
    num_hurdles: int = 4
    # 参考 stone_len = 0.1 + 0.2*d（m）⇒ ×0.4 = 0.04 + 0.08*d
    stone_len_range: tuple[float, float] = (0.04, 0.12)
    # 参考 hurdle_height_range = [0.2*d, 0.15 + 0.25*d]（m）⇒ ×0.4 = [0.08*d, 0.06 + 0.10*d]
    hurdle_height_min_slope: float = 0.08
    hurdle_height_max_base: float = 0.06
    hurdle_height_max_slope: float = 0.10
    # 参考 x_range = [1.2, 2.0]（m，相邻横栏间距）⇒ ×0.4 = [0.48, 0.80]
    spacing_range: tuple[float, float] = (0.48, 0.80)
    # 参考 platform_len = 2.0 m（起跳前的平地）⇒ ×0.4 = 0.80 m（仍覆盖 0.75 m 出生点）
    platform_length: float = 0.80
    spawn_x: float = 0.75


# 参考 `mix_obstacles_terrain` 是**硬编码固定图案**（不是按参数生成的），单位为
# 「水平索引 × 0.05 m、高度索引 × 0.005 m」，走廊半宽 20 索引（= 1.0 m），走廊外下沉。
# 下面逐段照抄该图案（段以索引表示），再整体 × REFERENCE_SCALE。
_MIX_X_UNIT = 0.05 * REFERENCE_SCALE   # 0.02 m
_MIX_Z_UNIT = 0.005 * REFERENCE_SCALE  # 0.002 m


def track_mix_terrain(difficulty: float, cfg: CMoETrackMixTerrainCfg):
    """参考 `mix_obstacles_terrain`：窄走廊上的「台阶上行 → 深坑 → 高台 → 高栏 → 深坑 → 平台」。

    参考图案（索引段 → 高度索引，高度再乘 ``diff = 1.1*d``）：:

        0:30 → 0        30:36 → 30      36:42 → 60      42:48 → 90      48:60 → 120
        60:(72-k) → 深坑                  (72-k):84 → 120  84:86 → 0（参考里未被赋值，照抄）
        86:96 → 96      96:99 → 170      99:111 → 120    111:(123-k) → 深坑
        (123-k):140 → 120                140:160 → 60      160:以后 → 0

    其中 ``k = round(10 - 10*d)``；走廊外（|y| > 0.4 m）整体下沉。
    **已知偏离**：参考的起步平台缩比后只有 0.60 m，装不下我们 0.75 m 的出生点 ⇒ 图案整体平移
    ``pattern_start_x = 0.30``（起步平台变 0.90 m），其余形状逐段一致；坑深取固定 0.50 m
    （参考为 0.5–1.5 m 随机，缩比区间 0.02–0.60 m）。

    2026-10-04（用户："mix 中每个地形间隔大一点 ×1.5~2.0"）：新增 ``pattern_spacing_scale`` —— 把
    **相邻图案（＝下面 ``segments`` 里的每一段）之间的 X 推进量**乘上该乘子，**图案自身宽度、高度与
    顺序一律不变**：

        x0 = pattern_start_x + start_units · x_unit · scale
        x1 = x0 + (end_units − start_units) · x_unit            # 宽度与 scale 无关

    * ``scale = 1.0``（默认，＝训练用值）时增量**精确为 ``+0.0``** ⇒ 生成结果与加这个字段之前**逐位相同**；
    * ``scale > 1`` 时相邻图案之间空出来的 X 区间仍由**原有几何**填充 —— 本图案的走廊只在 ``segments``
      那几段上，段与段之间是整宽的 ``-pit_depth`` 坑底 ⇒ 被拉开的空间会表现为**更长的整宽深坑**（不是
      平地跑道）。"更从容"只是几何意义（障碍之间更远），不是"加了平地"，见 docs 的同名说明。

    2026-10-04（第四批，用户："mix 还是太小了…课程长度太短：让 mix 占满整条道"）：新增
    ``fill_stretched_gaps``。上一段的"更长的整宽深坑"就是用户看到的"太大/太深"的来源，本字段把它改掉：

    * ``False``（**默认**）＝ 上面那种行为，**与改动前逐位相同**（训练侧不传该字段）；
    * ``True`` ＝ 在 ``scale > 1`` 时，把**因拉开而多出来的空档**（段与段之间、最后一段末端 → 图案末端
      的"尾段"）铺成 ``height=0`` 的可走面 ``_corridor``；**领先段**（``0 → pattern_start_x``，其
      图案锚点恒为 0 ⇒ 不随 scale 变化）与原有尾廊本来就在 0 高度，不需额外处理。
      每个间隙里**原图案本来就有的坑**（相邻两段的原始索引差 ``next_start − prev_end > 0``，本图案是
      ``60 → (72−k)`` 与 ``111 → (123−k)`` 两处）**原样保留为坑**，且坑紧贴**上游**那一块的末端
      （⇒ 坑宽 ``= (next_start − prev_end) · x_unit`` 只由图案与难度决定，**与 scale 无关**，位置相对
      上游障碍也不变），余下的"拉开余量"才铺平。

    ⇒ ``fill_stretched_gaps=True`` 时**障碍自身几何（每块宽度/顶面高度/顺序）与坑宽一字不变**，
    被拉开的只是"障碍之间的平地"；整块瓦片的可走面总长 = ``size[0] − 坑总长``（尾廊会吸收拉伸量，
    因此**整块瓦片的可走面总长不随 scale 变**——随 scale 单调增的是**图案区间内**的可走面与补出的平地）。

    2026-10-04（第五批，用户："整体地形放大，原本是10m长就改成20m长，还是布满，但是障碍数量不变，
    设置不变，只把间隔改大"）：本函数**一行未改** —— 放大的是瓦片 ``cfg.size = (20, 4)``（评测场景
    显式覆盖 `terrain_generator.size`），乘子按同一套反算式重算为 **6.00**，于是障碍数量/尺寸/高度/
    坑深/走廊宽/难度/顺序全部照旧，只有间隔变大。``size[0]`` 参与的两处都跟着变：
    溢出上限 (20−0.30)/(160×0.02) = **6.15625**（保护不放宽），图案末端 = 0.30+160×0.02×6.00 = **19.50 m**。

    **总长保护（不静默溢出）**：图案末端 = ``pattern_start_x + 160 · x_unit · scale``。超过
    ``size[0]`` 时**直接 raise ``ValueError``**（并给出该瓦片上 scale 的上限）——参照问题表 CMOE-13：
    ``track_gap_terrain`` 缺这层保护，超长时会静默截断/与邻块重叠。该保护**不因 fill 模式而放宽**。
    """
    diff = cfg.height_scale * difficulty
    gap_shrink = round(cfg.gap_shrink_units * (1.0 - difficulty))
    offset = cfg.pattern_start_x
    x_unit, z_unit = cfg.x_unit, cfg.z_unit
    spacing = cfg.pattern_spacing_scale

    # 图案最后一个索引（见下面 segments 的末段 140:160）+ 溢出保护。
    pattern_end_units = 160.0
    pattern_x_end = offset + pattern_end_units * x_unit * spacing
    if pattern_x_end > cfg.size[0] + 1.0e-9:
        max_spacing = (cfg.size[0] - offset) / (pattern_end_units * x_unit)
        raise ValueError(
            f"track_mix_terrain: 图案总长 {pattern_x_end:.4f} m 超出瓦片长 size[0]={cfg.size[0]:.4f} m"
            f"（pattern_start_x={offset} + {pattern_end_units:g} 索引 × x_unit={x_unit}"
            f" × pattern_spacing_scale={spacing}）。本瓦片上 pattern_spacing_scale 最大可取 "
            f"{max_spacing:.4f}。三选一：① 减少图案数量（改本函数的 segments）；② 增大 "
            f"terrain_generator.size[0]；③ 把乘子降到 {max_spacing:.4f} 以下。"
        )

    def _shift(anchor_units: float) -> float:
        """图案起点按乘子后移的增量；``spacing == 1.0`` 时**精确等于 0.0**（保证逐位不变）。"""
        return (spacing - 1.0) * anchor_units * x_unit

    # 坑底：整宽、顶面在 -pit_depth
    meshes: list[trimesh.Trimesh] = [_platform(0.0, cfg.size[0], cfg.size[1], -cfg.pit_depth)]
    # 图案之前的起步走廊（到 pattern_start_x）
    meshes.append(_corridor(0.0, offset, cfg.size[1], cfg.corridor_width, 0.0))

    segments = (
        (0.0, 30.0, 0.0),
        (30.0, 36.0, 30.0),
        (36.0, 42.0, 60.0),
        (42.0, 48.0, 90.0),
        (48.0, 60.0, 120.0),
        (72.0 - gap_shrink, 84.0, 120.0),
        (84.0, 86.0, 0.0),
        (86.0, 96.0, 96.0),
        (96.0, 99.0, 170.0),
        (99.0, 111.0, 120.0),
        (123.0 - gap_shrink, 140.0, 120.0),
        (140.0, 160.0, 60.0),
    )
    for start_units, end_units, height_units in segments:
        if end_units <= start_units:
            continue
        # 只改"图案之间的 X 推进量"：整段按**自己起点**的增量平移（⇒ 段宽不变 ⇒ 障碍尺寸/高度/顺序都不变）。
        shift = _shift(start_units)
        x0 = offset + start_units * x_unit + shift
        x1 = min(offset + end_units * x_unit + shift, cfg.size[0])
        if x1 <= x0:
            continue
        meshes.append(
            _corridor(
                x0, x1, cfg.size[1], cfg.corridor_width, height_units * z_unit * diff
            )
        )

    # 图案结束（索引 160）之后走廊回到 0 高度，直铺到瓦片末端
    tail_x0 = min(offset + pattern_end_units * x_unit + _shift(pattern_end_units), cfg.size[0])
    if tail_x0 < cfg.size[0]:
        meshes.append(_corridor(tail_x0, cfg.size[0], cfg.size[1], cfg.corridor_width, 0.0))

    # 2026-10-04（第四批）：把"因拉开而多出来的空档"铺成 height=0 的可走面。
    # 只在**显式开启**且确实拉开了（spacing > 1）时执行 ⇒ 默认分支的网格与顺序一字不动。
    # 每个间隙 = [上游块末端(拉伸后), 下游块起点(拉伸后)]：其中前 ``next_start − prev_end`` 个索引是
    # **原图案本来就有的坑**（本图案为 60→(72−k)、111→(123−k) 两处），紧贴上游块末端原样保留；
    # 其余"拉开余量"铺平。最后再补"最后一段末端 → 图案末端(索引 160)"这段尾段空档。
    if cfg.fill_stretched_gaps and spacing > 1.0:
        placed = [(start, end) for start, end, _ in segments if end > start]
        slack: list[tuple[float, float]] = []
        for (prev_start, prev_end), (next_start, _next_end) in zip(placed, placed[1:]):
            prev_x1 = min(offset + prev_end * x_unit + _shift(prev_start), cfg.size[0])
            next_x0 = min(offset + next_start * x_unit + _shift(next_start), cfg.size[0])
            # 原图案里这一段本来就空着（＝坑）的宽度：只由图案与难度决定，与 scale 无关。
            pit_width = max(0.0, next_start - prev_end) * x_unit
            slack.append((prev_x1 + pit_width, next_x0))
        # 尾段：最后一段末端 → 图案末端（尾廊从图案末端才开始）
        last_start, last_end = placed[-1]
        last_x1 = min(offset + last_end * x_unit + _shift(last_start), cfg.size[0])
        slack.append((last_x1, tail_x0))
        for fill_x0, fill_x1 in slack:
            if fill_x1 - fill_x0 > 1.0e-12:
                meshes.append(_corridor(fill_x0, fill_x1, cfg.size[1], cfg.corridor_width, 0.0))

    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], 0.0])
    return meshes, origin


@configclass
class CMoETrackMixTerrainCfg(SubTerrainBaseCfg):
    """参考 `mix_obstacles_terrain` 的缩比版（窄走廊 + 台阶/深坑/高台/高栏混合）。"""

    function = track_mix_terrain
    x_unit: float = _MIX_X_UNIT
    z_unit: float = _MIX_Z_UNIT
    # 参考 diff = hurdle_height_range[0] * 1.1 = 1.1 * d
    height_scale: float = 1.1
    # 参考 round(10 - 10*d)（索引单位）
    gap_shrink_units: float = 10.0
    # 参考走廊半宽 20 索引 = 1.0 m ⇒ ×0.4 = 0.40 m（全宽 0.80 m）
    corridor_width: float = 0.80
    # 参考坑深 -100..-300 索引 = 0.5–1.5 m ⇒ ×0.4 = 0.02–0.60 m，取固定值
    pit_depth: float = 0.50
    # 见函数 docstring 的「已知偏离」：图案整体平移，保证 0.75 m 出生点落在起步平台上
    pattern_start_x: float = 0.30
    # 2026-10-04：相邻图案之间的 **X 推进量**乘子（只改间距，不改障碍自身尺寸/高度/顺序）。
    # ⚠️ 默认 **1.0 ⇒ 训练用几何逐位不变**（增量精确为 +0.0）；评测用的 mix-test 场景取 **6.00**
    # （＝"刚好占满整条 **20 m** 道"的反算值，算法见 `CMoE_env_cfg.py::MIX_TEST_PATTERN_SPACING_SCALE`；
    # 第五批把该场景的单块瓦片 X 由 8 m 放大到 20 m，乘子随之由 2.25 重算为 `(20−0.30−0.50)/(160×0.02)`）。
    # 上限受瓦片长度约束：pattern_start_x + 160·x_unit·scale ≤ size[0]（mix-test 现在 size[0]=20 m、
    # x_unit=0.02 ⇒ scale ≤ 6.15625；训练侧 size[0]=8 m ⇒ 2.40625）；超出会在 `track_mix_terrain` 里
    # 直接 raise（不静默溢出）。
    pattern_spacing_scale: float = 1.0
    # 2026-10-04（第四批）：把"因 pattern_spacing_scale 拉开而多出来的空档"（段间、尾段）铺成
    # height=0 的可走面（原有坑宽/障碍几何不变）。默认 **False ⇒ 与改动前逐位相同**；训练侧不传它。
    fill_stretched_gaps: bool = False
    spawn_x: float = 0.75


def track_narrow_stairs_terrain(difficulty: float, cfg: CMoETrackNarrowStairsTerrainCfg):
    """窄走廊楼梯（参考 `narrow_stairs_terrain` 的缩比版）：上行 → 平台 → 下行，两侧是深坑。

    参考调用（`humanoid_terrain.py:241`）::

        narrow_stairs_terrain(num_stones=24, step_height=0.25*d, x_range=[0.30, 1.5],
                              half_valid_width=[1 - 0.5*d, 1.5 - 0.5*d])

    参考里步深固定取 ``x_range[0] = 0.30 m``、走廊半宽固定取 ``half_valid_width[0] = 1 - 0.5*d`` m，
    前 10 级上行、第 10–14 级保持、其后 9 级下行（净升高 ≈ 一个步高）。×0.4 后：步深 0.12 m、
    步高 0.10*d（d=1 → 0.09 m）、走廊全宽 0.80 - 0.40*d（0.44–0.80 m）、起步平台 1.00 m。
    **已知偏离**：参考的坑深是 0.05–1.5 m 随机（缩比 0.02–0.60 m），这里取固定 0.50 m。
    """
    step_height = cfg.step_height_max * difficulty
    corridor_width = 2.0 * (cfg.corridor_half_width_start - cfg.corridor_half_width_slope * difficulty)

    meshes: list[trimesh.Trimesh] = [_platform(0.0, cfg.size[0], cfg.size[1], -cfg.pit_depth)]
    # 参考的起步平台是**整宽**的（[0:platform_len, :] = 0）
    meshes.append(_platform(0.0, cfg.platform_length, cfg.size[1]))

    x = cfg.platform_length
    height = 0.0
    for index in range(cfg.num_steps):
        if index < cfg.num_steps // 2 - 2:
            height += step_height
        elif index > cfg.num_steps // 2 + 2:
            height -= step_height
        x1 = x + cfg.step_depth
        if x1 > cfg.size[0]:
            break
        meshes.append(_corridor(x, x1, cfg.size[1], corridor_width, height))
        x = x1
    # 参考在楼梯之后没有被赋值的区域 ⇒ 回到 0 高度的整宽平地
    if x < cfg.size[0]:
        meshes.append(_platform(x, cfg.size[0], cfg.size[1]))

    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], 0.0])
    return meshes, origin


@configclass
class CMoETrackNarrowStairsTerrainCfg(SubTerrainBaseCfg):
    """窄走廊楼梯。米制常量＝参考 `narrow_stairs_terrain` × ``REFERENCE_SCALE``。"""

    function = track_narrow_stairs_terrain
    num_steps: int = 24
    # 参考步深 = x_range[0] = 0.30 m ⇒ ×0.4
    step_depth: float = 0.12
    # 参考 step_height = 0.25*d（m）⇒ ×0.4 = 0.10*d
    step_height_max: float = 0.10
    # 参考 platform_len = 2.5 m ⇒ ×0.4 = 1.00 m（覆盖 0.75 m 出生点）
    platform_length: float = 1.00
    # 参考半宽 half_valid_width[0] = 1 - 0.5*d（m）⇒ ×0.4 = 0.40 - 0.20*d
    corridor_half_width_start: float = 0.40
    corridor_half_width_slope: float = 0.20
    # 参考坑深随机 0.05–1.5 m ⇒ 缩比 0.02–0.60 m，取固定值
    pit_depth: float = 0.50
    spawn_x: float = 0.75
