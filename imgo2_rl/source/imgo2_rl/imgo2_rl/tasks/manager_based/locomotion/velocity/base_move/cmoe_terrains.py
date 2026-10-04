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
    ⚠️ 本段描述的"粒度＝每一段"只在 ``fill_stretched_gaps=False``（默认）时成立；``True`` 时**粒度已由
    第六批改成"每个连续段组"**（见下面的第六批段落）—— 逐段铺平地会把 4 级楼梯拆成 4 个孤立小凸块，
    这正是用户看到的"台阶似乎只有一级"；且那时的"坑紧贴上游块末端"会让"高处下来立刻遇到坑"
    （用户追加反馈："一个较高的台子后紧接 gap"）⇒ 第六批把补出的平地改铺在**坑的上游**。

    2026-10-04（第五批，用户："整体地形放大，原本是10m长就改成20m长，还是布满，但是障碍数量不变，
    设置不变，只把间隔改大"）：本函数**一行未改** —— 放大的是瓦片 ``cfg.size = (20, 4)``（评测场景
    显式覆盖 `terrain_generator.size`），乘子按同一套反算式重算为 **6.00**，于是障碍数量/尺寸/高度/
    坑深/走廊宽/难度/顺序全部照旧，只有间隔变大。``size[0]`` 参与的两处都跟着变：
    溢出上限 (20−0.30)/(160×0.02) = **6.15625**（保护不放宽），图案末端 = 0.30+160×0.02×6.00 = **19.50 m**。

    2026-10-04（第六批，用户："台阶似乎只有一级"—— 实测 ``scale=6.00`` 时 4 级楼梯被拆成
    ``[3.9,4.02] 0.0462 / [4.62,4.74] 0.0924 / [5.34,5.46] 0.1386 / [6.06,6.30] 0.1848`` 四个孤立小凸块，
    中间各夹 0.60 m 平地）：**第四批的"逐段"粒度就是根因** —— ``segments`` 里那 4 级楼梯
    （``30:36 / 36:42 / 42:48 / 48:60``，每级仅 0.12 m 宽）在原始 units 上**首尾相接**，而逐段公式给
    每段**各按自己的起点**加 ``(scale−1)·start·x_unit`` ⇒ 相邻两级之间凭空多出
    ``(scale−1)·6·x_unit = 0.60 m`` 的空档，被补平地规则铺成平地 ⇒ 楼梯被拆散。本批把拉伸/补平地的
    粒度改成**每个障碍组**（run of contiguous segments）：

    * **分组**：把非空 ``segments`` 按**原始 units 上是否首尾相接**（``next.start == prev.end``）切成
      若干组。本图案在 ``d = 0.70`` 时是 **3 组**：``0→60``（起步平台 ＋ 4 级楼梯）、``69→111``、
      ``120→160``；空段（``end ≤ start``，``gap_shrink`` 很大时才会出现）已被过滤、不参与分组。
      ⚠️ ``0:30`` 与 ``30:36`` 在 units 上同样首尾相接（``30 == 30``）⇒ 按定义属**同一组**，
      起步平台与楼梯焊在一起（这正是**训练几何**：出生点 0.75 → 第一级台阶 0.90）；
      若把它俩当成两组，它们之间的原始空档宽度是 **0** ⇒ 也拿不到任何额外长度、几何完全相同。
    * **组内**：整组共用**一个**平移量 ⇒ 段与段仍然首尾相接、段宽与顶面不变 ⇒ **4 级楼梯连成楼梯**
      （真 trimesh 实测：``[0.90,1.02] 0.0462 / [1.02,1.14] 0.0924 / [1.14,1.26] 0.1386 /
      [1.26,1.50] 0.1848``，相邻级 X 严丝合缝、级间**没有平地**）。
    * **组间**：可拉伸的空档＝**原始 units 上本来就有空档的组间边界**（本图案两处坑所在位置）
      ＋图案末尾的**尾段**；每个空档里：
      1. **原有坑宽原样保留** = ``(next.start − prev.end) · x_unit``（``d = 0.70`` 时两处各
         **0.18 m**，总 0.36 m）；
      2. 分到的额外长度一律铺 ``height = 0`` 的 **可走面**，且铺在**坑的上游** —— 即
         ``[上游组末端, 坑左沿]``，坑**紧贴下游组起点**一侧（2026-10-04 追加：用户实测
         "一个较高的台子后紧接 gap" ⇒ 让"从高处下来"先有一段平地助跑、再遇坑）。
         ``d = 0.70, scale = 6.00`` 时每处坑前平地 **5.3333 m**、坑后落点平地 **0 m**
         （坑后直接就是下游组的抬高块，与 ``scale = 1.0`` 的原始图案一致）；
      3. 因乘子多出来的总长度 ``160 · x_unit · (scale − 1)`` 在这些空档之间**均匀分配**
         （``slot = 两处坑所在空档 + 尾段 = 3`` ⇒ ``scale = 6.00`` 时每处 **5.3333 m**）；
         理由：每处都拿到一段等长平地，既不会让某一处独吞 16 m、也不改变"障碍 → 紧邻坑"的相对位置。
         （尾段分到的那一份仍是"最后一段末端 → 图案末端"的平地。）
    * **`fill_stretched_gaps=False` / `scale = 1.0` ⇒ 旧路径逐位不变**：前者（默认，训练侧就是这条）
      仍是第四批的"逐段"公式（多出来的空间由整宽坑底填充），后者 ``shift`` 精确为 ``+0.0``；
      分组布局只在 ``fill_stretched_gaps=True`` **且** ``scale > 1`` 时启用。
    * **溢出保护不放宽**：仍按 ``pattern_start_x + 160 · x_unit · scale ≤ size[0]`` 判、超限
      ``ValueError``；分组不改变图案末端（仍是 ``offset + 160 · x_unit · scale``）。

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
    # 图案结束（索引 160）之后走廊回到 0 高度，从图案末端直铺到瓦片末端
    tail_x0 = min(pattern_x_end, cfg.size[0])

    def _shift(anchor_units: float) -> float:
        """图案起点按乘子后移的增量；``spacing == 1.0`` 时**精确等于 0.0**（保证逐位不变）。"""
        return (spacing - 1.0) * anchor_units * x_unit

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
    placed = [(start, end, height) for start, end, height in segments if end > start]

    # 2026-10-04（第六批）：**分组** —— 原始 units 上首尾相接（``next.start == prev.end``）的连续段
    # 属于同一组。本图案在 ``d = 0.70`` 时是 3 组（打印/审计见 `check_terrain_columns.py --task
    # mix-test` 与 `tests/test_cmoe_mix_test_scene.py`）：
    #   ① 0 → 60   起步平台 0:30 ＋ **4 级楼梯** 30:36 / 36:42 / 42:48 / 48:60（首尾相接）
    #   ② 69 → 111 跨第一处坑后的高台/平台/高栏/平台
    #   ③ 120 → 160 跨第二处坑后的平台/下行台阶
    # ⚠️ `0:30` 与 `30:36` 也是首尾相接（30 == 30）⇒ 按定义同组（起步平台与楼梯焊在一起，
    #    与训练几何一致）；把它们拆成两组也不会改变几何：二者之间的原始空档恒为 0。
    groups: list[list[tuple[float, float, float]]] = []
    for segment in placed:
        if groups and abs(segment[0] - groups[-1][-1][1]) <= 1.0e-9:
            groups[-1].append(segment)
        else:
            groups.append([segment])

    # 每个段的平移量：默认路径＝第四批的"逐段"公式（逐位不变）；
    # 分组路径＝该段所在组的**累计**平移量（组内共用 ⇒ 组内段与段仍然首尾相接）。
    segment_shifts: dict[tuple[float, float], float] = {}
    # 组间空档里要补的 `height=0` 可走面 `[(x0, x1), ...]`（坑本身不补、原样留空）。
    stretched_fills: list[tuple[float, float]] = []
    if cfg.fill_stretched_gaps and spacing > 1.0:
        # 可拉伸的空档 = **原始 units 上本来就有空档的组间边界**（本图案两处坑所在位置）＋ 尾段。
        # ① 坑宽保持原样：每处坑仍占 `(next.start − prev.end)·x_unit`（d=0.70 各 0.18 m）且**紧贴
        #    下游组起点**（⇒ 补出的平地正好铺在它**上游**）；② 多出来的总长度
        #    `160·x_unit·(scale−1)` 在这些空档之间**均匀分配**（理由见函数 docstring 与 docs：
        #    每处都拿到一段等长平地，不让某一处独吞）。
        stretch_gaps = [
            (index, group[0][0] - groups[index - 1][-1][1])
            for index, group in enumerate(groups)
            if index and group[0][0] - groups[index - 1][-1][1] > 1.0e-9
        ]
        slot_count = len(stretch_gaps) + 1  # ＋尾段（最后一段末端 → 图案末端，索引 160）
        extra_each = (spacing - 1.0) * pattern_end_units * x_unit / slot_count
        shift = 0.0
        previous_end_units = groups[0][-1][1]
        for index, group in enumerate(groups):
            if index and group[0][0] - previous_end_units > 1.0e-9:
                # 跨过一个"原始就有空档"的边界：把该空档分到的额外长度加到组平移量上
                # （坑宽只由图案与难度决定 ⇒ 不受影响；坑相对本组的位置由下面的补块计算固定）。
                shift += extra_each
            for start_units, end_units, _height_units in group:
                segment_shifts[(start_units, end_units)] = shift
            previous_end_units = group[-1][1]
        # 补块：**坑的上游**（上游组末端 → 坑左沿）→ 下一组起点；最后一段末端 → 图案末端（尾段）。
        # 2026-10-04（第六批追加，用户："我似乎有看到一个较高的台子后紧接 gap"）：补出的平地放在
        # 坑的**上游**（＝坑紧贴**下游组起点**一侧）⇒ 从高处下来先有一段平地助跑/落点，再遇到坑；
        # 坑宽不变（`(next.start − prev.end)·x_unit`），只是把坑从"紧贴上游组末端"改成"紧贴下游组"。
        for index, gap_units in stretch_gaps:
            previous_start, previous_end, _previous_height = groups[index - 1][-1]
            pit_upstream_x = (
                offset
                + previous_end * x_unit
                + segment_shifts[(previous_start, previous_end)]
            )
            first_start, first_end, _first_height = groups[index][0]
            next_x0 = offset + first_start * x_unit + segment_shifts[(first_start, first_end)]
            pushed_pit_x0 = next_x0 - gap_units * x_unit
            stretched_fills.append((pit_upstream_x, pushed_pit_x0))
        last_start, last_end, _last_height = groups[-1][-1]
        last_x1 = min(offset + last_end * x_unit + segment_shifts[(last_start, last_end)], cfg.size[0])
        stretched_fills.append((last_x1, tail_x0))
    else:
        for start_units, end_units, _height_units in placed:
            segment_shifts[(start_units, end_units)] = _shift(start_units)

    # 坑底：整宽、顶面在 -pit_depth
    meshes: list[trimesh.Trimesh] = [_platform(0.0, cfg.size[0], cfg.size[1], -cfg.pit_depth)]
    # 图案之前的起步走廊（到 pattern_start_x）
    meshes.append(_corridor(0.0, offset, cfg.size[1], cfg.corridor_width, 0.0))

    for start_units, end_units, height_units in segments:
        if end_units <= start_units:
            continue
        # 只改"图案之间的 X 推进量"：整段按**自己所在组**的增量平移
        # （⇒ 组内段与段仍首尾相接、段宽不变 ⇒ 4 级楼梯连成楼梯；默认路径＝逐段公式，逐位不变）。
        shift = segment_shifts[(start_units, end_units)]
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
    if tail_x0 < cfg.size[0]:
        meshes.append(_corridor(tail_x0, cfg.size[0], cfg.size[1], cfg.corridor_width, 0.0))

    # 2026-10-04（第四/六批）：把"因拉开而多出来的空档"铺成 `height=0` 可走面（**原有坑原样留空**）。
    # 默认与 `fill_stretched_gaps=False` 时这个列表为空 ⇒ 网格列表与顺序一字不动（逐位不变）。
    for fill_x0, fill_x1 in stretched_fills:
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
    # ⚠️ 默认 **1.0 ⇒ 训练用几何逐位不变**（增量精确为 +0.0）—— 训练侧就是这个默认值。
    # 2026-10-05（第七批）：**评测场景已不再使用 mix 图案**（换成 `track_composite_terrain`）⇒ 该字段
    # 现在只服务训练侧与既有回归测试；历史上的评测取值（第四批 2.25 → 第五批 6.00，＝"刚好占满整条
    # 20 m 道"的反算值 `(20−0.30−0.50)/(160×0.02)`）记录在 docs/cmoe_mix_test_scene_2026-10-04.md。
    # 上限受瓦片长度约束：pattern_start_x + 160·x_unit·scale ≤ size[0]（20 m / x_unit=0.02 ⇒ 6.15625；
    # 训练侧 size[0]=8 m ⇒ 2.40625）；超出会在 `track_mix_terrain` 里直接 raise（不静默溢出）。
    pattern_spacing_scale: float = 1.0
    # 2026-10-04（第四批）：把"因 pattern_spacing_scale 拉开而多出来的空档"（段间、尾段）铺成
    # height=0 的可走面（原有坑宽/障碍几何不变）。默认 **False ⇒ 与改动前逐位相同**；训练侧不传它。
    # 2026-10-04（第六批）：为 True 时**粒度＝每个连续段组**（组内保持首尾相接 ⇒ 4 级楼梯连成楼梯，
    # 额外长度均匀分到"原有两处坑所在空档 ＋ 尾段"、坑宽恒不变，见 `track_mix_terrain` 的第六批段落）。
    # False（默认）时仍是第四批的"逐段"公式（多出来的空间由整宽 -pit_depth 坑底填充）⇒ 逐位不变。
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


# ======================================================================================
# 2026-10-05：**复合道**（composite track）—— 评测任务 `Imgo2-basemove-rough-cmoe-mix-test` 专用。
#
# 用户反馈（本轮唯一目标，原话）：「是有些障碍分到一起了，我说的是障碍本身。gap 和突台也在一起。
# 我希望是和单独的那个几个场景类似的障碍放在这个整条的（道）上」；补充「每个地形要间隔开，而不是
# 混合到一起放在一个位置」；再补充（顺序与楼梯形态，已确认）：
#
#     出生平地 → 【坑】 → 平地间隔 → 【楼梯】 → 平地间隔 → 【箱子/台阶块】 → 平地间隔 → 【栏】
#     → 平地间隔 → 尾平地
#
# 楼梯形态：**上下楼梯紧贴**（先上 4 级、紧接再下 4 级、中间**没有平地**）＝经典金字塔形。
#
# ⇒ **放弃**参考实现的 `track_mix_terrain` 小图案（坑紧挨抬高块、4 cm 窄槽、0.26 m 长的尖峰、
#   间距只够一个脚掌）—— 那些是**参考图案专有**的结构，在任何一类"独立地形"里都不存在。
#   本函数把**各类独立地形**里的障碍**各自成形**、**按类型分组顺序**铺在这条 20 m 道上，
#   障碍单元之间统一留等长平地。**训练侧仍用 `track_mix_terrain`**（两函数互不影响；`mix` 一侧
#   的默认值与全部既有测试逐位不变）。
# ======================================================================================
# 障碍顺序（模块级常量，便于审计与测试）：坑 → 楼梯 → 箱子/台阶块 → 栏 → 坑（用户："首尾各一"）。
COMPOSITE_OBSTACLE_SEQUENCE: tuple[str, ...] = ("gap", "stairs", "boxes", "hurdle", "gap")
# 相邻障碍之间的**平地间隔下界**（m）。用户硬约束："任意两个相邻障碍之间都必须有 ≥ obstacle_spacing
# 的平地"（不只是"坑不紧贴抬高块"）⇒ 每种障碍在视觉上各自独立、彼此分开。运行期**不放宽**：
# 反算/显式给出的间隔小于它时 `track_composite_terrain` 直接 `raise ValueError`。
COMPOSITE_MIN_OBSTACLE_SPACING: float = 1.50
# 用户给出的**建议区间**（只用于报告"实测间隔是否落在建议区间内"，不参与判定）。
COMPOSITE_SUGGESTED_OBSTACLE_SPACING: tuple[float, float] = (1.50, 2.50)
# `obstacle_spacing` 字段的"自动反算"哨兵：**负值** ⇒ 忽略字段值、按"刚好占满整条道"反算；
# **非负值**（含 0）⇒ 当作用户显式给的间隔，必须 ≥ `min_obstacle_spacing`，否则直接 `raise`
# （⇒ "把 `obstacle_spacing` 设成 0（＝两个障碍贴在一起）"是一个**会报错**的构造，见测试的负向对照）。
COMPOSITE_SPACING_AUTO: float = -1.0



def _composite_obstacle_units(
    difficulty: float, cfg: "CMoETrackCompositeTerrainCfg"
) -> list[dict]:
    """按**各类独立地形自己的难度律**算出复合道上每个障碍单元的几何（与摆放位置无关）。

    尺寸一律**复用既有常量与公式**（不新造）：① 坑 ＝ `track_gap_terrain` 的沟宽律；② 楼梯 ＝
    `track_stairs_terrain` 的步高律 ＋ `step_depth`（上 `num_steps` 级 ＋ 下 `num_steps` 级，
    **首尾相接**）；③ 箱子/台阶块 ＝ `track_step_terrain` 的高/长律 ＋ `step_spacing`；④ 栏 ＝
    `track_hurdle_terrain` 的高度律 `[min_slope·d, max_base + max_slope·d]` 与厚度律。

    返回列表每项：``{"kind", "label", "width", "params", "raised": [(dx0, dx1, top), ...]}``，
    其中 `raised` 是**相对单元起点**的抬高块（`_corridor` 用）；坑的 `raised` 为空
    ⇒ 该 X 区间**不铺走廊**，露出 `-pit_depth` 的整宽基座（＝坑底）。
    """
    units: list[dict] = []
    for kind in COMPOSITE_OBSTACLE_SEQUENCE:
        if kind == "gap":
            width = cfg.gap_width_range[0] + difficulty * (
                cfg.gap_width_range[1] - cfg.gap_width_range[0]
            )
            units.append(
                {
                    "kind": kind,
                    "label": "坑（横向沟）",
                    "width": float(width),
                    "params": {"gap_width": float(width)},
                    "raised": [],
                }
            )
        elif kind == "stairs":
            step_height = cfg.stairs_step_height_range[0] + difficulty * (
                cfg.stairs_step_height_range[1] - cfg.stairs_step_height_range[0]
            )
            step_depth = float(cfg.stairs_step_depth)
            count = int(cfg.stairs_num_steps)
            # 上 n 级：顶面 step_height … n·step_height；紧接**下 n 级**：顶面 (n−1)·step_height … 0
            # （逐字复用 `track_stairs_terrain` 的 ascending / ascending=False 两段，中间不加平地）。
            raised = [
                (index * step_depth, (index + 1) * step_depth, (index + 1) * step_height)
                for index in range(count)
            ]
            raised += [
                (
                    count * step_depth + index * step_depth,
                    count * step_depth + (index + 1) * step_depth,
                    (count - index - 1) * step_height,
                )
                for index in range(count)
            ]
            units.append(
                {
                    "kind": kind,
                    "label": f"楼梯（上/下各 {count} 级，紧贴）",
                    "width": 2.0 * count * step_depth,
                    "params": {
                        "num_steps": count,
                        "step_height": float(step_height),
                        "step_depth": step_depth,
                        "peak_height": float(count * step_height),
                        "step_count_total": 2 * count,
                        # 8 级（上 4 + 下 4）的顶面序列（审计/测试直接用，不必再重建）
                        "tread_tops": tuple(float(top) for _dx0, _dx1, top in raised),
                    },
                    "raised": raised,
                }
            )
        elif kind == "boxes":
            box_height = cfg.box_height_range[0] + difficulty * (
                cfg.box_height_range[1] - cfg.box_height_range[0]
            )
            box_length = cfg.box_length_range[0] + difficulty * (
                cfg.box_length_range[1] - cfg.box_length_range[0]
            )
            count = int(cfg.box_count)
            spacing = float(cfg.box_spacing)
            # 与 `track_step_terrain` 相同：块之间是 `step_spacing` 米平地（本单元**内部**的间距）。
            raised = [
                (index * (box_length + spacing),
                 index * (box_length + spacing) + box_length,
                 box_height)
                for index in range(count)
            ]
            units.append(
                {
                    "kind": kind,
                    "label": "箱子/台阶块",
                    "width": count * box_length + (count - 1) * spacing,
                    "params": {
                        "count": count,
                        "block_height": float(box_height),
                        "block_length": float(box_length),
                        "block_spacing": spacing,
                    },
                    "raised": raised,
                }
            )
        elif kind == "hurdle":
            hurdle_length = cfg.hurdle_len_range[0] + difficulty * (
                cfg.hurdle_len_range[1] - cfg.hurdle_len_range[0]
            )
            height_min = cfg.hurdle_height_min_slope * difficulty
            height_max = cfg.hurdle_height_max_base + cfg.hurdle_height_max_slope * difficulty
            # 独立地形对每道栏在 [height_min, height_max] 内**随机**取值；复合道要确定性 ⇒
            # 取该区间的 `hurdle_height_fraction` 分位（默认 0.5 ＝ 中点）。
            hurdle_height = height_min + cfg.hurdle_height_fraction * (height_max - height_min)
            # 独立地形的相邻栏间距在 `spacing_range` 内**随机**；同样取 `hurdle_spacing_fraction`
            # 分位（默认 0.5 ＝ 中点）。
            gap = cfg.hurdle_spacing_range[0] + cfg.hurdle_spacing_fraction * (
                cfg.hurdle_spacing_range[1] - cfg.hurdle_spacing_range[0]
            )
            count = int(cfg.hurdle_count)
            raised = [
                (index * (hurdle_length + gap),
                 index * (hurdle_length + gap) + hurdle_length,
                 float(hurdle_height))
                for index in range(count)
            ]
            units.append(
                {
                    "kind": kind,
                    "label": "栏（薄横栏）",
                    "width": count * hurdle_length + (count - 1) * gap,
                    "params": {
                        "count": count,
                        "bar_length": float(hurdle_length),
                        "bar_height": float(hurdle_height),
                        "bar_spacing": float(gap),
                        "height_range": (float(height_min), float(height_max)),
                    },
                    "raised": raised,
                }
            )
        else:  # pragma: no cover - 常量表里只有上面四种
            raise ValueError(f"track_composite_terrain: 未知障碍种类 {kind!r}")
    return units


def composite_track_layout(
    difficulty: float, cfg: "CMoETrackCompositeTerrainCfg"
) -> dict:
    """复合道的**完整布局**（纯算术，不生成网格）：审计脚本、测试与 `track_composite_terrain` 共用。

    摆放规则（用户确认的顺序与"间隔开"硬约束）::

        出生平地(0 → first_obstacle_x) → 障碍#1 → 平地(spacing) → 障碍#2 → … → 障碍#N → 平地(尾段)

    * `first_obstacle_x = spawn_x + spawn_clearance`（默认 `0.75 + 1.50 = 2.25 m`）
      ⇒ 出生点是实心平地，且**前方 ≥ 1.5 m 平地**才遇到第一个障碍；
    * **所有**相邻障碍单元之间（含尾段）都是**同一个** `obstacle_spacing`
      ⇒ "任意两个相邻障碍之间都有等长平地"这条硬约束由构造保证；
    * `obstacle_spacing` **反算**（`cfg.obstacle_spacing` 为负）::

          spacing = (size[0] − first_obstacle_x − Σ 障碍宽度) / N       # N ＝ 障碍个数
                  = (20 − 2.25 − ΣW) / 5

      即"把余量按障碍个数均分"（每个障碍之后各分到一段，最后一段就是尾段）⇒ 整条道**刚好占满**；
    * `cfg.obstacle_spacing ≥ 0` 时按显式值摆，尾段吸收余量；两者都必须 ≥ `min_obstacle_spacing`，
      且总长不得超过 `size[0]`，否则**直接 `raise ValueError`**（溢出保护不放宽）。

    返回字典含：`obstacles`（含 `x0/x1/width/params/raised`（绝对 X））、`flats`（平地区间）、
    `adjacent_gaps`（**逐对相邻障碍**之间的平地表）、`obstacle_spacing`、`end_x` 等。
    """
    units = _composite_obstacle_units(difficulty, cfg)
    size_x, size_y = float(cfg.size[0]), float(cfg.size[1])
    spawn_x = float(cfg.spawn_x)
    clearance = float(cfg.spawn_clearance)
    first_x = spawn_x + clearance
    total_width = float(sum(unit["width"] for unit in units))
    count = len(units)
    min_spacing = float(cfg.min_obstacle_spacing)

    if first_x + total_width + count * min_spacing > size_x + 1.0e-9:
        raise ValueError(
            f"track_composite_terrain: 放不下 —— 出生平地区间 {first_x:.4f} m ＋ 障碍总宽 "
            f"{total_width:.4f} m ＋ {count} 段间隔（每段 ≥ {min_spacing:.4f} m）"
            f" = {first_x + total_width + count * min_spacing:.4f} m > size[0]={size_x:.4f} m。"
            f"三选一：① 减障碍（改 COMPOSITE_OBSTACLE_SEQUENCE）；② 增大 "
            f"terrain_generator.size[0]（现在 {size_x:.4f}）；③ 降 "
            f"min_obstacle_spacing（现在 {min_spacing:.4f}）。"
        )

    explicit = float(cfg.obstacle_spacing)
    if explicit >= 0.0:
        spacing = explicit
        auto = False
        if spacing < min_spacing - 1.0e-9:
            raise ValueError(
                f"track_composite_terrain: 显式 obstacle_spacing={spacing:.4f} m < 下界 "
                f"min_obstacle_spacing={min_spacing:.4f} m ⇒ 两个相邻障碍会贴在一起（用户硬约束："
                f"任意相邻障碍之间必须有 ≥ {min_spacing:.4f} m 的平地）。"
            )
        if first_x + total_width + count * spacing > size_x + 1.0e-9:
            raise ValueError(
                f"track_composite_terrain: 总长 {first_x + total_width + count * spacing:.4f} m 超出 "
                f"瓦片长 size[0]={size_x:.4f} m（obstacle_spacing={spacing:.4f}）。三选一："
                f"① 减少障碍；② 增大 terrain_generator.size[0]；③ 把 obstacle_spacing 降到 "
                f"{(size_x - first_x - total_width) / count:.4f} 以下。"
            )
    else:
        spacing = (size_x - first_x - total_width) / count
        auto = True
        if spacing < min_spacing - 1.0e-9:
            raise ValueError(
                f"track_composite_terrain: 反算出的间隔 {spacing:.4f} m < 下界 "
                f"min_obstacle_spacing={min_spacing:.4f} m ⇒ 障碍会被挤在一起（{count} 个障碍总宽 "
                f"{total_width:.4f} m 装进 {size_x:.4f} m 的道）。三选一：① 减障碍（改 "
                f"COMPOSITE_OBSTACLE_SEQUENCE）；② 增大 terrain_generator.size[0]；③ 降 "
                f"min_obstacle_spacing（现在 {min_spacing:.4f}）。"
            )

    obstacles: list[dict] = []
    flats: list[dict] = [{"kind": "lead_in", "x0": 0.0, "x1": first_x, "length": first_x}]
    cursor = first_x
    for index, unit in enumerate(units):
        raised = [
            {
                "x0": cursor + float(dx0),
                "x1": cursor + float(dx1),
                "top": float(top),
            }
            for dx0, dx1, top in unit["raised"]
        ]
        obstacles.append(
            {
                "index": index,
                "kind": unit["kind"],
                "label": unit["label"],
                "x0": cursor,
                "x1": cursor + unit["width"],
                "width": unit["width"],
                "params": unit["params"],
                "raised": raised,
            }
        )
        cursor += unit["width"]
        if index < count - 1:
            length = spacing
        else:
            length = size_x - cursor  # 尾段（自动反算时恰好等于 spacing，浮点上容差 1e-15）
        if length < min_spacing - 1.0e-9:
            raise ValueError(
                f"track_composite_terrain: 第 {index + 1} 段平地只有 {length:.4f} m < 下界 "
                f"{min_spacing:.4f} m（尾段被显式间隔挤没了）。三选一：① 减少障碍；"
                f"② 增大 terrain_generator.size[0]；③ 降 min_obstacle_spacing。"
            )
        flats.append(
            {
                "kind": "tail" if index == count - 1 else "between",
                "after_index": index,
                "x0": cursor,
                "x1": cursor + length,
                "length": length,
            }
        )
        cursor += length

    adjacent_gaps = [
        {
            "pair": (obstacles[index]["kind"], obstacles[index + 1]["kind"]),
            "labels": (obstacles[index]["label"], obstacles[index + 1]["label"]),
            "x0": flats[index + 1]["x0"],
            "x1": flats[index + 1]["x1"],
            "length": flats[index + 1]["length"],
            "ok": flats[index + 1]["length"] >= min_spacing - 1.0e-9,
        }
        for index in range(count - 1)
    ]

    return {
        "difficulty": float(difficulty),
        "size": (size_x, size_y),
        "corridor_width": float(cfg.corridor_width),
        "pit_depth": float(cfg.pit_depth),
        "spawn_x": spawn_x,
        "spawn_clearance": clearance,
        "first_obstacle_x": first_x,
        # 键名与 `check_terrain_columns.composite_layout()`（**只用标准库**的算术复算）保持一致 ⇒
        # 审计脚本、测试与生成器读的是同一套字段，避免两套名字漂移。
        "spacing": float(spacing),
        "obstacle_spacing": float(spacing),
        "spacing_is_auto": auto,
        "min_obstacle_spacing": min_spacing,
        "suggested_spacing_range": tuple(cfg.suggested_spacing_range),
        "sequence": COMPOSITE_OBSTACLE_SEQUENCE,
        "obstacle_sequence": COMPOSITE_OBSTACLE_SEQUENCE,
        "total_width": total_width,
        "total_obstacle_width": total_width,
        "obstacles": obstacles,
        "flats": flats,
        "adjacent_gaps": adjacent_gaps,
        "last_obstacle_end": obstacles[-1]["x1"],
        "end_x": size_x,
        "tail_length": flats[-1]["length"],
    }


def composite_adjacency_violations(layout: dict, min_spacing: float | None = None) -> list[dict]:
    """**逐对相邻障碍**的间隔违规清单（空列表 ＝ 全部合格），供审计脚本与负向对照使用。

    `min_spacing` 缺省取 `layout["min_obstacle_spacing"]`。返回的每一项含 `pair`/`x0`/`x1`/`length`/
    `min_spacing`，便于直接打印。**不抛异常**（生成期已经抛过；这里给"人造布局"的检查用）。
    """
    bound = float(layout["min_obstacle_spacing"] if min_spacing is None else min_spacing)
    return [
        dict(gap, min_spacing=bound)
        for gap in layout["adjacent_gaps"]
        if gap["length"] < bound - 1.0e-9
    ]


def track_composite_terrain(difficulty: float, cfg: "CMoETrackCompositeTerrainCfg"):
    """生成**复合道**：一段 0.80 m 宽、20 m 长的走廊上按类型顺序各自成形的独立障碍。

    几何（`_corridor` ＝走廊宽 × 顶面高度；基座 ＝整宽、顶面 `-pit_depth`）::

        [整宽基座]                     顶面 -pit_depth（走廊外与坑底都是它）
        ├ 出生平地 [0, first_x]        顶面 0
        ├ 坑                          **不铺走廊** ⇒ 露出 -pit_depth 基座（＝坑底）
        ├ 平地 [spacing]
        ├ 楼梯（上 n 级 ＋ 下 n 级紧贴）  逐级顶面 (1..n)·h、(n−1..0)·h
        ├ 平地 [spacing]
        ├ 箱子/台阶块                  单元内平地 ＋ 2 块抬高块
        ├ 平地 [spacing]
        ├ 栏                          单元内平地 ＋ 2 道薄横栏（**栏周围是 z=0 平地**）
        ├ 平地 [spacing]
        ├ 坑                          **不铺走廊**
        └ 尾平地 [spacing] → size[0]

    * **每个障碍都用它自己那类地形的难度律与几何**（`_composite_obstacle_units`），本函数不新造尺寸；
    * **所有相邻障碍之间都是同一个 `obstacle_spacing`**（反算或显式，≥ `min_obstacle_spacing`）
      ⇒ 不会出现"坑紧挨抬高块"或任何两个障碍贴在一起；
    * 长度不够时**直接 `ValueError`**（三选一：减障碍／加长道／降间隔），**不静默截断**；
    * 通道宽度沿用评测场景既有的 `corridor_width = 0.80 m`、坑底深度沿用 `pit_depth = 0.50 m`
      （与 `track_mix_terrain` 同一套"走廊 ＋ 整宽基座"约定）。
    """
    layout = composite_track_layout(difficulty, cfg)
    size_x, size_y = layout["size"]
    corridor = layout["corridor_width"]

    meshes: list[trimesh.Trimesh] = [_platform(0.0, size_x, size_y, -layout["pit_depth"])]
    # 平地（出生平地 ＋ 相邻障碍之间的平地 ＋ 尾段）：一律顶面 0
    for flat in layout["flats"]:
        if flat["length"] > 1.0e-12:
            meshes.append(_corridor(flat["x0"], flat["x1"], size_y, corridor, 0.0))
    for obstacle in layout["obstacles"]:
        if obstacle["kind"] == "gap":
            continue  # 坑：走廊断开（露出基座当坑底）
        # 障碍单元内先铺一块 0 高度走廊（箱子/栏的"周围是 z=0 平地"；楼梯的每一级都盖住它）
        meshes.append(_corridor(obstacle["x0"], obstacle["x1"], size_y, corridor, 0.0))
        for part in obstacle["raised"]:
            meshes.append(_corridor(part["x0"], part["x1"], size_y, corridor, part["top"]))

    origin = np.array([cfg.spawn_x, 0.5 * size_y, 0.0])
    return meshes, origin


@configclass
class CMoETrackCompositeTerrainCfg(SubTerrainBaseCfg):
    """**复合道**：各类独立地形里的障碍各自成形、按类型顺序铺在一条 20 m 道上（训练侧不使用）。

    字段默认值＝**训练侧各类地形的实例值**（`CMoE_env_cfg.py::Imgo2CMoERoughEnvCfg` 里
    `gap` / `pyramid_stairs` / `boxes` / `hurdle` 的显式参数），所以"同一难度下本道障碍的尺寸
    ＝它在自己那类地形里的尺寸"（逐项对照见 `docs/cmoe_mix_test_scene_2026-10-04.md`）。
    与独立地形的**有意差异只有三处**，逐条写在字段注释里：
    ① `stairs_num_steps = 4`（任务书要求 4 级；训练实例是 6 级，**单级步高/步深逐字相同**）；
    ② `hurdle_height_fraction` / `hurdle_spacing_fraction` —— 独立地形在区间内**随机**，本道取确定值；
    ③ `obstacle_spacing` —— 复合道自己的**布局**参数（相邻障碍单元之间的平地），独立地形没有这一层。
    """

    function = track_composite_terrain
    # ---------------- 走廊 / 基座 / 出生点（沿用评测场景既有设定，**不新造**）
    corridor_width: float = 0.80
    pit_depth: float = 0.50
    spawn_x: float = 0.75
    # `spawn_x` 前方**至少**这么多米持平地，之后才允许出现第一个障碍（用户硬约束 ≥ 1.5 m）。
    spawn_clearance: float = 1.50
    # ---------------- 布局：相邻障碍之间的**均匀平地间隔**
    # 负值（= COMPOSITE_SPACING_AUTO）⇒ **反算**成"刚好占满整条 `size[0]`"的值；
    # 非负值 ⇒ 显式间隔（必须 ≥ `min_obstacle_spacing`，否则 `ValueError`）。
    obstacle_spacing: float = COMPOSITE_SPACING_AUTO
    min_obstacle_spacing: float = COMPOSITE_MIN_OBSTACLE_SPACING
    suggested_spacing_range: tuple[float, float] = COMPOSITE_SUGGESTED_OBSTACLE_SPACING
    # ---------------- ① 坑（`track_gap_terrain` 的沟宽律）
    # 训练实例 `CMoETrackGapTerrainCfg(gap_width_range=(0.12, 0.32), …)`；本道放 **2 个坑**（首尾各一，
    # 见 `COMPOSITE_OBSTACLE_SEQUENCE`）。
    gap_width_range: tuple[float, float] = (0.12, 0.32)
    # ---------------- ② 楼梯（`track_stairs_terrain` 的步高律 ＋ step_depth）
    # 训练实例 `CMoETrackStairsTerrainCfg(step_height_range=(0.05, 0.20), num_steps=6)`；
    # **本道取 4 级**（用户确认的"上 4 级 + 下 4 级紧贴"），单级步高/步深与训练实例逐字相同
    # （d = 0.70 ⇒ 0.05 + 0.70×0.15 = 0.155 m、步深 0.30 m）。
    stairs_num_steps: int = 4
    stairs_step_height_range: tuple[float, float] = (0.05, 0.20)
    stairs_step_depth: float = 0.30
    # ---------------- ③ 箱子/台阶块（`track_step_terrain` 的高/长律 ＋ step_spacing）
    # 训练实例 `CMoETrackStepTerrainCfg(step_height_range=(0.08, 0.30), step_length_range=(0.30, 0.50),
    # step_spacing=1.30, num_steps=2)` —— 逐字相同（`box_spacing` 是单元**内部**间距）。
    box_count: int = 2
    box_height_range: tuple[float, float] = (0.08, 0.30)
    box_length_range: tuple[float, float] = (0.30, 0.50)
    box_spacing: float = 1.30
    # ---------------- ④ 栏（`track_hurdle_terrain` 的高度律与厚度律；训练实例＝类默认值）
    hurdle_count: int = 2
    hurdle_len_range: tuple[float, float] = (0.04, 0.12)
    hurdle_height_min_slope: float = 0.08
    hurdle_height_max_base: float = 0.06
    hurdle_height_max_slope: float = 0.10
    hurdle_spacing_range: tuple[float, float] = (0.48, 0.80)
    # 独立地形对**每道栏**的高度、以及相邻栏的间距**在区间内随机**；复合道必须确定性可核
    # ⇒ 取区间的这两个分位（0.5 ＝ 中点）。栏高 d = 0.70 ⇒ [0.056, 0.130]，中点 **0.093 m**；
    # 栏间距中点 = (0.48 + 0.80)/2 = **0.64 m**。`track_hurdle_terrain` 本身一字未改。
    hurdle_height_fraction: float = 0.5
    hurdle_spacing_fraction: float = 0.5
