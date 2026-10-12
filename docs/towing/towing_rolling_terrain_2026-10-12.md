# 平滑起伏地形（2026-10-12）

用户决定：原长下坡／单坡包保留；新增 rough 类似的连续波动地形，每个小坡平滑过渡。

## 已修／已接入

新增可选任务 `Imgo2-towing-upper-rolling-rl-lab`；原 `Imgo2-towing-upper-rl-lab` 仍使用单坡包。起伏生成器沿用 40×20 网格、连接配置、资产、控制频率与 v3 网络维数。地形变化不意味着已有策略已经能适应。

跑道仍为 15×6 m；出生区和出口为平地。x=2.25–9.0 m 放四个不等宽的小坡包，水平宽度依次为 1.5、2.0、1.25、2.0 m，模拟不规则的连续起伏。当前是确定性、纵向为主的平滑高度场，没有直接叠加 rough 的逐点随机噪声，以免又形成细碎棱角。

每个小坡的纵向形状为抬升余弦，局部坐标 u∈[0,w]：

`h(u,y) = tan(|grade|) · w/(2π) · (1−cos(2πu/w)) · cos²(πy/6)`。

车道外横向因子为零。车道中线最大纵向坡度为 0°/5°/10°，延续原列分配；最高小坡的峰高分别约 0、0.0557、0.1123 m。坡顶、坡谷及平地区的高度和一阶导数连续（C1；不宣称曲率连续）。横向 ±3 m 处高度及横向斜率归零，使不同坡度列的接缝也连续。高度场由三角面近似：纵向采样≤0.1 m，横向0.5 m，所有小坡边界均显式采样；网格表面并非解析曲面，已检查面内高度近似误差<3 mm。

实现源为 [rolling_geometry.py](../../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp/rolling_geometry.py)，张量高度计算为 [rolling_torch.py](../../imgo2_rl/source/imgo2_rl/imgo2_rl/tasks/manager_based/towing/mdp/rolling_torch.py)。生成器使用闭合实体网格，避免沿凹凸剖面使用原单坡包的扇形端面三角化。

训练跌倒判据改按所选地形扣除机器人当前 `(x,y)` 的地面高度。起伏任务的 timeout 用纵向坡度上限计算保守弧长界：10 m 目标最多按 10.1041 m 算，0.5 m/s 下约24.2083 s，20 Hz 向上取整为485步；STOP 仍在出口平地10 m处。横向运动不计入这个行程上界，仍依赖现有中线PD。

测试台新增 `--terrain-profile hill|rolling`（默认 hill），同时选择实际 mesh、机器人与小车各自的二维高度、纵向相对俯仰；地形身份写入 experiment/report。离线分段统计按各小坡重复上/下坡划分；没有坡顶平台，不再沿用单坡包的固定位置分段。不同地形不能标为同条件策略/基线配对；历史报告缺该字段时按 hill 解释。相对俯仰仍沿用纵向剖面口径，没有新增横向相对roll指标。

## 验证及限制

解释器：`C:/Users/qmq/AppData/Local/Programs/Python/Python311/python.exe`（Python3.11，标准库 unittest）。本轮五个测试文件共242项：224通过，18因缺torch跳过：

| 测试文件 | 通过 | 跳过 |
|---|---:|---:|
| test_towing_rolling_geometry.py | 7 | 1 |
| test_towing_slope_geometry.py | 8 | 2 |
| test_towing_connection_grid.py | 10 | 0 |
| test_towing_play_test.py | 144 | 7 |
| test_towing_upper_rl_contract.py | 55 | 8 |

新地形数值检查包括出生/停车平地、各接点高度与斜率连续、峰高/坡度、相邻板接缝、弧长余量、三档网格闭合且绕向朝外、无退化面、面内高度近似误差；CLI dry-run 与分段/配对兼容检查通过。张量与标量等价测试已保留，因缺torch未执行。Python AST解析、`git diff --check`、已跟踪忽略文件检查通过。没有启动训练或仿真。

## 训练机短跑入口

从仓库根目录运行（脚本均已确认存在）：

```bash
# 测试台：只核对计划，不启动仿真
python imgo2_rl/scripts/towing/play_towing_test.py --terrain-profile rolling --dry-run

# 新运行先做短冒烟；40环境覆盖所有地形列，4环境只会覆盖平地
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/train.py \
  --task Imgo2-towing-upper-rolling-rl-lab --num_envs 40 --max_iterations 10 --headless

# 冻结底层基线：慢速走出起伏段需要延长tow时长
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
  --terrain-profile rolling --num-envs 800 --tow-duration 20 --coast-duration 5 \
  --write-csv all --headless
```

测试台停车目前仍按时间，20秒是0.5 m/s名义走到10 m的时长；带载跟速滞后可能仍未越坡，应检查实际进度。不要把它与训练侧按距离STOP混为一谈。以上训练／仿真命令本轮未执行。

## 待修复／待确认（TOW-ROLL-01）

状态：**已修，待动态验证**。缺训练机torch与Isaac Lab：需执行张量等价测试、导入mesh与环境短冒烟、读回timeout、确认出生/轮接触、过顶/过谷、lane接缝与离面高度；然后对同地形冻结基线和上层策略做配对，核对跟速、追尾与横向稳定性。现有长下坡没有删除，也没有宣布不可解：持续坡上停车是否可行取决于制动力、滚阻、连接类型、观测与允许前移距离，需要另做能力边界实验。起伏地形提供有限、反复的载荷方向变化，不预先承诺上层收益。
