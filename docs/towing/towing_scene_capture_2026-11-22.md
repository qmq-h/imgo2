# Isaac Sim 场景截图（机器人+小车 / 地形）

> 用于产出两张图：
> 1. 机器人 + 被拉小车的侧视；
> 2. 仿真训练地形的俯视。
>
> 入口在 `play_towing_test.py` 的 `--capture-dir`；需要 `--enable-cameras`。
> **本机无 GPU，未实际渲染验证**；相机位置/构图需在训练机上看第一版后再微调。

## 运行

```bash
cd /root/Desktop/imgo2

bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/towing/play_towing_test.py \
    --headless \
    --enable-cameras \
    --capture-dir docs/towing/data/scene_capture_2026-11-22 \
    --num-envs 1 \
    --columns 0 \
    --velocities 1.0 \
    --cart-masses 10 \
    --tow-duration 24 \
    --coast-duration 5 \
    --stop-progress 10 \
    --write-csv none \
    --compact-log \
    --policy ppo \
    --policy-checkpoint=imgo2_deploy/policy/imgo2/ppo/policy.pt \
    --output-dir=imgo2_rl/logs/towing/play_test/scene_capture_2026-11-22
```

- 不传 `--headless` 会打开 Isaac Sim GUI；相机截图逻辑相同。
- `--capture-dir` 会强制 `--num-envs 1`，并只取第一列，避免多 env 干扰构图。
- 侧视相机：机器人和小车**实际位置的中点**，相机在中心 + `(0, -4.5, 1.8)`，看向中点；
- 地形相机：网格**中间行**、第 30 列上方 `(-40, -80, 80)` 看总览；另拍一张第 45 列（减速带/波浪附近）`(-28, -35, 22)` 近景；
- 第一版曾用 env0 原点/第 0 行边缘，导致机器人被裁到左下、地形一半是空背景；2026-11-22 已修正。
- 也可以在 GUI 里手动调 Isaac Sim viewport 相机后截图；这个脚本只是可复现版本。

## 输出

```text
docs/towing/data/scene_capture_2026-11-22/
  side.png       # 机器人 + 小车侧视
  terrain.png          # 地形总览
  terrain_close.png    # 地形近景（减速带/波浪附近）
  terrain_montage.png  # 5 类地形各一张近景拼图
  terrain_flat/5deg/10deg/bump/wave.png
```

## 可能需要调整

- 侧视如果机器人/小车不在画面中心，改 capture block 里的 `center/eyes`；
- 地形图如果太平/看不清坡，优先看 `terrain_montage.png`（flat/5°/10°/bump/wave 五格）；
- 仍不够清楚就调近景 `(-16,-20,11)` 或 `focal_length=18.0`；
- 相机路径在 env0 下，地形相机用世界坐标；多 env 时只拍 env0。

## 限制

- 这是 **Camera sensor 渲染**，不是 viewport GUI 截图；图像是 Isaac Sim 的 RGB 输出。
- 未在本机运行；Camera 配置/构图需 GPU 训练机确认。
- 需要 Isaac Sim 支持 headless camera：`--enable-cameras`；
- 保存后直接 `os._exit(0)`，避免 Kit 的 Python finalize 挂住（上一版卡在保存后）。
