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
- 侧视相机：env0 原点 + `(2.5, -5.0, 2.5)` 看向 `(2.5, 0, 0.6)`；
- 地形相机：第 0 行、第 30 列（10°/减速带附近）上方 450 m 俯视。
- 也可以在 GUI 里手动调 Isaac Sim viewport 相机后截图；这个脚本只是可复现版本。

## 输出

```text
docs/towing/data/scene_capture_2026-11-22/
  side.png       # 机器人 + 小车侧视
  terrain.png    # 地形俯视
```

## 可能需要调整

- 侧视如果机器人/小车不在画面中心，改 capture block 里的 `eyes/targets`；
- 地形图如果拍不全/太远，调整 `450.0` 高度或 `focal_length=18.0`；
- 相机路径在 env0 下，地形相机用世界坐标；多 env 时只拍 env0。

## 限制

- 这是 **Camera sensor 渲染**，不是 viewport GUI 截图；图像是 Isaac Sim 的 RGB 输出。
- 未在本机运行；Camera 配置/构图需 GPU 训练机确认。
- 需要 Isaac Sim 支持 headless camera：`--enable-cameras`。
