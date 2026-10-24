# towing 策略目录：v6 代码已去掉 vx 偏移头，等待重训导出

## 当前状态（2026-10-24 v6）

- 训练/运行时契约已升 **v6**：动作 **12 维纯关节残差**，policy 帧 **57**、actor **78**、
  critic **72**；原 1 维 vx 偏移头已删除。
- **本目录没有 `policy.pt`**：上一版 v5（58→13）导出件与 v6 不兼容，已从工作树移除
  （需要追溯见 git 历史）；在 v6 重训并导出前，不要放一个旧 `policy.pt` 进来。
- `model_1300.pt` 是 **v5 训练 checkpoint 留档**（10,245,431 B，
  sha256 `a5478081…`），**不能**被 v6 代码加载；留着只作历史对照。

## 重训后如何导出

用 v6 新 run 的 checkpoint（不要用 `model_1300.pt`）：

```bash
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/towing/play.py \
    --task=Imgo2-towing-upper-rl-lab --agent=rl_lab_cfg_entry_point --headless \
    --checkpoint=logs/towing_rl_lab/towing_upper/<v6_run>/model_<iter>.pt \
    --export
```

产物默认在 checkpoint 同级 `exported/policy.pt`，接口是 **`[1,57]` frame → `[1,12]`
归一化关节残差**；拷到本目录后再补 `config.yaml`/FSM 接线。导出器会硬校验
`towing_contract` v6（frame 57 / explicit 5 / latent 16），v5 及更早的 checkpoint 直接拒绝。

## 仍未完成（TOW-05）

- `config.yaml` 仍不存在；FSM 没有 towing 状态；C++ `Model::forward` 没有 reset 入口；
- libtorch 实载、action_scale 接线未验证；
- 因此即使 v6 导出成功，也只是“可加载的导出件”，不是已接入部署的策略。
