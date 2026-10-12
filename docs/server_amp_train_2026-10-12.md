> 本文为同日早期历史步骤。当前按用户要求重写的两种环境路线见 [服务器训练.md](服务器训练.md)，以其版本、目录和命令为准。

# 服务器安装 imgo2：简化步骤

适用已经能启动Isaac Sim的isaaclab311环境。用户当前报错是模型路径不存在，不是pip安装失败；本轮不要求重新安装Sim或Lab。

## 1. 上传完整项目

把本地当前完整的imgo2文件夹上传到服务器 `/root/Desktop/imgo2`。这是服务器目录示例；若采用此位置，后面的命令可直接使用。

目录至少要是：

```text
/root/Desktop/imgo2/
├── imgo2_rl/
│   ├── source/imgo2_rl/
│   ├── scripts/rl_lab/
│   └── datasets/imgo2_motion_new/
└── imgo2_description/
    ├── urdf/imgo2_real.urdf
    └── meshes/
```

不要只上传imgo2_rl一个子目录。当前代码和新数据有未提交更改，直接clone远端不一定包含本轮版本；使用当前本地文件最直接。模型目录必须包含网格，不是只放一份URDF。

## 2. 激活环境、安装项目

在已经能启动Isaac Sim的服务器终端执行：

```bash
conda activate isaaclab311
cd /root/Desktop/imgo2/imgo2_rl

# 清掉之前指向别处的模型/数据覆盖，让代码按项目目录自动寻找。
unset IMGO2_URDF_PATH IMGO2_AMP_MOTION_DIR

python -m pip install -e source/imgo2_rl -e scripts/rl_lab
```

只安装两个包；源码目录需要保留。已有环境不需重装Isaac Sim/Isaac Lab。若新终端还没加载镜像Sim环境，沿用之前配置的激活钩子或先source `/isaac-sim/setup_conda_env.sh`；这是Sim已有环境前提，不是重新pip安装。

## 3. 先跑10轮

```bash
IMGO2_ISAACLAB_PYTHON="$CONDA_PREFIX/bin/python" \
  bash scripts/run_isaaclab.sh scripts/rl_lab/amp/train.py \
  --task=Imgo2-basemove-flat-amp-height \
  --num_envs=4 --max_iterations=10 --seed=1 \
  --run_name=motion_new_smoke --device=cuda:0 --headless
```

开头一行自动用当前环境解释器，无需填写Python绝对路径。检查新数据加载、env.yaml中的速度范围与feet_air_time:null，以及agent.yaml里的混合比例（当前默认0.3）。

## 4. 短训练正常后正式训练

```bash
IMGO2_ISAACLAB_PYTHON="$CONDA_PREFIX/bin/python" \
  bash scripts/run_isaaclab.sh scripts/rl_lab/amp/train.py \
  --task=Imgo2-basemove-flat-amp-height \
  --num_envs=4096 --max_iterations=24000 --seed=1 \
  --run_name=xiaoji_real1_motion_new_no_air --device=cuda:0 --headless
```

从头训练，无resume。4096环境显存不够时改2048/1024。使用当前代码范围vx±1、vy±.4、yaw±1 rad/s、关闭滞空、任务比例0.3，没有命令强制改为0.7。日志保存在imgo2_rl/logs/amp_rsl_rl/base_move_amp_height中。


## 后续反馈：服务器当前实际位置为gpufree-data

用户最新堆栈脚本位于/root/gpufree-data/imgo2/imgo2_rl，Isaac Sim启动后import imgo2_rl.tasks报ModuleNotFoundError。此前/Desktop示例不能作为当前安装目录。应在当前isaaclab311解释器里，从实际gpufree-data目录重新editable安装任务包和rl_lab，随后用同一解释器核对find_spec与包来源（避免在SimulationApp启动前导入会注册任务的包）。可能是未在该环境安装、旧editable路径失效或解释器不同；仅凭堆栈不确认具体一项。

```bash
conda activate isaaclab311
cd /root/gpufree-data/imgo2/imgo2_rl
"$CONDA_PREFIX/bin/python" -m pip install -e source/imgo2_rl -e scripts/rl_lab
"$CONDA_PREFIX/bin/python" -c 'import importlib.util; s=importlib.util.find_spec("imgo2_rl"); print(s.origin if s else "Not found"); assert s is not None'
```

这段是服务器命令，本轮没有跨机执行。PhysX/插件卸载警告发生于主导入异常后，优先恢复项目包可见性。若同解释器自检成功而启动仍失败，需要启动命令及其打印的Python来源进一步核对，不先重装Isaac Sim。服务器导入与短训练仍待验证（AMP-16）。
