> 此文是此前IsaacLab-rsl5分支安装的历史记录。当前AMP服务器安装不要求切换该分支；已有环境和空白环境两条路线统一见 [服务器训练.md](../xiaoji_cmd/服务器训练.md)。

# 全新 Ubuntu 安装 AMP 训练环境（2026-10-10）

适用 Ubuntu 22.04/24.04 x86_64、具有 RT Core 的 NVIDIA GPU，Bash 与 sudo 权限。以当前 `xiaoji/imgo2_real` 分支的 AMP Height 为目标；不把普通 PPO 与旧 RSL-RL 接口兼容性视为已验证。安装位置 `~/work` 和 `~/miniforge3` 都是新机器示例，已有同名环境／目录时不要覆盖。

采用 Python 3.11、Isaac Sim 5.1.0 的 pip 安装、Isaac Lab 提交 `c22775241e28f465fe345fa1a482ad6d29d712b0`（声明 RSL-RL 5.0.1）。pip 安装的 Sim 在环境内可见，不需要 `/isaac-sim` 链接或 `setup_conda_env.sh`。此提交用于复现现有栈，并非推荐升级到最新版本。

## 1. 主机 GPU 前提

先运行 `nvidia-smi`。云服务器／容器驱动由宿主提供；如 GPU 不可见，需要平台修复挂载。裸机 Ubuntu 尚未安装驱动时：

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

重启后检查 `nvidia-smi`。官方 5.1 要求页列出 Linux 测试驱动 580.65.06；这不是对所有更早驱动均不可运行的断言。A100/H100 等无 RT Core 的 GPU 不在该 Sim 的支持范围。驱动不能以 pip CUDA 运行库替代。

## 2. 系统工具与 Python 环境

以下按阶段执行，任何阶段失败先处理报错。

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates build-essential cmake \
  libgl1 libegl1 libglib2.0-0 libx11-6 libxext6 libxi6 libxrender1 \
  libxrandr2 libxcursor1 libxinerama1 libvulkan1 vulkan-tools libsm6 libice6

curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o /tmp/imgo2_Miniforge3.sh
bash /tmp/imgo2_Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
conda create -n isaaclab311 python=3.11 pip --override-channels -c conda-forge -y
conda activate isaaclab311
unset VIRTUAL_ENV
python -c 'import sys; print(sys.executable); print(sys.version)'
python -m pip install --upgrade pip 'setuptools<82' wheel toml
```

已有可用 Conda 时跳过 Miniforge 下载／安装，source 对应的 conda.sh 后创建环境。root shell 可直接执行 apt-get，其他命令不需要 sudo。

## 3. Sim 与 Lab

```bash
python -m pip install 'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com
mkdir -p "$HOME/work"
cd "$HOME/work"
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab-rsl5
cd "$HOME/work/IsaacLab-rsl5"
git checkout c22775241e28f465fe345fa1a482ad6d29d712b0
./isaaclab.sh -i rsl_rl
```

无需再装第二份 Isaac Lab 2.3.2；保持可编辑安装的源码目录。CUDA 运行库随 PyTorch 安装，本流程不要求另装完整 CUDA Toolkit。首次 Sim 导入／启动按提示接受许可。

## 4. 项目与持久的解释器选择

```bash
cd "$HOME/work"
git clone --branch xiaoji/imgo2_real https://github.com/qmq-h/imgo2.git
cd "$HOME/work/imgo2/imgo2_rl"
python -m pip install -e source/imgo2_rl
python -m pip install -e scripts/rl_lab

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/imgo2_python.sh" <<'EOF'
export IMGO2_ISAACLAB_PYTHON="$CONDA_PREFIX/bin/python"
EOF
source "$CONDA_PREFIX/etc/conda/activate.d/imgo2_python.sh"
```

分支来自本机已存在的远端跟踪引用；未执行本轮远端 fetch。clone 只取得已推送的版本，本地未提交修改需另外同步。本轮不推送、不修改 Git 状态。

## 5. 前置检查与短训练

```bash
python -m pip check
python -m pip show isaacsim isaaclab isaaclab_rl rsl-rl-lib imgo2_rl rl_lab
python -c 'import imgo2_rl, rl_lab, pybullet_utils; print(imgo2_rl.__file__); print(rl_lab.__file__)'
python scripts/tools/check_asset_paths.py
bash scripts/run_isaaclab.sh --check

bash scripts/run_isaaclab.sh scripts/rl_lab/amp/train.py \
  --task=Imgo2-basemove-flat-amp-height \
  --num_envs=4 --max_iterations=10 --device=cuda:0 --headless
```

检查需要看到正确环境／项目路径、CUDA available True 和 CUDA solve OK；短训练需实际跑完 10 轮，不能以 Sim 已启动认定成功。首次加载可能下载资源／编译 shader，保留联网。

## 6. 后续新终端与正式训练

```bash
conda activate isaaclab311
cd "$HOME/work/imgo2/imgo2_rl"
bash scripts/run_isaaclab.sh scripts/rl_lab/amp/train.py \
  --task=Imgo2-basemove-flat-amp-height \
  --num_envs=4096 --max_iterations=24000 --device=cuda:0 --headless
```

4096 只是当前任务默认规模，显存不足时用 1024／2048。长训练命令仅供用户在短训练通过后执行，本轮未启动。日志保存在项目 `imgo2_rl/logs/amp_rsl_rl/base_move_amp_height/`。

## 状态与依据

已修复：安装指令覆盖新系统工具、Python 3.11、固定 Lab 提交、两份可编辑项目包、持久解释器选择和本仓库 CUDA 库启动器，明确区别已有二进制 Sim 复用路线。已核对：本机 setup.py、启动器与训练 CLI、资源路径离线检查，命令 shell 语法。待确认：新机器实际驱动／GPU、下载和依赖解析、Sim 导入、完整 10 轮 AMP；本轮未在新 Ubuntu 安装或运行仿真。普通 PPO 与 RSL-RL 5 的兼容性另需对应入口验证。

来源：[NVIDIA 5.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)、[NVIDIA 5.1 Python 安装](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)、[Miniforge 官方](https://github.com/conda-forge/miniforge)、[固定 Lab 提交的依赖](https://github.com/isaac-sim/IsaacLab/blob/c22775241e28f465fe345fa1a482ad6d29d712b0/source/isaaclab_rl/setup.py)。
