# IsaacLab-rsl5 服务器安装核对（2026-10-10）

已确认本机 `/home/tommy0929/Documents/IsaacLab-rsl5` 是 Isaac Lab 本体；目录名不是额外软件。`VERSION` 写 2.3.2，但当前实际 Git 提交为 `c22775241e28f465fe345fa1a482ad6d29d712b0`（origin 为官方 `isaac-sim/IsaacLab`，工作树干净）。该提交的 `source/isaaclab_rl/setup.py` 声明 `rsl-rl-lib==5.0.1`；不是仅凭 VERSION 就能确定等价于官方 v2.3.2 标签。服务器应固定相同提交或上传相同源码，不需再并排安装第二份 Isaac Lab。

本机 conda `env_isaaclab` 分发版本：isaacsim 5.1.0.0、isaaclab 扩展 0.54.4、rsl-rl-lib 5.0.1、torch 2.7.0+cu128。扩展版本 0.54.4 与仓库 VERSION 的 2.3.2 是不同层级。旧服务器 2.2.1/RSL-RL 2.3.3 基线不能当成本机新栈已经完成训练验证。

下面以 Linux x86_64、新 conda 环境、工作目录 `~/work` 为安装示例；镜像已有可用的 Isaac Sim 5.1/Python 3.11 环境时，复用该环境，跳过新建环境与再次安装 Sim。仅仅镜像文件里存在 Isaac Sim，不代表新 conda 环境能导入它。

```bash
conda create -n isaaclab python=3.11 -y
conda activate isaaclab
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

mkdir -p ~/work
cd ~/work
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab-rsl5
cd IsaacLab-rsl5
git checkout c22775241e28f465fe345fa1a482ad6d29d712b0
./isaaclab.sh -i rsl_rl
```

已读本机 `isaaclab.sh`：`-i rsl_rl` 安装 Lab 扩展及该算法 extra，Linux x86_64 自动选择 torch 2.7.0 / torchvision 0.22.0 / cu128。可能安装所需系统包，服务器需具备相应权限。此处 clone 路径是新建目录示例，已有同名目录时直接进入现有副本检查版本。

把当前 Imgo2 工作区（包括尚未提交的滞空奖励修改）同步至服务器 `~/work/imgo2` 后：

```bash
cd ~/work/imgo2/imgo2_rl
python -m pip install -e source/imgo2_rl
python -m pip install -e scripts/rl_lab
export IMGO2_ISAACLAB_PYTHON="$(command -v python)"
bash scripts/run_isaaclab.sh --check
```

版本检查：

```bash
python - <<'PY'
import sys
from importlib.metadata import version
print(sys.version)
for name in ("isaacsim", "isaaclab", "rsl-rl-lib", "torch"):
    print(name, version(name))
PY
```

依据：[官方同提交的 RSL-RL 依赖](https://github.com/isaac-sim/IsaacLab/blob/c22775241e28f465fe345fa1a482ad6d29d712b0/source/isaaclab_rl/setup.py)、[官方 pip 安装指令](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/docs/source/setup/installation/pip_installation.rst)。

已完成／已修正：查清目录与版本关系，提供对应真实提交的安装命令，避免把 VERSION 等同于 Git 标签。待确认：服务器镜像已有环境、驱动、安装和 CUDA 自检、新栈 AMP 小环境冒烟；本轮没有服务器访问或安装执行。项目 AMP 使用自有 rl_lab，普通 PPO/HIM-Loco 仍需按训练栈兼容性记录核对，不能因安装成功就宣布所有任务兼容。
