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

## 后续安装失败：解释器疑似为 3.13

用户提供的 pip 日志下载了 `pillow-11.3.0-cp313-cp313-...whl`，并列出 ml-dtypes 对 Python ≥3.13 要求 numpy ≥2.1 与 Isaac Lab numpy<2 的冲突；强烈提示安装实际目标是 Python 3.13，与此前手动查看的 Python 3.11.3 不同。尚不能从日志确定是哪一层环境切换造成。

已核对安装脚本：有 CONDA_PREFIX 时取 `$CONDA_PREFIX/bin/python`，其次 VIRTUAL_ENV；uv 虚拟环境存在时 pip 路径又可能走 `uv pip install`。先在服务器核对实际脚本解释器：

```bash
cd /root/Desktop/work/IsaacLab-rsl5
python -c 'import sys; print(sys.executable); print(sys.version)'
./isaaclab.sh -p -c 'import sys; print(sys.executable); print(sys.version)'
python -m pip --version
```

如脚本实际为 3.13，而服务器仅有 base，可建立独立 3.11 环境：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda create -n isaaclab311 python=3.11 -y
conda activate isaaclab311
unset VIRTUAL_ENV
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
cd /root/Desktop/work/IsaacLab-rsl5
./isaaclab.sh -p -c 'import sys; print(sys.executable); print(sys.version)'
./isaaclab.sh -i rsl_rl
python -m pip check
```

保持项目的 numpy<2 约束，先修复解释器选择，不用 `--no-deps` 绕过解析。已定位日志证据与脚本选择机制；待确认服务器实际解释器输出、3.11 下重新解析是否通过。这些修复命令尚未在服务器执行。

## 已确认 base 3.13；复用独立 Isaac Sim Python 的路线

用户后续直接执行 `/opt/conda/bin/python`，确认 base 实际为 3.13.9；之前 `python` 输出来自 `/workspace/isaaclab/_isaac_sim/kit/python/bin/python3`，是 Sim 自带 3.11.13。用户不能新建环境，故改用已有 Isaac Sim 的 `python.sh` 安装与运行。无需删除旧源码或重装 Sim，也无需修改 base Python。此路线会更新镜像原有的 Sim Python 中同名 Lab/RSL-RL 包，并非隔离副本。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda deactivate
unset VIRTUAL_ENV
SIM_ROOT="$(readlink -f /workspace/isaaclab/_isaac_sim)"
"$SIM_ROOT/python.sh" -c 'import sys; print(sys.executable); print(sys.version)'
cd /root/Desktop/work/IsaacLab-rsl5
if [ ! -e _isaac_sim ] && [ ! -L _isaac_sim ]; then
    ln -s "$SIM_ROOT" _isaac_sim
fi
readlink -f _isaac_sim
./isaaclab.sh -p -c 'import sys; print(sys.executable); print(sys.version)'
# 确认上面指向已知 Sim 的 Python 3.11 后再安装
./isaaclab.sh -i rsl_rl
./isaaclab.sh -p -m pip show isaaclab isaaclab_rl rsl-rl-lib
./isaaclab.sh -p -m pip check
```

如已有 `_isaac_sim` 链接指向其他位置，不覆盖它，先检查其实际路径与解释器。后续项目安装使用 `./isaaclab.sh -p -m pip install -e <项目包路径>`，训练使用 `./isaaclab.sh -p <训练脚本> ...`，避免裸 python 再次解析到 base。依据为官方二进制安装路线与脚本在无 CONDA_PREFIX/VIRTUAL_ENV 时回退 `_isaac_sim/python.sh` 的逻辑。已确认根因；修复命令、Sim 具体版本、安装后依赖一致性和 AMP 冒烟仍待服务器验证。

## Conda 创建环境失败的另一原因：无效频道

用户的创建日志在 repodata 获取阶段报清华 `anaconda/pkgs/pro` HTTP 404，环境尚未创建。这是失效频道配置，不能据此认为服务器不允许新建虚拟环境。使用 `--override-channels` 和一个完整频道 URL 临时绕过 pro/free 等历史设置，不修改共享配置：

```bash
conda create -n isaaclab311 python=3.11 --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main -y
conda activate isaaclab311
unset VIRTUAL_ENV
"$CONDA_PREFIX/bin/python" -c 'import sys; print(sys.executable); print(sys.version)'
```

成功后可按镜像提示 source 已有 Sim 的 setup_conda_env.sh，并在新 Lab 目录连接 `_isaac_sim`，无需重新下载 Sim。频道方案已按清华官方帮助和 Conda create 文档核对；待确认服务器执行创建、安装与冒烟结果。暂不替用户重写 .condarc。

## 311 环境训练启动缺少 isaacsim 模块

用户后续确认镜像 Sim 位于 `/isaac-sim`，链接修正后 VS Code 只剩 extraPaths 警告（不影响运行）。但 AMP 训练使用 `/opt/conda/envs/isaaclab311/bin/python` 时，AppLauncher 导入 SimulationApp 报 `ModuleNotFoundError: isaacsim`。解释器已经选对，缺少的是当前 Conda 对已有 Sim 模块的可见性；仅建立 `_isaac_sim` 链接不能保证完成这一步。尚未确认当前终端的 setup_conda_env.sh 是否已生效。

优先在同一 Bash 终端依次激活 isaaclab311、`cd /isaac-sim && source ./setup_conda_env.sh`，再用 `$CONDA_PREFIX/bin/python -c 'from isaacsim import SimulationApp; print("isaacsim import OK")'` 检查。成功后从用户实际项目位置 `/root/gpufree-data/imgo2/imgo2_rl` 重跑 4 环境 × 10 轮 AMP。若仍失败，需提供 `/isaac-sim/setup_conda_env.sh`、`setup_python_env.sh` 内容及实际 `isaacsim/__init__.py` 位置，排查环境脚本和目录布局；不以安装成功的 pip 日志替代 Sim 导入或训练验证。

已完成：区分正确 Conda 解释器与独立 Sim 模块路径，依据 [NVIDIA 5.1 Python 环境说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html) 提供同终端加载与导入检查；待修复／待确认：服务器重新导入与短训练结果。本轮未访问服务器执行。

## 每次激活环境自动加载已有 Sim

用户最新日志已进入 Sim 启动阶段，随后在 `import imgo2_rl.tasks` 失败；因此该终端的 Sim 导入问题已经解除，但项目包的安装环境／可编辑路径仍待核对，不能把两种错误混为一谈。用户称两个项目包已安装，先检查 `pip show` 和 `find_spec`，不直接重复安装。

为持久保存加载步骤，可在服务器的 `isaaclab311` 环境创建激活脚本（以下为服务器专用路径）：

```bash
conda activate isaaclab311
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/imgo2_isaacsim.sh" <<'EOF'
if [ -f /isaac-sim/setup_conda_env.sh ]; then
    pushd /isaac-sim > /dev/null
    source ./setup_conda_env.sh
    popd > /dev/null
fi
EOF
source "$CONDA_PREFIX/etc/conda/activate.d/imgo2_isaacsim.sh"
"$CONDA_PREFIX/bin/python" -c 'from isaacsim import SimulationApp; print("isaacsim import OK")'
```

后续新 Bash 终端只需 `conda activate isaaclab311`，Conda 会运行此脚本。该方法复用 NVIDIA 的环境脚本，避免只补 PYTHONPATH 而遗漏动态库路径。已修复：补充自动加载方案与错误区分；待确认：服务器创建脚本、新终端导入、项目包实际路径和 10 轮短训练。本机未执行服务器命令；激活脚本并非完整的环境变量撤销机制，切换到其他项目建议使用新终端。
