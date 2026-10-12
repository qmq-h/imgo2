# `libtorch_cuda_linalg.so: cannot open shared object file` 的根因、复现与修法（2026-09-20）

## 1. 现象

用户在终端跑步态回放时，**环境还没建起来**就崩（`gym.make` 内部建场景阶段）：

```
File ".../isaaclab_tasks/utils/hydra.py", line 81, in wrapper
  ...
File ".../isaacsim/core/utils/torch/transformations.py", line 53, in get_local_from_world
    local_transform = torch.linalg.solve(
RuntimeError: Error in dlopen: libtorch_cuda_linalg.so: cannot open shared object file: No such file or directory
```

调用链：`gym.make` → `InteractiveScene` → `spawn_from_urdf` → `create_prim`
→ `XFormPrim.__init__` → `set_world_poses` → `get_local_from_world` → `torch.linalg.solve`。
训练跑了几万轮都没事，**同一个安装、同一份文件**，回放却崩了。

## 2. 根因（三步，全部离线可验证）

1. **Isaac Lab 把 Isaac Sim 的 prim/变换后端硬编码成 torch**：
   `isaaclab/sim/simulation_context.py:269/279` 里 `super().__init__(..., backend="torch", ...)`
   （Isaac Sim 的 `SimulationContext` 默认是 `backend="numpy"`）⇒ 建场景时所有
   `isaacsim.core.utils.torch.*` 的工具都会用 torch，`device` 又是 `cuda` ⇒ `torch.linalg.solve`
   跑 CUDA 张量。
2. **这一步的实现放在一个延迟加载的库里**：CUDA 的 linalg 内核在
   `<site-packages>/torch/lib/libtorch_cuda_linalg.so`，torch 在第一次用到时 `dlopen` 它。
   该文件在 `libtorch_cuda.so` 里的字符串是**裸文件名**（没有目录前缀）：

   ```bash
   $ strings -a .../torch/lib/libtorch_cuda.so | grep libtorch_cuda_linalg
   libtorch_cuda_linalg.so
   ```

3. **glibc 不会用 `DT_RPATH` 解析显式 `dlopen`**。`libtorch_cuda.so` 只有 RPATH（没有 RUNPATH）：

   ```bash
   $ readelf -d .../torch/lib/libtorch_cuda.so | grep -i runpath
    0x000000000000000f (RPATH)  Library rpath: [$ORIGIN/../../nvidia/cublas/lib:...:$ORIGIN]
   ```

   RPATH 只作用于 **`DT_NEEDED`** 的解析（所以 `ldd`/`readelf` 看起来一切正常、
   依赖也都能找到），对 `dlopen("裸名")` 无效；`/etc/ld.so.cache` 里也没有这个库。
   ⇒ 裸名 `dlopen` 只能靠 **`LD_LIBRARY_PATH`** 命中 `torch/lib`。

**这就解释了"训练能跑、回放崩"**：能不能成功只取决于**当时 shell 的 `LD_LIBRARY_PATH`** 里有没有
`<site-packages>/torch/lib`，而不是文件在不在。

## 3. 离线复现（不需要 GPU，逐字相同的报错）

```bash
P=/opt/conda/envs/isaaclab/bin/python
$P -c "import ctypes; ctypes.CDLL('libtorch_cuda_linalg.so')"
# => OSError: libtorch_cuda_linalg.so: cannot open shared object file: No such file or directory

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$($P -c 'import sysconfig,os;print(os.path.join(sysconfig.get_paths()["purelib"],"torch","lib"))')" \
  $P -c "import ctypes; ctypes.CDLL('libtorch_cuda_linalg.so'); print('OK')"
# => OK
```

（沙箱里没有 GPU，所以只复现了 `dlopen` 这一步；CUDA 上的 `linalg.solve` 由用户终端确认。）

## 4. 修法：统一启动器

新增 `imgo2_rl/scripts/run_isaaclab.sh`：

* 把 `<site-packages>/torch/lib` 与所有 `nvidia/*/lib` **追加**到 `LD_LIBRARY_PATH`
  （追加而不是前插：不遮蔽系统 `/usr/local/cuda/lib64` 的同名库），再 `exec` 目标脚本；
* `site-packages` 由解释器自己 `sysconfig` 报出，脚本里不写死 conda 路径；
  解释器默认 `/opt/conda/envs/isaaclab/bin/python`，可用 `IMGO2_ISAACLAB_PYTHON` 覆盖；
* `--check` 自检：打印 torch 版本/路径、那个 `.so` 是否存在、能否 `dlopen`、
  CPU / CUDA 上的 `torch.linalg.solve` 是否可用（有卡时会多打 `CUDA solve : OK`）。

用法（**本项目所有 Isaac Lab 命令都建议走它**）：

```bash
cd /root/Desktop/Imgo2
bash imgo2_rl/scripts/run_isaaclab.sh --check                       # 自检
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/rl_lab/amp/train.py --task=Imgo2-basemove-flat-amp-rlamp --headless
bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/tools/eval_gait.py --task=Imgo2-basemove-flat-amp-rlamp-play --checkpoint=... --headless
```

## 5. 如果还崩（按顺序排查）

1. 先跑 `--check`：
   * `linalg .so : **不存在**` ⇒ torch 是 CPU-only 或装坏了：按 Isaac Sim 5.0 对应的版本重装
     （`pip install --force-reinstall torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128`），
     或用 Isaac Lab 的 `environment.yml` 重建环境。
   * `dlopen(linalg) : **失败**` ⇒ 缺的是**依赖**（CUDA 运行库），检查 `nvidia-cusolver/cublas/nvjitlink`
     是否齐全、`LD_LIBRARY_PATH` 是否被别的 CUDA 版本遮蔽；`ldd <那个 .so> | grep "not found"` 会点名。
   * `CUDA solve : **失败**` ⇒ 把整段输出贴给维护者，再看驱动/容器 CUDA 版本。
2. **备选方案（绕开 torch 后端）**：Isaac Lab 硬编码的 `backend="torch"` 可以被 monkey-patch 成 numpy，
   让 prim/变换工具走 numpy、根本不调用 `torch.linalg.solve`：

   ```python
   # 放在 import isaaclab 之后、gym.make 之前
   from isaacsim.core.simulation_manager import SimulationManager
   _orig = SimulationManager.set_backend.__func__
   SimulationManager.set_backend = classmethod(lambda cls, val: _orig(cls, "numpy"))
   ```

   代价：Isaac Sim 的 prim 工具全部改用 numpy（建场景稍慢），属于**绕过**而不是修好，
   只在启动器方案确实无效时用；用之前先确认 `--check` 里那两行是好的。
3. **不推荐**：把 `torch/lib` 拷到 `/usr/local/lib` 再 `ldconfig`（污染系统、且换了 torch 版本就失效）。

## 6. 验证与限制

* **已验证（离线）**：`strings`/`readelf` 的证据；裸名 `dlopen` 的负例与正例（第 3 节命令）；
  `run_isaaclab.sh --check` 通过；`tests/test_run_isaaclab_launcher.py` 4 项把上述三条锁住
  （含"子进程 `LD_LIBRARY_PATH` 必须含 `torch/lib` 与 `nvidia/`"）⇒ 全仓 87 项通过。
* **未验证**：沙箱没有 GPU，`torch.linalg.solve(cuda)` 与 Isaac Sim 实际建场景只能在有卡的终端确认；
  本记录给出的是**机制层面**的结论 + 可复现的 `dlopen` 证据，不是"回放已经跑通"的证明。
* 附注：这条与项目代码无关，属于**运行环境**问题；只要所有 Isaac Lab 命令都走启动器，
  就不再依赖"某个 shell 里碰巧设过什么"。
