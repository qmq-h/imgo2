#!/usr/bin/env bash
# 用「带完整库搜索路径」的解释器启动 Isaac Lab 脚本（本项目所有 Isaac Lab 命令的统一入口）。
#
# 为什么需要它：Isaac Lab 的 `SimulationContext` 把 Isaac Sim 的 prim/变换工具后端硬编码成
# `backend="torch"`（`isaaclab/sim/simulation_context.py:269/279`），于是建场景时
# `isaacsim.core.utils.torch.transformations` 会调用 `torch.linalg.solve`。CUDA 张量上的
# `linalg.solve` 由 torch **延迟 dlopen** 的 `libtorch_cuda_linalg.so` 提供；如果那个文件的
# 目录不在加载器的搜索路径里，就会看到：
#
#   RuntimeError: Error in dlopen: libtorch_cuda_linalg.so: cannot open shared object file:
#                 No such file or directory
#
# （2026-09-20 用户跑 `eval_gait.py` 时实测到，卡在 `spawn_from_urdf` → `create_prim` 里。）
# 光靠 `PATH` 上的 python 不够：`LD_LIBRARY_PATH` 里既没有 `<site-packages>/torch/lib`，
# 也没有 pip 安装的 CUDA 运行库目录 `<site-packages>/nvidia/*/lib`。本脚本把这两类目录
# **追加**到 `LD_LIBRARY_PATH`（追加而不是前插，避免遮蔽系统 `/usr/local/cuda` 的同名库），
# 再 exec 目标脚本 —— 这样就不依赖当前 shell 里碰巧设了什么。

set -euo pipefail

# 解释器：默认用本机 Isaac Lab 的 conda 环境，可用 IMGO2_ISAACLAB_PYTHON 覆盖
PY="${IMGO2_ISAACLAB_PYTHON:-/opt/conda/envs/isaaclab/bin/python}"

if [[ ! -x "${PY}" ]]; then
    echo "[run_isaaclab] 找不到解释器：${PY}" >&2
    echo "  用 IMGO2_ISAACLAB_PYTHON=/path/to/python 指定。" >&2
    exit 1
fi

# site-packages 由解释器自己报出来，避免在脚本里写死 conda 路径
SITE="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
TORCH_LIB="${SITE}/torch/lib"

# 组装要追加的目录：torch/lib + 所有 nvidia-*/lib（pip 版 CUDA 运行库）
EXTRA=()
[[ -d "${TORCH_LIB}" ]] && EXTRA+=("${TORCH_LIB}")
while IFS= read -r d; do
    [[ -n "${d}" ]] && EXTRA+=("${d}")
done < <(find "${SITE}" -maxdepth 3 -type d -path "*/nvidia/*/lib" 2>/dev/null | sort)

prepend_or_append() {
    # 把 ${EXTRA[@]} 追加到现有值后面（去重）
    local joined
    joined="$(IFS=:; echo "${EXTRA[*]:-}")"
    if [[ -n "${joined}" ]]; then
        if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
            export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${joined}"
        else
            export LD_LIBRARY_PATH="${joined}"
        fi
    fi
}

if [[ "${1:-}" == "--check" ]]; then
    # 自检：不启动 Isaac Sim，只报告「哪个 torch / 有没有那个 .so / 能不能 dlopen / CUDA 可不可用」
    echo "[run_isaaclab] python      : ${PY}"
    echo "[run_isaaclab] site-packages: ${SITE}"
    prepend_or_append
    echo "[run_isaaclab] LD_LIBRARY_PATH 追加了 ${#EXTRA[@]} 个目录"
    "${PY}" - <<'PYCHECK'
import ctypes
import os
import sys

import torch

lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda_linalg.so")
print(f"  torch            : {torch.__version__} (cuda {torch.version.cuda}) @ {torch.__file__}")
print(f"  linalg .so       : {'存在' if os.path.exists(lib) else '**不存在**'} ({lib})")
try:
    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    print("  dlopen(linalg)   : OK")
except OSError as exc:
    print(f"  dlopen(linalg)   : **失败** {exc}")
try:
    torch.linalg.solve(torch.eye(3), torch.eye(3))
    print("  CPU solve        : OK")
except Exception as exc:  # noqa: BLE001
    print(f"  CPU solve        : **失败** {type(exc).__name__}: {exc}")
print(f"  cuda available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    a = torch.eye(3, device="cuda")
    try:
        torch.linalg.solve(a, a)          # 这一步就是训练/回放里崩的那次调用
        print("  CUDA solve       : OK  ← 问题已解决")
    except Exception as exc:  # noqa: BLE001
        print(f"  CUDA solve       : **失败** {type(exc).__name__}: {exc}")
        sys.exit(2)
PYCHECK
    exit $?
fi

if [[ $# -lt 1 ]]; then
    echo "用法: $0 <脚本.py> [参数...]   |   $0 --check" >&2
    exit 1
fi

prepend_or_append
exec "${PY}" "$@"
