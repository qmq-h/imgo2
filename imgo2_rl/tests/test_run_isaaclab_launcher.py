"""`imgo2_rl/scripts/run_isaaclab.sh` 的离线回归测试（纯标准库 + 子进程，不需要 GPU）。

守两件事：
1. `--check` 能跑通，并报告 `libtorch_cuda_linalg.so` 存在且 `dlopen` 成功；
2. 启动器确实把 `<site-packages>/torch/lib` 追加进了子进程的 `LD_LIBRARY_PATH`
   —— 这是 `libtorch_cuda_linalg.so: cannot open shared object file` 这个报错的直接解药
   （机制见 `docs/isaaclab_launcher_libtorch_dlopen_2026-09-20.md`：TP 里的
   `libtorch_cuda.so` 只有 `DT_RPATH`，而 glibc **不会**用 RPATH 解析显式 `dlopen(裸名)`，
   所以必须靠 `LD_LIBRARY_PATH`）。

Run: python3 -m unittest discover -s tests -p test_run_isaaclab_launcher.py
"""

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_isaaclab.sh"
PYTHON = os.environ.get("IMGO2_ISAACLAB_PYTHON", "/opt/conda/envs/isaaclab/bin/python")


def _run(args, **kwargs):
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        capture_output=True, text=True, timeout=300, **kwargs)


@unittest.skipUnless(Path(PYTHON).exists(), f"没有 {PYTHON}（用 IMGO2_ISAACLAB_PYTHON 指定）")
class LauncherTests(unittest.TestCase):
    def test_script_is_executable(self):
        self.assertTrue(os.access(LAUNCHER, os.X_OK), f"{LAUNCHER} 需要可执行位")

    def test_check_reports_linalg_library_is_loadable(self):
        """`--check` 必须证明 `libtorch_cuda_linalg.so` 能找到且能 dlopen。"""
        result = _run(["--check"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("linalg .so       : 存在", result.stdout)
        self.assertIn("dlopen(linalg)   : OK", result.stdout)
        self.assertIn("CPU solve        : OK", result.stdout)

    def test_launcher_puts_torch_lib_on_ld_library_path(self):
        """子进程的 `LD_LIBRARY_PATH` 里必须出现 `torch/lib`（这就是修好的关键）。"""
        code = "import os; print(os.environ.get('LD_LIBRARY_PATH', ''))"
        result = _run(["-c", code])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("torch/lib", result.stdout)
        # 同时也应带上 pip 版 CUDA 运行库目录（libtorch_cuda_linalg.so 的依赖）
        self.assertIn("nvidia/", result.stdout)

    def test_launcher_reproduces_and_fixes_the_bare_name_dlopen(self):
        """同一句 `dlopen(裸名)`：不给路径会失败、走启动器会成功 —— 这就是报错的机理。

        负例用 `env -u LD_LIBRARY_PATH` 保证与当前 shell 无关；若这台机器的 ld.so 缓存里
        恰好有该库（正常安装下没有），负例就不成立，此时只打印不改判。
        """
        code = ("import ctypes;"
                "ctypes.CDLL('libtorch_cuda_linalg.so', mode=ctypes.RTLD_GLOBAL);"
                "print('bare-dlopen-ok')")
        bare = subprocess.run(["env", "-u", "LD_LIBRARY_PATH", PYTHON, "-c", code],
                              capture_output=True, text=True, timeout=300)
        with_launcher = _run(["-c", code])
        self.assertEqual(with_launcher.returncode, 0, with_launcher.stdout + with_launcher.stderr)
        self.assertIn("bare-dlopen-ok", with_launcher.stdout)
        if bare.returncode == 0:
            print("[note] 这台机器 ld.so 缓存里已经有该库，负例不成立（不影响结论）")


if __name__ == "__main__":
    unittest.main()
