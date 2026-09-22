"""打印拖曳环境的接触传感器实际配置，用于定位 `Filter pattern` 报错来源。

用法（在有 GPU 的训练机、从仓库根执行）：

    bash imgo2_rl/scripts/run_isaaclab.sh imgo2_rl/scripts/tools/dump_towing_contacts.py

只构造 1 个环境、不做任何训练，几秒即退出。它回答三个问题：

1. 场景里到底有哪几个 ContactSensor，各自的 `prim_path`（传感器本体）是什么；
2. 每个传感器的 `filter_prim_paths_expr` 在**展开后**的具体内容（这是 PhysX 报错里
   `Filter pattern` 的来源）；
3. 每个传感器的 body 数与 `filter_count`，用来判断「每个 filter 项是否每环境只解析出 1 个 prim」。

`expected/found` 的语义见 `docs/towing_observability_2026-09-22.md`。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Imgo2-towing-upper-rl-lab")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import imgo2_rl.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab.sensors import ContactSensor  # noqa: E402


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    scene = env.unwrapped.scene

    print("\n" + "=" * 78)
    print(f"场景接触传感器清单（num_envs={args_cli.num_envs}）")
    print("=" * 78)
    for name in sorted(scene.sensors.keys()):
        sensor = scene.sensors[name]
        cfg = sensor.cfg
        raw_filter = list(getattr(cfg, "filter_prim_paths_expr", []) or [])
        print(f"\n[{name}]  type={type(sensor).__name__}")
        print(f"  sensor prim_path      : {cfg.prim_path}")
        print(f"  num_bodies(每环境)    : {getattr(sensor, '_num_bodies', '?')}")
        print(f"  body_names            : {getattr(sensor, 'body_names', '?')}")
        if isinstance(sensor, ContactSensor):
            view = getattr(sensor, "contact_physx_view", None)
            print(f"  filter_count          : {getattr(view, 'filter_count', '?')}")
            fm = sensor.data.force_matrix_w
            print(f"  force_matrix_w.shape  : {None if fm is None else tuple(fm.shape)}")
        print(f"  filter 项数            : {len(raw_filter)}")
        for i, expr in enumerate(raw_filter[:25]):
            print(f"    [{i:2d}] {expr}")
        if len(raw_filter) > 25:
            print(f"    ... 其余 {len(raw_filter) - 25} 项省略")

    print("\n" + "=" * 78)
    print("原始 env_cfg 里的静态定义（未经 Isaac Lab 展开）")
    print("=" * 78)
    seen = set()
    found = False
    # ConfigClass 可能把无注解赋值保留在类命名空间、也可能搬到实例上，两处都扫。
    for source, ns in (("class", vars(type(env_cfg.scene))), ("instance", vars(env_cfg.scene))):
        for name, term in ns.items():
            if term.__class__.__name__ != "ContactSensorCfg" or name in seen:
                continue
            seen.add(name)
            found = True
            print(f"\n[{name}] ({source} 命名空间) prim_path={term.prim_path}")
            filters = list(term.filter_prim_paths_expr)
            print(f"  filter_prim_paths_expr ({len(filters)} 项):")
            for expr in filters:
                print(f"    {expr}")
    if not found:
        print("\n!! 未在 class/instance 命名空间里找到 ContactSensorCfg；"
              "请把上面「场景接触传感器清单」贴出即可。")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
