# 部署策略 Git 历史核对（2026-10-14）

本轮仅核对本地 Git 历史与文件内容身份，未 fetch、切换分支、更新策略或运行模型。

## 已确认

- 当前分支为 `xiaoji/imgo2_real`，HEAD `a4b2103`；本地 `main` 为 `287c196`，本地缓存 `origin/main` 为 `4d66f2c`。两份主分支引用已分叉，不能混为同一版本。
- 本地 main 的 AMP policy 最后修改提交为 `287c196`（2026-10-07）；Git blob 为 `b3bc748c36d5319b447185be95d1f003b037971a`。
- origin/main 缓存历史中 AMP policy 最后修改为 `dc21184`（2026-09-18，提交说明为 24500 轮 AMP）；blob 为 `edce3140f331a3dd45adcd038de4e8887e873606`。PPO policy 最后修改为 `ee0a669`（2026-09-17）；towing policy 于 `ddf118e`（2026-09-25）新增。
- 当前分支 AMP policy 最后修改为 `a4b2103`（2026-10-14）；blob 为 `b4891a0d3a10c864a2434d172e153a59d05545d0`，工作区文件 `git hash-object` 与 HEAD 一致，且不同于两份 main 引用。当前分支 PPO/towing policy 与 origin/main 相同。
- 依据：`git log <ref> --name-status -- imgo2_deploy/policy/**/policy.pt`、`git ls-tree -r <ref> imgo2_deploy/policy`、`git hash-object imgo2_deploy/policy/imgo2/amp/policy.pt`。对比的是实际 Git 对象，不是时间戳。

## 已修复与待确认

用户报告已执行 fetch 后再次核对：`origin/main` 仍为 `4d66f2c`（2026-10-12），AMP policy 最后修改仍为 `dc21184`；仓库非浅克隆。已将该引用的 AMP policy 与同版本 config 提取到 `/tmp/imgo2_origin_main_amp_dc21184/`，提取文件 Git blob 与源对象一致（`edce3140f331a3dd45adcd038de4e8887e873606`）。未覆盖当前部署策略，未运行旧策略；是否替换及与当前 real 模型适配仍待确认。临时提取目录不作为永久归档。

- 已修复：本轮未修改代码或策略；已明确本地 main、origin/main 缓存与当前分支的策略差异。
- 待确认：未连接远程刷新引用，GitHub 当前最新 main 是否有后续更新需 fetch 后复核；本轮不评价模型质量或重新验证导出契约。
