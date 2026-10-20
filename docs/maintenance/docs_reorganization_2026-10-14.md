# docs 散文件归类（2026-10-14）

## 现象与原因

rebase 合回的近期记录未沿用已有分类，docs 根目录重新积累 AMP、Gazebo、模型、安装和 Git 文件。部分历史 Markdown 链接将仓库根路径写成了文档相对路径，另有目标已被移入子目录。

## 已修复

本轮将 docs 根目录 106 个文件归入现存目录，保留原文件名；没有新建目录。用户在本轮开始前已将两份命令文档移入 `../xiaoji_cmd/`，保持该位置并修正入口链接。

| 现存目录 | 本轮移入数量 | 内容 |
|---|---:|---|
| `../amp/` | 15 | AMP 奖励、数据采集、步态、训练和策略分析 |
| `../amp/data/` | 39 | AMP/Isaac 离线 JSON 与图表 |
| `../deployment/` | 37 | Gazebo、sim2sim、硬件/IMU 记录及对应数据、日志 |
| `../training/` | 5 | Isaac Lab 环境、安装、服务器配置和 RL 阅读指南 |
| 本目录 `maintenance/` | 10 | 架构、模型核对、Git/rebase 与命令维护记录 |

同步修正 README、command.md、各文档相对链接及源码/配置注释中的文档位置。另修正 27 处原有失效链接，仅采用仓库实际存在的目标或唯一同名文档。机器可读实验 JSON 中的来源路径保留原值，以维持实验出处和字节身份。

AGENTS.md 和 README 已固定默认约定：除非用户明确另有要求，新文档直接归入现存对应子目录，不放在 docs 根目录、不新建目录；机器可读结果优先使用对应主题已有 data/，没有则放主题目录。用户命令文档继续使用 xiaoji_cmd/。

## 验证

使用 `/usr/bin/python3` 标准库完成路径和哈希核对：

- docs 根目录没有散文件，分类前后目录集合一致。
- 106 个来源均有归类目标，没有覆盖同名文件。
- 其中 67 份非 Markdown 产物（JSON、PNG、日志）移动前后 SHA256 相同。
- Markdown 本地引用没有新增失效目标；剩余两条属于原有缺失报告。
- Git 无未解决冲突索引、无 rebase 状态目录，当前提交 `a4b2103`；README 同步更新 rebase 状态，历史记录仍保留当时情况。
- `git diff --check` 通过；训练策略、模型、motion 和算法代码未改。源码仅修正文档位置注释。

逐文件迁移表、引用修复表与剩余缺失清单见 [机器可读记录](docs_reorganization_2026-10-14.json)。未运行训练、Gazebo、硬件或算法测试；本轮仅整理文档和引用。整理期间用户同步将部署 PD 调整为 25 / 0.5，本轮保留这些数值，不将其计入文档整理改动或运行验证。

## 待修复／待确认

[拖曳 latent 记录](../towing/towing_latent_cluster_2026-10-12.md) 仍引用本机不存在的两份历史报告：

- `imgo2_rl/logs/towing/play_test/v3_800_policy_c008/report.json`
- `imgo2_rl/logs/towing/play_test/v3_800_policy/report.json`

需要原训练机提供报告，或维护者确认可替代的归档后，才能完成这两处引用核对。本轮没有虚构报告或删除历史结论；README 的 DOC-CLASS-01 保留这一待确认项。
