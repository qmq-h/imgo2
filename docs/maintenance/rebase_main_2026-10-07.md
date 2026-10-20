# real 分支 rebase 完成记录（2026-10-07）

## 现象、根因与处理

用户执行 rebase，重放 f1742db（修改为real urdf）时 README.md 内容冲突。主分支和该提交同时修改顶部状态与维护记录。保留主分支较新的训练状态及全部新增维护记录，合入 real 分支的两条模型生成历史；修正该历史行的表格分隔符。未选择 skip，也未覆盖主分支整份 README。

`git add README.md` 后使用临时 `core.editor=true` 运行 `rebase --continue`，成功更新 xiaoji/imgo2_real。新提交 45d2f8e，父提交为 origin/main 的 ddf118e。原提交仍可用 f1742db 查阅。

## 已修复与验证

- README 两处冲突解决，rebase 完成，分支重新挂接。
- `git rev-list --left-right --count origin/main...HEAD` 为 0 / 1，最新远程主分支是当前分支的祖先。
- 原提交修改的 12 个 imgo2_description 文件逐个比较 Git blob／工作区 hash-object，一致；原模型改动全部保留。
- `git diff --check` 通过；`git ls-files -i -c --exclude-standard` 为空。
- 使用 `C:\Python311\python.exe` 运行 check_asset_paths.py，通过。
- stash 中的代理排查文档已恢复，README 中相应结论合入新的维护记录；原 stash 保留作备份，无需再次 pop。恢复后的文档尚未提交。

## 待修复／待确认与限制

同一解释器运行 check_model_sync.py，未通过：网格为 14 文件／66c496556453，登记基线为 10 文件／8dc5b5995a11；imgo2_real.urdf 与 imgo2_real.gazebo.urdf 未登记。这些资源来自保留的原提交，不能把 rebase 完成视为模型验收通过。README MODEL-05 跟踪：需确认新模型／网格设计基线与 ROS 包映射，确认当前 xacro 与生成物契约，再重新生成、登记检查并复跑。

原提交还新增 imgo2_description/README.md，违反现行单一根 README 约定；本轮为保留提交内容未删除。需先核对引用，将有效说明合入根 README 或 docs 后再移除，见 README DOC-02。

未运行 Isaac Lab、Gazebo、C++ 构建、训练或真机；未推送远程。尚未由用户筛选要舍弃的模型改动，本次保留原提交全部模型内容。
