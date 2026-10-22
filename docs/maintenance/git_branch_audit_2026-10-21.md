# Git分支与远端提交核对

2026-10-21。用户要求检查状态、比较远端与当前代码、解释为何自己的提交无法对应。本轮执行`git fetch origin --prune`成功，仅更新远端引用；没有merge、pull、rebase、reset、切分支、提交或push。

## 1. 检查时状态

- 仓库`/home/tommy0929/Documents/imgo2`，当前`xiaoji/imgo2_real`。
- 检查开始时工作区/暂存区均干净，用户当日四个新提交已经完成；最新`fd9e30e`（提交剩余文件），前面依次92220d3、44c83bf、4473e41。
- fetch后本地HEAD `fd9e30ee1b247fd8f446c4f1415a588990f41b86`，远端同名分支`1c6f6c9bc3ec281f4582a74cc39de035d60cab91`。
- 真实分叉：本地独有62、远端独有6；共同祖先`ddf118e3b19b0779f1535b93bf346d5680c3d7d6`。
- origin/main为82533ad；当前分支相对它14 ahead /10 behind。这与同名real远端的62/6是不同对照，不能混为一谈；本轮没有把最新main合入。
- `git ls-files -i -c --exclude-standard`为空，没有已跟踪且被忽略的文件。

“干净”是本轮记录文档前的状态。本轮按维护约定新增本报告/JSON/文件清单并更新根README，之后会有这些未提交文档；不自动提交。

## 2. 根因：10月14日rebase改写了提交身份

reflog显示10月14日10:58从origin/main的4d66f2c开始rebase，13:42完成。远端real仍停在10月9日push后的旧历史，未更新为该次重放结果。提交ID取决于父提交等内容，不能仅凭作者/提交说明判断同一个提交；同一人的同一修改换到新父提交上，Git也会认作新提交。

有证据的对应关系：

| 远端旧提交 | 本地rebase后 | 核验 |
|---|---|---|
| 45d2f8e 修改为real urdf | 85f64ce | 原13个变化文件在对应重放提交中均存在；12个非README文件blob相同，仅README随新基线合并不同 |
| 11a4fce 修改readme | 2ae4c12 | 原3个文件均存在，2个非README文件blob相同，README不同 |
| c642c73 保存更改 | 72617a7 | Git patch等价；原变化文件blob相同 |
| 287c196 保存更改 | 8c44e1c | 原17个变化文件均存在，16个非README blob相同，仅README不同 |
| 97ce9c9 修改readme（合并提交） | rebase后的线性历史 | 该提交有两个父节点；相对第一父节点追加Isaac Lab学习入口，当前README仍保留入口并更新为docs/training路径 |
| 1c6f6c9 更换real amp policy | 2ddc671 | Git patch等价；作者日期相同，提交日期变为10月14日 |

`git cherry`的两个`-`验证补丁等价；README冲突解决/基线变化的提交不会总是标成patch等价。`git range-diff`会对重复“保存更改”及合并展平做启发式匹配，部分自动配对与文件证据不同，不单独依赖它判丢失。

因此“落后6”不是已经发现漏了6份最新工作，也不是说明另一个人改了远端。已核对这些旧提交的主要修改有对应继承；后续文件还被用户继续修改/迁移，不能宣称当前全树与远端逐字节一致。

62个本地独有提交包含：共同祖先ddf118e至rebase基线4d66f2c的48个main侧提交，加之后14个本地线性提交。不是62个都为刚才新增的本机改动。

## 3. 文件内容差异

比较的是两个已提交树：远端旧real HEAD → 当前本地HEAD，不包含本轮新文档。

- 共707条变更：536新增、128重命名、40修改、2删除、1类型变化；Git识别重命名具有阈值，数字不是实际源码文件独立增加量。
- 主要内容：real独立硬件/FSM/诊断接口、ros2_ws迁移与仿真脚本/反馈延迟、部署排查文档/图/JSON，以及从main带入的拖曳训练/工具/数据和文档归档。
- 按目标子树统计：real121、docs/deployment111、docs/towing106、ros2_ws105、docs/amp69、RL脚本44、RL datasets42等。这是跨目录历史和开发积累，不只是四个最近提交。
- [逐文件状态TSV](git_branch_audit_2026-10-21_files.tsv)、[机器可读快照](git_branch_audit_2026-10-21.json)保存完整列表/引用。

本轮只比较Git历史/树，没有重新验证707个文件的训练/硬件行为。build/log等忽略产物即使本地存在，也不一定被提交；工作区干净不代表忽略文件也已上云。

## 4. 为什么普通push不行、后续如何处理

远端1c6f6c9不是当前HEAD祖先，普通push要求的fast-forward不成立。Git保护的是提交图，不识别“都是我本人提交”作为跳过保护的理由。

当前证据符合“个人分支已rebase，本地应作为新历史”的情况。若用户确认以本地为准，适合先保存远端旧历史备份，再用**force-with-lease**更新同名远端；应核对期望远端仍为本次1c6f6c9，若已变化重新检查。普通`--force`不提供该并发保护，本轮不执行这两种推送。

如果需要保留两条原始提交图，可选择merge远端分支；但会重新引入旧历史，README/已迁移路径可能冲突，不必为了“落后6”盲目pull或再次rebase。何种同步方式须由用户决定，当前检查不授权改写远端或合并代码。

## 5. 已完成与待处理

已完成：真实fetch、干净工作区确认、分支计数/共同祖先、reflog与旧新补丁/非README blob核验、合并入口保留检查、707变更清单与ignored tracked检查；文档git diff --check通过。

待处理：尚未同步远端real、未决定是否纳入最新main的10个提交；本轮记录尚未提交。用户本地新修改已在fd9e30e等提交中，不需要再保存成另一次业务提交才能解释分叉。没有对任何代码作合并、覆盖或推送。


## 6. 2026-10-22再次fetch复核

用户再次要求检查。实际执行git fetch origin --prune成功；远端real仍为1c6f6c9，本地仍fd9e30e，ahead62/behind6、共同祖先ddf118e均未变化。reflog再次确认10月14日rebase onto origin/main的4d66f2c。重新按旧提交修改的非README路径比较对应新提交blob：45d2f8e→85f64ce的12项、11a4fce→2ae4c12的2项、c642c73→72617a7的1项、287c196→8c44e1c的16项、1c6f6c9→2ddc671的1项均相同。git cherry再次确认c642c73与1c6f6c9有等价补丁。

本次检查时工作区有2个已跟踪修改、21个未跟踪文件，无暂存改动：主要为此前排查/曲线记录、归档数据和诊断采集/绘图脚本。与上次“检查开始干净”的时间点不同，不将上次状态套到现在。忽略的build/log不包含在这些统计里。git ls-files -i -c --exclude-standard仍为空。

已完成：更新远端引用、再次验证提交身份改写和主要内容保留，确认当前未提交内容范围，diff检查通过。待处理：未提交诊断产物、远端real同步方式、是否纳入main更新。没有pull/merge/rebase/reset/commit/push；未修改控制代码。用户“基于远端继续开发”与本地历史后来rebase并不矛盾：远端保留旧身份，本地重放后的父链已改变。


## 7. 2026-10-22用户选择逐项审查merge：隔离预演结果

本机Git2.34.1不支持新版merge-tree --write-tree。使用/tmp下共享对象的独立clone，checkout已提交fd9e30e，在临时副本执行merge --no-commit --no-ff 1c6f6c9。未在用户仓库启动merge、切分支、stage、提交或push；原仓库未提交内容不包含在预演中。远端引用沿用本轮此前fetch的1c6f6c9。

实际冲突5项：README.md；imgo2_description/urdf/imgo2_real.gazebo.urdf；imgo2_description/urdf/imgo2_real.urdf；imgo2_description/xacro/core.xacro；imgo2_rl/source/imgo2_rl/imgo2_rl/assets/imgo2.py。前两份real URDF是add/add，README/core/assets是内容冲突。core有ixz正负号差异；assets有默认motion_new与motion路径差异。不能自动整文件选远端，也不能只按文本是否冲突判断物理配置正确。

自动合并但必须审查的项：旧路径docs/{amp_viewing_2026-09-24,git_fetch_proxy_2026-10-07,isaaclab_basics_2026-09-24,rebase_main_2026-10-07}.md共4篇重新出现；AMP policy从当前SHA256 cba59d44e387834ea11983b7489f4fd0351ddba8e8eba3c41e6713e33b520c54变成21e7ae53091f83ffc62eb465fd2db413acd3e063db51353db4d71b81fee77a29；package.xml从软链变为内容仅package.ros2.xml的普通文件。后一结果不能作为有效XML包描述使用。Git只做三方内容合并，无法理解这些历史重放与迁移意图。

已完成：独立副本真实merge预演、未合并索引清单、自动变更diff、策略哈希和类型变更检查，见[JSON](git_merge_preview_2026-10-22.json)。原仓库HEAD/暂存区/合并状态不变；仅按维护约定写入记录。

待处理：先审查并保存当前未提交产物，再在正式审查分支merge --no-commit --no-ff；人工决定5项冲突与自动变更的取舍，模型生成物与源一致性、资源路径、策略来源及ROS包软链验证后才提交合并。不能把临时预演当正式合并/部署验证。合并提交将同时保留两条历史，正常push可用；审查分支完成后原real可ff-only跟进，不需要force。若中止正式未提交merge可merge --abort，但开始前必须先保存工作区。


## 8. 2026-10-22正式审查分支continue受阻的只读检查

当前在xiaoji/merge_real_review，实际merge尚未完成。Git索引仍有3项未解决：README.md、real.gazebo.urdf、real.urdf。工作文件README与real.gazebo.urdf仍有真实<<<<<<< /======= />>>>>>>标记；real.urdf未发现标记且XML可解析，但尚未git add标记解决。

已暂存的core.xacro与assets/imgo2.py也不能视为合并正确：core新增6处inertial重复origin/mass/inertia；assets出现重复关键字及相邻重复gains块，当前文本存在Python语法问题。没有自行清除重复或还原文件。前述AMP策略替换、package.xml软链转普通文件及旧文档复现仍在暂存区，继续需审查。

已完成：status、ls-files -u、工作区/暂存区diff检查、XML解析与重复节点检查。待修：清理残留标记、确认完整current取舍、去除重复并验证，然后显式stage解决文件，最后审查自动合并项。未执行add/restore/commit/continue/push；用户正在手工处理冲突，未编辑仍冲突中的根README，维护入口更新待冲突解决后补齐。本节仅新增审查记录，不自动暂存。


### 第8节后续复查

用户再次询问状态。当前HEAD为438b845，仍在merge_real_review合并中。README已无标记且已暂存；两份real URDF工作文件均无标记、XML可解析、未发现重复inertial子节点，但索引仍未解决，需要确认内容后git add。core的6处inertial重复仍在；assets compile(...,'exec')静态语法检查失败于第68行；package.xml仍是普通文件且XML解析失败。AMP仍为21e7ae...，不同于当前HEAD中的策略，旧4篇文档仍待审查。不能continue/提交为合并正确。仅更新本记录，未改模型/代码、未暂存/提交；根README正在用户合并审查中，避免插入额外工作区改动。


## 9. 正式merge已提交，但文件正确性复查未通过

用户询问能否继续。当前merge_real_review HEAD=dddf956，父节点438b845与1c6f6c9，合并操作已经完成，无unmerged paths，不需要merge --continue。原real仍在438b845。工作区原有本记录未暂存修改。

只读复查：5个原冲突文件均无标记，两份real URDF XML解析通过；但core.xacro仍有6处重复inertial子项，assets/imgo2.py第68行附近compile语法失败，package.xml仍非软链且XML解析失败。三项问题已包含在dddf956合并提交中。旧AMP策略21e7ae...及旧4篇文档也仍在合并结果。git diff --check通过只表示当前差异无该类格式问题，不表示已提交代码正确。

已完成：确认合并提交和父链、残留标记/语法/XML/策略哈希复查。待修：上述三项内容错误、策略/文档取舍；修复后追加普通提交并验证，再让原real快进与正常push。未修改代码/配置/模型，未暂存或提交/推送；本轮仅同步维护状态。


## 10. 恢复操作后检查通过，等待修复提交与同步

用户已按步骤恢复。当前仍在merge_real_review，HEAD dddf956；8项预期恢复已暂存。core/assets/package.xml/AMP与合并前438b845逐字节或软链目标一致；core及两份real URDF XML解析通过且无重复inertial子节点；assets compile语法通过；package.xml已恢复软链且目标XML有效；AMP回到cba59d44...；旧4篇文档副本不存在，迁移后文档仍在各主题目录。无未解决冲突，工作区/暂存区diff格式检查通过，tracked ignored为空。

将暂存树与438b845比较，除README新增2行导航外，其余文件一致；说明本次恢复没有额外改动部署代码。该导航仍指向已删除的docs/isaaclab_basics_2026-09-24.md，已在README工作文件修正为已存在的docs/training/isaaclab_basics_2026-09-24.md，并更新维护状态，未自动暂存。

已修复且完成对应静态验证：合并引入的重复代码/模型子项、包描述类型、策略回退和旧文档复现。待完成：将README及本记录的未暂存更新一并提交修复，再让原real快进及普通push；未执行上述Git写操作。恢复只回到此前代码，未重新构建、训练、运行仿真或硬件，不宣称既有部署振荡问题已解决。
