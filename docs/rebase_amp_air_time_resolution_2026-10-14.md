# rebase中的滞空奖励接线冲突

日期：2026-10-14。范围：用户询问AMP任务保留名单里的feet_air_time_deficit与feet_air_time应选哪一个。本机也处于相同冲突状态；本轮只处理该配置接线，不完成整个rebase。

## 选择与依据

保留feet_air_time_deficit。当前分支已有函数、动态RewTerm创建、模块级threshold=0.20与每步系数−3以及配套数值测试，保留名单必须与这些接线对应。

- 原feet_air_time：落地时返回sum((air−threshold)×first_contact)，通常用正权重；超过阈值继续给正奖励。
- deficit：落地时返回sum(max(threshold−air,0)×first_contact)，用负权重；超过阈值代价为0。当前0.02s控制周期下raw weight=−3/0.02=−150。

对于用户希望限制滞空不足的目标，缺口版避免奖励无限延长滞空；它仍不是完整周期控制器。无落地事件同样可能为0，不能据此保证抬脚或0.6s周期。阈值应重新核对新专家数据，未用旧统计作为新数据的已验证结论。

## 修复

`base_move/amp_env_cfg.py` 保留deficit并删除该处冲突标记；同时删除旧feet_air_time.weight/threshold/sensor_cfg赋值及对应旧注释。否则过滤循环将feet_air_time置None后，后续访问None.weight会报AttributeError。原feet_air_time函数仍保留供其它任务使用，不同时启用两项。

`test_amp_alignment.py` 的已有任务尺度回归桩加入旧feet_air_time项并确认被过滤为None，防止重新合并时两项重复保留。

## 验证与状态

本机env_isaaclab Python运行4项既有配置/缺口测试，全部通过：三个step_dt下的每步系数、缺口接线/负权重、阈值上下的数值、零指令和直立门控。源码AST解析通过、该配置已无冲突标记。本轮未启动训练/仿真，不宣称慢步态收益已验证。

已修复：该配置文件内容冲突、旧奖励None赋值风险；离线回归通过。

待完成：git索引仍UU，因为未git add；README仍有其它冲突，整个rebase未结束。用户须核对并解决其它冲突，再暂存并继续rebase。正式训练前仍需运行构建/短训与新数据下阈值、接触/周期品质验证。本轮未暂存、提交或执行git rebase --continue。

## 复审当前文件

2026-10-14用户随后要求检查当前文件。用户在检查期间已暂存部分冲突解决；重读后索引无UU，但README还留有两个未闭合导航冲突标记。只清掉残留标记和重复介绍，保留现有三类连接内容与合入的代码入口表，没有用Git某侧整份覆盖README。按唯一现存同名文件修正10条移动后的文档链接；缺失research_exploration_plan没有猜测替代，改为缺失标注待确认。

发现合入test_amp_recipe_contract仍依赖旧feet_air_time正权重，AST执行缺口配置时因桩未提供RewTerm而NameError。改为注入缺口所需RewTerm/SceneEntityCfg/mdp与常量，检查旧项None、缺口负权重/实际足端匹配，并继续确认Fanziqi移除该项。原test_amp_alignment增加的旧项关闭断言保留。

修正amp_env_cfg里遗留的“滞空使用原始权重”说明：实际缺口与其它项均按每步系数/dt换算。仅修高度强弱误导注释，不改任何奖励数值。README当前奖励条目也改为缺口版，并避免与两套任务配置的AMP-09重复编号。

验证：本机env_isaaclab Python运行test_amp_alignment和test_amp_recipe_contract完整19项，全部通过，包括真实CPU torch数值/归一化更新检查；当前修改Python AST、git diff --check与cached --check通过，配置/README无冲突标记；资产路径检查通过。没有Isaac环境构造、GPU训练或Gazebo运行，离线通过不等于运行gate完成。

当前默认读取imgo2_rl/datasets/imgo2_motion，只有这个数据目录，已不是前轮600帧的imgo2_motion_new。抽查forward/backward_0.3为250帧、stance100帧；backward_0.3存储body_x均值约−0.029m/s。这是当前rebase中间状态的事实，不能据文件名确认已加载用户最新重采数据。本轮不移动/覆盖数据、不猜测后续提交；完整rebase后重新核对最终模型/data SHA、路径覆盖变量与最新采集身份，正式训练前再做短测。

已修复并离线验证：旧recipe测试不兼容、README残留标记与可定位的失效链接、当前奖励尺度说明。待确认：内容修改重新暂存、整次rebase继续/后续冲突、缺失研究文档、最终模型与新数据身份及训练/周期效果。尚未git add或执行rebase --continue；使用git add -A会包含用户其它改动，应按明确文件选择暂存。

### 暂存状态复核

用户反馈rebase --continue提示仍需git add。现场git ls-files -u和diff --diff-filter=U均为空，git status明确all conflicts fixed；但最新README、amp_env_cfg.py、test_amp_alignment.py、test_amp_recipe_contract.py四份修复未暂存，本文未跟踪。需要将这五个路径加入索引后再continue。当前已完成6条rebase命令，还有3条待应用，包括新增专家数据提交；不能把当前中间数据版本视为最终。此轮只核对并补记录，未修改Git索引或执行continue。
