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

## 后续95fab8d再次冲突

同日用户继续rebase后，应用95fab8d（新专家数据）再次发生冲突，涉及README、amp_env_cfg.py、test_amp_recipe_contract.py三个文件。该提交的六项新数据实验关闭全部滞空项，与上一轮选择保留缺口的七项配方冲突；沿用用户此前选择保留deficit，保留本提交其余数据路径/速度范围更新。

配置四处冲突保留缺口及对应每步换算说明、数值不变；不能选择六项名单却继续动态创建并给deficit.weight赋值，否则它被筛成None后会报错。测试保留缺口检查，同时合入本提交不同dt的检查并更新活跃集合为七项；补回因六项测试不再用到而被提交删除的re/ET导入。README七处冲突按内容合并，保留双方导航、拖曳说明、新实验和维护记录；当前policy SHA确为bf792f98...，保留对应25000历史行，未替换策略。避免问题表AMP-11重名，本轮缺口项编号改为AMP-17。

验证：env_isaaclab Python完整19项AMP alignment/recipe测试通过；资产路径检查通过，新默认imgo2_rl/datasets/imgo2_motion_new存在21段600帧，外层副本也有21段600帧（此轮只查目录/帧数，未重新证明两份内容相同）。Python AST、三文件无冲突标记、README实际本地链接和相关diff格式检查通过。

已修复并离线验证：三文件内容冲突、七项测试与缺失导入、README新旧状态说明。待完成：三个冲突文件及本文git add后继续rebase；Git索引在本轮结束前仍UU，不能只看内容无标记就认定Git已解决。后续提交/最终模型数据版本、阈值与训练冒烟及实际步态仍待验证；本轮未操作索引、执行continue、训练或硬件。

### 编辑器不再显示冲突按钮的核对

用户保存后再次检查：README已暂存，配置与recipe测试文件仍在Git索引中有stage1/2/3条目（UU），但磁盘内容没有任何冲突标记。两份源码AST与相关diff格式检查通过，19项AMP alignment/recipe再次全部通过，不需要再改源码。无按钮是文件内容已经消除标记，与索引的未解决状态不同；git add这两个文件才会清除未合并的索引条目。本文当时还有未暂存更新，本轮补核对记录也需一起暂存；未执行Git写操作或continue。

## 后续64ac334的文档冲突

用户继续后已通过95fab8d，当前停在64ac334（go_imu udev及文档），只有README存在索引冲突，尚剩一个9f3228e模型/数据提交。README四处内容冲突已合并，保留双方维护历史、新服务器入口与部署/训练问题记录；AMP-16沿用缺口开启的七项状态，未恢复六项禁用说明。新4000导出条目的AMP-17与本轮新增缺口条目重名，后者改为AMP-DEFICIT-01，保留原历史编号条目。

本轮只改文档；README冲突标记、相关diff格式、本地链接和唯一问题编号核对完成，没有重跑训练/硬件或新增代码测试。已修复：README内容冲突与编号碰撞。待完成：README及本文暂存后continue；最后9f3228e可能再有冲突，最终模型/数据身份与短训仍待验证。本轮未操作索引或执行continue。

编号扫描另外发现既有AMP-09、TOW-03、TOW-05、CART-01重复；本轮只消除新合入的AMP-17碰撞，未据编号删除其它历史事实，既有重复编号留待文档整理。

## 最后9f3228e及最新策略算法身份

2026-10-14用户继续后停在最后9f3228e，README与amp_env_cfg再次冲突，rebase尚未完成。配置两处只涉及注释/空格，保留现有七项缺口配方；README五处合并双方档案/导航/维护记录，并用实际文件身份纠正旧24000/旧4000默认策略说明。保留历史性能记录，不将它们作为最新策略测量。

已逐字节确认：当前core、real纯/Gazebo URDF及amp_ppo、判别器、motion_loader、amp_on_policy_runner、AMP runner超参数文件均与原最新提交9f3228e一致。实际RL目录21份新数据与2026-10-13前置清单逐文件SHA一致，包含前后1.2、无1.0遗留。外层imgo2_dataset副本仍是此前旧内容，训练默认仅用RL新目录。

当前部署policy SHA256 `9c6bf22baa552f39bcf023a288b7fa50ed82e687e66d2ed9c68ece3877e6fc66`，与唯一当前本机最新run `2026-10-14_00-20-45_motion_new_model_new/exported/policy.pt`完全一致。该run的model_25000.pt内iter=25000，SHA `616e0884f6f01e370152facb7587dc5465d35827dd863aa5d79e3c2e44a21c95`；CPU torch按保存的actor参数构造前向，与部署JIT在seed42的128组45维输入上max diff=0、输出12维。此过程没有手工导出策略、替换policy或运行仿真。见 [身份证据](rebase_latest_policy_identity_2026-10-14.json)。

必须区分配置差异：最新25000保存的env.yaml只有六项基础奖励、没有deficit；目前合并后的Height额外开启deficit（0.20s、每步−3），是用户此前选择的七项微调配方。算法核心与最新提交一致，但未来训练配方不等同该25000的原配方；不会反向改变已经训练好的policy。新七项收益与阈值仍需短训/实际回放。不能用原六项新run的训练完成证明新增奖励已验证。

训练PD25/0.5（保存yaml及资产），部署24/0.6（用户现场值）仍有差异，未擅自改数值；用于动态对照时需记录/决定是否对齐。当前模型物理一致性/惯量和路径检查通过，两套AMP19项离线测试再次通过；最新策略Gazebo、部署自身libtorch加载和服务器USD缓存本轮未重测。

已修复并验证：最后两个文件的内容冲突、旧策略身份注释、当前导出流程路径、相互矛盾的命令说明；将导出示例改为实际存在的10-14 model_new run，历史审计文档保留。待完成：本轮所有修改/新身份JSON暂存并continue结束rebase，七项训练gate、PD动态对照、外层旧数据副本与全仓MODEL-05/06、既有重复问题编号。未操作Git索引或continue，未启动长期训练/硬件。
