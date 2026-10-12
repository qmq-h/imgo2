# AMP 新数据、速度范围与关闭滞空奖励复审

## 用户意图与发现

用户已将Height训练范围修改为vx±1 m/s、vy±.4 m/s、yaw±1 rad/s，并注释feet_air_time的赋值，要求检查剩余问题。复审以本地实际源码为准，保留用户速度值和格式变更。

发现并修正：

1. feet_air_time仍在Height保留名单。基类weight0、threshold.5、sensor body_names为空，注释赋值会保留该无效选择器。当前IsaacLab-rsl5 RewardManager._prepare_terms即使weight0也会解析SceneEntityCfg，存在空足端名称解析风险；旧command.md再覆盖weight100还会重新启用或报错。现在从保留名单移除该项，由实际筛选函数置None，删除失效常量/注释赋值块。Fanziqi继承筛选安全，仍只保留两个tracking项。物理滞空日志可以继续打印，它是诊断，不代表奖励启用。
2. 用户新数据实际位于仓库imgo2_dataset/datasets/imgo2_motion_new，而assets默认指向不存在的imgo2_rl/datasets/imgo2_motion_new。已改为_REPO_ROOT推导正确目录，不复制数据。IMGO2_AMP_MOTION_DIR仍有优先级，旧终端若设置旧目录应自行核对/清除。
3. command.md三条训练命令含env.rewards.feet_air_time.weight100，已移除并说明当前None禁用。保留agent.amp_task_reward_lerp0.7覆盖，以免擅自改变用户近期比例；明确源码默认仍0.3，想用默认须删除该CLI覆盖。

## 范围与新数据检查

Height实际范围正确，课程仍关闭；play固定vx1/其他0是独立评估覆盖。Fanziqi的RLAMP_COMMAND_RANGES当前仍vx[-1,2]、vy±.3、yaw±1.57，只影响Fanziqi，不覆盖Height。若“整体”指所有任务，尚未统一PPO/HIM-Loco/Fanziqi；本轮不擅自改变其它任务。

新数据21段12600帧，均600帧、FrameDuration.02 s，与Height策略50Hz匹配。全部61列非空有限值和归一化四元数通过，real URDF正确腿序的逐片段FK坐标RMSE均0；FK一致不等于接触真实或时间同步已经证明。根高度片段均值约.3073–.3185 m，当前高度目标.30 m略低，并非启动错误，本轮未调整。

实际运动均值：前进1.0标签约.984、后退1.0约.984 m/s；横向.4约.402；转向±1约±1 rad/s。旧数据近静止backward_0.3的新版本约.294 m/s，标签与运动更吻合。新范围基本符合单方向参考覆盖，但同时最大前进/横移/转向组合未必有对应专家动作，不称全部组合已覆盖。[新数据统计](amp_motion_new_summary_2026-10-12.json)、[real FK](amp_motion_new_fk_2026-10-12.json)。

## 验证

使用/usr/bin/python3：6项AMP recipe测试通过，执行实际筛选函数确认Height滞空None、六项任务奖励、三个步长tracking尺度不变，Fanziqi最后两项；check_asset_paths通过，实际正确目录21文件、real URDF及17 mesh引用存在；inspect_amp_collection_data/audit_amp_dataset运行通过；修改源码AST和command.md所有bash块语法通过；git diff --check通过。更新的测试取代已失效的“必须开启滞空”旧契约。

未启动Isaac Lab短训练或新sim2sim。因此配置阻断已修并离线验证，环境构建/4环境10轮/新数据训练效果待验证（AMP-16）。若未设置环境变量，默认会使用新数据；运行中进程不会自动重新加载。

## 建议与待确认

先在目标训练环境进行4环境10轮，检查打印的21个专家文件、新保存env.yaml范围及feet_air_time:null、agent.yaml混合系数。换数据后旧判别器和归一化统计与输入分布绑定，本轮不建议直接用普通--resume恢复旧AMP全部状态；需要新建运行，或专门设计并验证只迁移actor的流程。当前command.md续训段是兼容配置模板，已注明不能直接用于未核对的新数据。

已修复：默认目录错位、滞空残留配置、CLI重新启用冲突、旧测试契约；待确认：目标机器短训练、旧环境变量、实际选择0.3还是0.7、其它任务是否也要统一、参考数据真实接触/时间同步及新策略收益。未启动硬件、录制或长期训练。

## 用户随后调整数据位置

用户已将新数据放到imgo2_rl/datasets/imgo2_motion_new，并将assets默认路径改为_PROJECT_ROOT对应此目录；该目录现存在21文件，与外层副本逐字节一致。本轮服务器流程按这一实际配置编写，不恢复此前根目录默认。资源路径检查通过，短训练待服务器执行。见 [服务器流程](server_amp_train_2026-10-12.md)。
