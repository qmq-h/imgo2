# 分支推送与部署目录建议（2026-10-20）

本轮按用户要求只检查并给命令/架构建议：没有执行add、commit、fetch、merge、rebase、push，没有移动代码、改构建或运行硬件。维护文档同步不代表目录方案已经实施。

用户随后已将目录移到real/ros2_ws，本文件中的目录方案为此前建议；实际路径修复和验证以 [迁移记录](../deployment/deploy_layout_migration_2026-10-20.md) 为准。Git数字是本文件检查时快照，不代表后续远端状态。

## 1. Git核对结果

- 工作区根是单一Git仓库；当前分支 `xiaoji/imgo2_real`，不是含空格的“xiaoji real”。
- 本地HEAD `457f98a`，说明为“sim2real测试基本成功,剩余电机控制频率问题与机器人低速抖动问题”。本轮最初检查时工作区仅README修改及三个新排查文件，暂存区为空；本记录随后新增。
- origin目前实际使用HTTPS：`https://github.com/qmq-h/imgo2`，不沿用AGENTS中的历史SSH状态。
- 本地跟踪引用 `origin/xiaoji/imgo2_real` 为 `1c6f6c9bc3ec281f4582a74cc39de035d60cab91`；已用只读ls-remote查询GitHub，远端该分支头相同，因此本次58 ahead/6 behind有远端头核对依据。
- 远端独有6提交：`1c6f6c9`、`97ce9c9`、`287c196`、`c642c73`、`11a4fce`、`45d2f8e`，涉及policy/文档/模型等；不能当作无关提交直接强推覆盖。尚未逐项完成内容合并/冲突审阅。
- 本机默认HTTPS请求失败，临时清空Git的http/https代理后只读查询成功；不改全局代理或remote。

用户希望将当前全部内容上传到同名远端分支时，建议先检查待提交文件、提交，再fetch并merge同名分支。以下是建议命令，未替用户执行：

```bash
cd /home/tommy0929/Documents/imgo2
git status --short --branch
git add -A
git commit -m "记录实机通信排查与部署架构建议"
git -c http.proxy= -c https.proxy= fetch origin
git merge origin/xiaoji/imgo2_real
git -c http.proxy= -c https.proxy= push origin HEAD:refs/heads/xiaoji/imgo2_real
```

逐条执行；merge冲突后不要继续push。处理冲突、审阅policy/模型/真机配置并完成merge后再推送。不提供强制覆盖作为默认方案；不擅自rebase用户历史。远端在操作间可能再次更新，push若再次拒绝应重新fetch核对。

## 2. 当前目录的真实职责

- `imgo2_deploy/`是部署工作区/容器目录，不是单一ROS包。
- `imgo2_deploy/src/imgo2_deploy/`是现有ROS1/ROS2兼容部署包；旁边还有消息、关节控制器和模型软链等包。
- 当前真机工程是 `imgo2_deploy/go_imu/`，已经没有上轮上传时的 `go_imu/go_imu/`双层；它有自己的CMakeLists、hardware、Motor、Imu、serialport、sdk、fsm、config和程序入口。旧记录/JSON中的嵌套路径属于当时快照，后续命令需用当前路径。
- 当前go_imu的CMake直接构建硬件、SDK、FSM和工具，已经提供`go_imu::hardware`/`go_imu::sdk`目标；不依赖ROS节点或executor。旧ROS构建脚本的`USE_CMAKE`路线指向`src/imgo2_deploy`，不能误认为它在编译新的go_imu。

## 3. 推荐：真机核心独立CMake，ROS按需在外层适配

把真机代码放在imgo2_deploy之下本身合理。混乱主要来自ROS工作区、独立工程、旧参考入口、运行产物和备份同时出现且入口相似，不是“没有ROS package”造成。

推荐日后整理成两条平级链路，示意如下，**尚未执行，也不要求立即改名**：

```text
imgo2_deploy/
├── real/                 # 现go_imu完整独立工程；名称待用户决定
│   ├── CMakeLists.txt
│   ├── hardware/         # 整机接口与适配
│   ├── Motor/ Imu/ serialport/
│   ├── sdk/ fsm/ config/ policy/
│   ├── apps/ display/ function/ tests/
│   └── 程序入口
├── ros_ws/               # 现ROS1/ROS2兼容工作区，含现src各包
│   └── src/
├── policy/               # 仿真/策略资源来源规则需随整理明确
└── scripts/              # 明确区分真机、仿真构建与运行
```

`real`名称只是建议；保留`go_imu`也可以。不要为整理目录同时改底层协议、FSM语义、主函数或可执行名称。迁移时必须逐项修正资源路径、脚本、测试、安装及文档并重新验证，不能只移动文件。

当前先冻结可工作的真机目录，优先解决通信时延。需要ROS话题、遥控或可视化时，再加薄的ROS适配包：复用硬件/SDK，向控制循环传线程安全命令、发布状态；电机事务/看门狗/FSM/推理继续在独立核心内运行，不放到话题往返链路。包装成ROS包本身不会消除USB约1.65ms，也不自动保证实时性。

后续整理待办：区分两个同名`rl_real_imgo2`入口的构建/运行命令；明确源policy与运行时副本的生成/发布关系，避免手改两套；区分源码与build/install/log、zip备份及库依赖。不要未经确认删除备份或已有成功产物。独立搬走使用仍需携带或定位LibTorch/yaml-cpp和运行配置，不因独立CMake就保证已完成安装搬迁验证。

## 4. 完成项与待确认

已完成：分支/远端头/工作区只读检查，当前真实目录与构建目标核对，给出不覆盖远端历史的命令及分层建议，根README纠正过时嵌套路径描述。

待处理：用户提交与合并远端6提交、可能冲突与push认证；目录命名/迁移范围需用户决定；若后续实施则补完整离线构建、资源路径、安装和回归验证。本轮无推送、无架构改造、无生产代码修复。

## 5. 迁移后再次只读核对Git状态（同日）

当前HEAD仍457f98a，分支xiaoji/imgo2_real；本地跟踪引用显示58 ahead/6 behind，本次未fetch或查询远端最新状态。暂存区为空；23项修改、221项旧路径删除、229个未跟踪文件。220项删除均在real/ros2_ws找到新位置，剩余为已集中根规则而删除的子.gitignore，没有发现额外丢失目的文件。6份新位置策略文件不被忽略，没有待提交的构建/下载二进制产物；tracked-but-ignored与diff空白检查均为空。

另有已跟踪生成软链log/latest_list改指向本轮colcon list的运行目录，应在提交前排除/确认；本次未恢复或暂存它。没有执行add/commit/merge/push，也没有修改源码。已完成检查与迁移文件对应确认；待用户审阅暂存，并处理分支分叉后推送。以上为检查时快照。
