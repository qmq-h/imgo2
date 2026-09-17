# Imgo2 工作区维护约定

## 项目文档

- 根目录 `README.md` 是项目总览、训练部署流程、已知问题和维护记录的统一入口。开始涉及项目代码或配置的工作前，先阅读它的相关章节。
- 修改项目结构、任务注册、运行命令、模型/数据来源、训练配置或部署接口时，在同一次工作中同步更新 `README.md` 对应内容。
- 实际修复问题或完成验证后，更新问题状态、验证依据、顶部核对日期和维护记录；保留简短历史，不把未运行的检查标为通过。
- **每次工作都要把「已修复」和「待修复／待确认」写进维护文档**，不要只留在对话或提交信息里：
  - 已修复的 → 写入 `README.md` 维护记录；若本轮是一次排查或复审，另在 `docs/` 下建对应记录（如 `code_review_YYYY-MM-DD.md`），写清现象、根因、修法、验证方式和限制。
  - 待修复、待决定的 → 写入 `README.md` 问题表，并在 `docs/` 记录里写清「缺什么才能完成」。
  - 单次工作的细节放 `docs/`，`README.md` 只留结论、状态与链接，不要两处各自展开。
- **代码改了不等于修好**：提交时逐条确认。只有经过编译、运行或测试验证的项才可以把状态改成已完成、或从问题表移除；只改了代码但没验证的，一律写成「已修，待验证」并保留条目，同时写明需要什么环境才能验证。
- 研究计划是建议和历史背景，不是已完成结果；结论优先依据源码、配置和真实运行产物。
- 本维护约定不要求为无关改动重跑训练或部署，也不授权额外启动长期训练、硬件运行或后台任务。

## 文档结构

- 仓库是**单一项目**，文档只放三处：根 `README.md`（总览、流程、问题表、维护记录）、`AGENTS.md`（维护约定）、`docs/`（单项排查记录与离线结果）。**不保留子项目 `README.md`**；「哪个目录属于哪条链路」由根 README 的目录职责表说明，不要再为子目录另建入口文档。
- 顶层目录命名统一为小写下划线：`imgo2_description/`、`imgo2_dataset/`、`imgo2_rl/`、`imgo2_deploy/`（`imgo2_model/` 已于 2026-09-17 并入 `imgo2_description/` 后删除）。新增顶层目录沿用该风格。
- 同一结论只在一处展开，避免两处各自漂移。
- 忽略规则只放在根 `.gitignore`。只对某个子树成立的规则必须带路径前缀（例如 `/imgo2_deploy/**/mujoco/`），否则会在整仓范围内过度匹配；改动后跑一遍 `git ls-files -i -c --exclude-standard`，它必须为空（没有「已跟踪却被忽略」的文件）。
- 模型与 URDF 目录不放临时笔记、质量草稿一类的文件；这类数字写进 `docs/` 或直接删除。
- 路径、安装命令、checkpoint 示例写进文档前先确认存在；个人训练机路径要标为示例，不要让人直接沿用。
- 内容已被取代或失真的文档应更新或删除，删除前确认没有其他文档引用它。

## 模型与资源路径

- **模型唯一源是 `imgo2_description/`**（2026-09-17 用户决定并已完成，README MODEL-02）。原先并存的 `imgo2_model/`、`imgo2_rl/source/.../data/`、`imgo2_deploy/robot_description/` 三份副本**已删除**（git 可追溯）；不要重新引入副本，需要改模型就改源再重新生成。
  - **分层**：`xacro/core.xacro` = 物理内核（17 link/16 joint，命名 `FL/FR/RL/RR`，关节 `FL_hip_joint`/`*_thigh_joint`/`*_shank_joint`，足端固定关节 `*_Ankle`，link `base`/`FL_HIP`/`FL_THIGH`/`FL_SHANK`/`FL_FOOT`）。Gazebo 专用内容全部独立成模块：`transmission.xacro`（仅 ros_control）、`gazebo.xacro`（插件 + per-link 接触参数）、`imu.xacro`（`base_imu`）。`xacro/robot.xacro` 是组装入口，开关 `transmission`/`gazebo`/`imu` 默认全 false，所以不带参数时只输出内核。
  - **生成物入库、勿手改**：`urdf/imgo2.urdf`（纯 URDF，RL 与非 Gazebo 消费者用）与 `urdf/imgo2.gazebo.urdf`（+transmission+gazebo+imu，Gazebo/ROS 用）。重新生成：`cd imgo2_description/xacro && xacro robot.xacro [transmission:=true gazebo:=true imu:=true] > ../urdf/<名字>`（ROS 2 Humble 的 `xacro` 即可，include 用相对路径）。**网格前缀由 `robot.xacro` 的 `mesh_prefix` 属性决定**：纯 URDF 用 `../meshes/`，`gazebo:=true` 时用 `package://imgo2_description/meshes/`（Gazebo 经 ament 解析，且 0.4.x 的 `gazebo_ros2_control` 不吃 `package://` 作为 `<parameters>`，那处由 launch 换成真实路径）。改完 `gazebo.xacro` 记得重生成这份 URDF；`check_model_sync.py` 的网格检查两种写法都认。
  - **本包同时是 ROS 2 包**：`package.ros1.xml`／`package.ros2.xml` 两份包描述，`build.sh` 按 `ROS_DISTRO` 生成 `package.xml` 软链（与 `src/` 下其它包同一套机制，包扫描用 `find -L`）；`imgo2_deploy/src/imgo2_description` 是指向仓库根的**软链**（模型唯一源仍在仓库根，不要改成复制）。Gazebo/ROS 2 链路要点见 [Gazebo 链路打通记录](docs/gazebo_ros2_bringup_2026-09-17.md)：`<parameters>` 必须真实路径、`robot_description` 传给 controller_manager 前必须压成单行。
  - **MuJoCo 模型**：`mjcf/imgo2.xml`（机器人本体）+ `mjcf/scene.xml`（世界，`<include file="imgo2.xml"/>`），由 `rl_sim_mujoco` 通过编译期 `IMGO2_MODEL_DIR` 读取。物理来自训练 URDF，求解器/接触按可用的参考部署对齐（`cone=elliptic impratio=100`、关节 `damping=1 armature=0.1`、碰撞 `condim=3 solref="0.005 1"` + 分几何 friction），并带一个 C++ 尚未读取的 `framelinvel`（adr 43）。
  - **物理参数的唯一准绳仍是训练侧那套值**（用户 2026-09-15 决定，现在是 `core.xacro`）：关节轴/限位、每个 link 的质量/质心/惯量、碰撞几何、base 规范值（质量 `5.53394020` 配惯量 `0.03866860/0.10411461/0.12554111`）。只改质量不改惯量会造成不自洽。
  - **不要改关节轴符号/限位**：AMP 恒等映射的依据是模型与录制的真机数据 FK 吻合到 0.002 m，改轴线会让吻合崩到 0.22 量级、整套对齐结论作废，而真机数据不会跟着变。
  - **改完必须跑** `python imgo2_rl/scripts/tools/check_model_sync.py`：按逻辑关节名与逻辑 link 名比对两份已登记 URDF（Gazebo 份允许多 `base_imu`）的关节轴/限位、逐 link 质量/质心/惯量/碰撞、base 规范值；校验网格目录指纹（`8dc5b5995a11`，10 文件）与每个 mesh 引用都存在；对每份做 FK 复现录制数据；并枚举全仓 URDF，未登记即报错。
  - 网格全仓唯一一套：`imgo2_description/meshes`（改名曾用 LF 命名，2026-09-17 统一为 FL 命名，内容未变）。
- 代码里的资源路径一律由 `Path(__file__)` 推导或走环境变量，不写机器绝对路径。RL 的模型路径经 `_REPO_ROOT`（`parents[5]`）指向 `imgo2_description/urdf/imgo2.urdf`；改动后跑 `python imgo2_rl/scripts/tools/check_asset_paths.py` 自检（它会从源码读出实际声明的路径，而不是复制一份预期值）。
- **部署用的 `policy.pt` 一律由对应算法的 `play.py` 导出（headless 模式），不要手写导出脚本、也不要手工拼 TorchScript**（用户 2026-09-18 决定）。入口：`imgo2_rl/scripts/rl_lab/{ppo,amp,himloco}/play.py`；它内部调用 `export_policy_as_jit(actor_critic, normalizer=None, …)`，产物在 checkpoint 同级的 `exported/`（`policy.pt` + `policy.onnx`），再拷进 `imgo2_deploy/policy/imgo2/<算法>/policy.pt`。前提是**配置的观测维数与 checkpoint 一致**（AMP 的 45 维 actor 已随 `f028802` 落地，见 [Gazebo 记录](docs/gazebo_ros2_bringup_2026-09-17.md) 9.5 节），否则 `runner.load()` 会在 `load_state_dict` 处尺寸不匹配而失败。导出后按三条契约复核：与 checkpoint 的 actor 在同一批确定性输入上 max diff = 0、能被部署自己的 libtorch 加载、部署 interface 的观测维数与顺序和训练侧一致。

## 删除与可复现性

- 仓库已可用（见「多机与同步」），删除内容可由 `git` 追溯，但仍要在维护记录里写明删了什么、为什么。删除前先确认没有其他文档或脚本引用它。
- 离线检查脚本放在 `imgo2_rl/scripts/tools/` 且只用标准库，使结论能在没有 Isaac Lab 的机器上复现；不要把验证只留在临时目录。
- 记录验证时写清所用解释器或环境，PATH 上的 `python` 不一定可用（本机该别名返回退出码 9009）。
- 提交长训练前先做前置检查并跑一次短训练；判别器和归一化统计与输入尺度绑定，改用新配置时应新建运行而不是续训。

## 多机与同步

- 仓库是单一 monorepo（根目录），远程为 `https://github.com/qmq-h/imgo2`（本机 `origin` 已改为 SSH：`git@github.com:qmq-h/imgo2.git`，2026-09-17），默认分支 `main`。训练服务器 clone 这一份即可拿到模型、训练代码、动作数据与文档。仓库曾用名 `imgo2_rl`（2026-09-15 改名，旧地址由 GitHub 重定向）。
- 本机 git 推送需要临时 `-c` 覆盖，不要擅自改用户的全局配置：
  - `~/.gitconfig` 的 `http.proxy`／`https.proxy` 指向 `127.0.0.1:10808`，**但该端口没有进程监听**；代理客户端实际监听 **7890**。且 `https.proxy` 被写成 `https://` 是错的，对 HTTPS 目标代理本身仍用 `http://`。
  - 系统 `gitconfig` 的 `http.sslBackend=schannel` 在受限 shell 下报 `SEC_E_NO_CREDENTIALS`，需改用 `openssl`。
  - 直连（`-c http.proxy=`）只够跑小请求：`ls-remote` 能成功，但批量 push 会被重置（`Recv failure: Connection was reset`）。
  - 因此实际可用命令是走 7890 代理：
    `git -c http.sslBackend=openssl -c http.proxy=http://127.0.0.1:7890 -c https.proxy=http://127.0.0.1:7890 push origin main`
- 受限 shell 下 git 的凭据助手无法创建命名管道（`Win32 error 5`），需要放宽沙箱才能推送；这是环境限制，不是仓库问题。
- **Linux 主机（`~/RL/imgo2` 那份）情况不同，别照抄上面的 Windows 命令**：环境变量 `http_proxy`／`https_proxy`／`ALL_PROXY` 指向 `127.0.0.1:7897`，该端口在监听但 TLS 握手失败（`unexpected eof while reading`）；`7890`/`10808` 无人监听；**直连可通**（`curl --noproxy '*' https://github.com` 返回 200）。因此推送要先清掉代理环境变量：
  `env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u ftp_proxy -u FTP_PROXY git -c http.proxy= -c https.proxy= push origin main`
  **推送凭据（2026-09-17 已解决）**：该机原先没有 credential helper、`~/.git-credentials`、`gh` 与 token 环境变量，`git push` 报 `could not read Username`（harness 的 shell 无 TTY，无法交互输入；注意 `user.name`/`user.email` 只是提交署名，不是认证凭据）。现已在 `~/.ssh/id_ed25519` 配置 SSH key 并加到 GitHub，`origin` 改为 `git@github.com:qmq-h/imgo2.git`，`git push origin main` 可直接使用（`ssh -T git@github.com` 返回 `Hi qmq-h!`）。GitHub 的 22 与 `ssh.github.com:443` 都通。`~/.ssh/config` 里的 `Host qmq`（183.147.142.40:30069）是另一台机器，与 GitHub 无关。
- 合并前的嵌套仓库历史保存在 `.git-backups/`（git bundle，已被忽略、不入库），恢复方法见该目录的 `README.md`。不要删除它。
- 换行由根 `.gitattributes` 固定为 LF 并把模型网格标记为二进制；本机 `core.autocrlf` 已设为 `false`。新增二进制类型时同步补进 `.gitattributes`，否则可能被当文本转换而损坏。
- harness 只在它所在 Host 的文件系统上读写，本身不做跨机同步。会话记录也留在 Host 侧，在云服务器上运行 harness 得到的是「操作服务器那份文件」的 agent，不会把本机的改动和上下文带过去。
- 因此跨机交流只有文件系统这一条通道：`AGENTS.md` 在会话开始时自动加载，用来传规则；根 `README.md` 与 `docs/` 用来传状态和依据。没有写进文件的信息不会跨机传递。
- 「第 8.3 节维护记录」就是事实上的消息日志：每次工作结束追加一行（日期／变更／验证与限制），接收方动手前先读它。已验证的版本用内容哈希标注，因为时间戳不可靠。
- 同一结论只写一处。两端并行维护同一份文档会漂移（DOC-01 即由此产生）；改动前先 `git pull`，避免覆盖对方的提交。
- 要明确哪一份为准：训练在哪台机器跑，就以那台为准，另一份当镜像。
- 把 Web GUI 暴露到网络必须谨慎：`dsh-host-webserver` 只支持 `127.0.0.1` 与 `0.0.0.0`，且自身不带 TLS、认证或源策略。跨机访问优先用 SSH 端口转发保持回环绑定。
