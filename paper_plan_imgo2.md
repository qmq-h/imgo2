IsaacLab 接下来实施计划
阶段	IsaacLab 实现任务	验收标准	状态
P0	复用现有 locomotion	无负载下正常接收 \(v_x\) cmd	✅ 已完成
P1	建立 wheeled cart USD/Articulation	车体 + 4 passive wheels 可正常落地滚动	⬜
P2	标定 wheel resistance	给 cart 初速度后能自然减速停止	\(v_L(t)\)、停止距离合理
P3	实现 rope model	Go2 与 cart 之间只产生拉力	slack 时 \(T=0\)，taut 时 \(T>0\)
P4	联调 towing	Go2 能稳定把 cart 从静止拉起来	无爆炸、穿模、异常振荡
P5	实现测量量	记录 \(T,v_R,v_L,d,\theta_{\rm pitch}\) 等	可导出 CSV/plot
P6	Direct-command baseline	直接给 locomotion 阶跃启停 cmd	获得完整 start/stop transient
P7	参数扫描	mass × resistance × velocity	确认不同负载动态确实存在差异
P8	Fixed-ramp baseline	cmd 前加入固定加速度限制器	找不同工况下合理 \(a_{\max}\)
P9	验证研究假设	比较 Direct / 不同 Ramp	证明不存在统一最优 ramp
P10	上层环境	upper policy 输出 \(\dot v_{\rm ref}\)	frozen locomotion 正常执行
P11	训练 adaptive shaper	history/load interaction → \(\dot v_{\rm ref}\)	未知负载自适应启停
P12	完整实验	Direct / Ramp / Adaptive / Oracle	论文数据
现在最先做 P1：负载小车

第一版不要做真实万向脚轮。

IsaacLab 里建立：

$$ \text{cart body}+4\times\text{wheel} $$

每个 wheel：

$$ \text{RevoluteJoint} $$

不设置 actuator：

$$ \tau_{\rm drive}=0 $$

第一阶段甚至只允许小车沿 \(x\) 方向运动都可以，因为现在验证的是纵向拖曳瞬态，不是 trailer steering。

建议 USD 层级类似：

Cart
├── base_link
├── wheel_FL
│   └── revolute_joint_FL
├── wheel_FR
│   └── revolute_joint_FR
├── wheel_RL
│   └── revolute_joint_RL
├── wheel_RR
│   └── revolute_joint_RR
└── rope_attachment

其中 rope_attachment 非常重要，后面绳索力作用在这里。

P2：先不要接 Go2，单独验证小车

这是我最建议你做的一步。

给 cart：

$$ v_L(0)=0.5,\ 1.0\;m/s $$

然后完全不施加外力，让它自己滑。

记录：

$$ v_L(t) $$

和：

$$ x_L(t) $$

最终得到：

$$ D_{\rm coast} $$

如果出现：

$$ 1m/s\rightarrow5m+ $$

才停下来，说明 wheel resistance 太低。

如果：

$$ 1m/s\rightarrow0.1m $$

就停，说明阻力太大。

我们希望第一版大概进入：

$$ \boxed{0.5\sim2m} $$

这种有明显惯性但又不会无限滑行的范围。这个数字先作为工程调试目标，以后用真车实测替换，不要作为论文中的“真实工业脚轮标准值”。

轮轴可以先采用：

$$ \tau_{\rm resist} = -\tau_c\operatorname{sgn}(\omega)-b\omega $$

甚至第一版只：

$$ \tau=-b\omega $$

都可以。

这一阶段不要碰 RL。

P3：实现绳索

我建议第一版甚至不要做 rope USD / 多刚体绳索。

直接在 environment step 中计算虚拟 rope force。

机器人后挂点：

$$ p_R $$

cart 挂点：

$$ p_L $$

定义：

$$ d=\|p_L-p_R\| $$

绳长：

$$ L_0 $$

伸长：

$$ \delta=d-L_0 $$

然后：

$$ T= \begin{cases} 0,&\delta\leq0\\ \max(0,k\delta+c\dot d),&\delta>0 \end{cases} $$

方向：

$$ e=\frac{p_L-p_R}{d} $$

于是：

$$ F_R=Te $$ $$ F_L=-Te $$

分别施加到 Go2 rear attachment 和 cart attachment。

这样你马上就有：

$$ \boxed{T(t)} $$

而且自然支持：

$$ \text{slack}\leftrightarrow\text{taut} $$

第一版不要模拟一根会碰地、弯曲的真实 rope，那只会增加 PhysX 接触问题，对当前论文问题几乎没收益。

P4：接入你现有 locomotion

这一步你的代码结构应该尽量保持：

user cmd
   │
   ▼
existing locomotion policy
   │
   ▼
Go2
   │
 rope force
   │
   ▼
cart

先完全不改 locomotion。

例如：

$$ v_{\rm cmd}=0.5m/s $$

运行 5 秒。

检查：

$$ v_R\approx v_L $$

以及稳定拖曳阶段：

$$ T(t) $$

是否进入相对稳定区间。

然后：

$$ v_{\rm cmd}=1.0m/s $$

再试。

这里需要先回答一个非常现实的问题

你现有 locomotion 到底能拉多大的东西？

先扫：

$$ m_L= 5,\ 10,\ 15,\ 20,\ 25\ kg $$

不是为了论文结果，而是确定 working envelope。

可能最终发现：

$$ 5\sim15kg $$

比较合理，那后面训练就用这个范围，不需要强行做到 25 kg。

P5：现在就把 Logger 写好

这个最好不要等 RL 做完。

每个 timestep 至少记录：

$$ t $$ $$ v_{\rm user} $$ $$ v_{\rm ref} $$ $$ v_R $$ $$ v_L $$ $$ T $$ $$ d $$ $$ \theta_{\rm pitch} $$ $$ \omega_{\rm pitch} $$

以及：

$$ \text{foot slip} $$

后面最好再加：

$$ \tau_{\rm joint} $$

最终 CSV：

time
user_cmd
ref_cmd
robot_vx
load_vx
rope_tension
rope_distance
body_pitch
body_pitch_rate
foot_slip
...

这套 logger 后面直接就是论文画图的数据来源。

P6：第一组真正重要的实验——阶跃停止

先稳定拖：

$$ v_{\rm cmd}=0.8m/s $$

例如：

$$ t<5s:\quad v_{\rm cmd}=0.8 $$

突然：

$$ t\geq5s:\quad v_{\rm cmd}=0 $$

即：

$$ \boxed{0.8\rightarrow0} $$

不要加任何 smoothing。

然后画：

$$ v_R(t),\quad v_L(t),\quad T(t) $$

再画：

$$ \theta_{\rm pitch}(t) $$

重点观察 \(t=5s\) 后：

$$ T_{\rm steady}\rightarrow0 $$

到底有多快，以及机器人有没有明显瞬态响应。

这一步实际上是项目的第一个 Go/No-Go

如果 Direct Stop 下：

$$ T\rightarrow0 $$

但机器人：

pitch 几乎不变；
没有 slip；
locomotion 完全不受影响；
不同负载都差不多；

那么我们就要重新审视这个研究问题。

反过来，如果明显出现：

$$ m_L\uparrow \Rightarrow \text{transient response变化明显} $$

那课题就立住了。

P7：然后做参数扫描

暂时不要 RL。

先固定：

$$ v_0=\{0.4,0.7,1.0\}\ m/s $$

质量：

$$ m_L=\{5,10,15,20\}\ kg $$

轮阻：

$$ F_r=\{\text{Low, Medium, High}\} $$

于是：

$$ 3\times4\times3=36 $$

个工况。

对每个工况执行同样：

$$ v_0\rightarrow0 $$

获得：

$$ T_{\rm peak}, \quad \max|\dot T|, \quad \theta_{\rm peak}, \quad D_{\rm stop}, \quad t_{\rm settle} $$

这一阶段就能看出你的研究问题到底主要受：

$$ m_L $$

还是：

$$ F_r $$

还是：

$$ m_L+F_r $$

影响。

P8：Fixed Ramp 是训练 RL 前最后一道验证

加入非常简单的 command manager：

$$ v_{\rm ref}^{t+1} = v_{\rm ref}^{t} + \operatorname{clip} ( v_{\rm user}-v_{\rm ref}^{t}, -a_{\max}\Delta t, a_{\max}\Delta t ) $$

测试：

$$ a_{\max} = \{0.2,0.4,0.6,0.8,1.0\}\ m/s^2 $$

于是你可以得到：

$$ J(a_{\max};m_L,F_r) $$

真正需要看到的是：

$$ \boxed{ a_{\max}^{*} \text{随 load dynamics 改变} } $$

例如：

Load	合理减速度
轻载 + 高阻	0.8
中载 + 中阻	0.5
重载 + 低阻	0.3

具体数字现在未知，但如果实验呈现这种趋势，Adaptive Upper Policy 就有了充分理由。

如果最后发现：

$$ a_{\max}=0.5 $$

所有工况都很好，那其实没必要训练神经网络——这也是为什么我建议先做这一阶段。

最后才进入 Upper Policy

确认上述假设成立以后，再把：

user cmd
   │
   ▼
┌────────────────────┐
│ Adaptive Cmd Shaper│
│ history → a_ref    │
└─────────┬──────────┘
          ▼
        v_ref
          │
          ▼
   Frozen Locomotion

接进去。

你已有 locomotion：

$$ \boxed{\text{不更新参数}} $$

只训练上层：

$$ \pi_{\rm upper} $$

我建议第一版 action 就 1 维：

$$ \boxed{a_t^{upper}=\dot v_{\rm ref}} $$

不要让它直接输出关节动作。

这样整个问题会非常干净。

所以你现在实际 Todo 可以压缩成 7 项
 1. Cart USD：车体 + 4 passive wheels + attachment
 2. Cart 单体测试：初速度 → 自然停止，调 wheel resistance
 3. Rope force：实现 unilateral spring-damper
 4. Go2 + Cart：现有 locomotion 稳定拖动
 5. Logger：\(v_R,v_L,T,d,\theta_{\rm pitch}\)
 6. Direct Stop：不同 \(m_L,F_r,v_0\) 做阶跃停止
 7. Fixed Ramp Sweep：验证不同 load 是否需要不同 acceleration/deceleration profile

做到第 7 项先停一下。