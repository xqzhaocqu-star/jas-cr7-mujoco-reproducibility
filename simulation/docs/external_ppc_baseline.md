# 独立外部协作 PPC 基准

本实现对应 Verginis、Mastellaro、Dimarogonas 的 **Robust Cooperative Manipulation without Force/Torque Measurements: Control Design and Experiments**，arXiv:1710.11088 **v4**，第 IV-B 节式 (25)–(36)，以及式 (20) 的负载分配与 Remark 5 的已知内力任务项。原文：[HTML](https://arxiv.org/html/1710.11088v4)、[PDF](https://arxiv.org/pdf/1710.11088v4)、[版本记录](https://arxiv.org/abs/1710.11088)。已核对原 PDF 的负载分配矩阵、PPC 公式与内力说明。源码为 `simulation/comparison/external_ppc.py`。

这是完整外部控制器，独立计算运动学反馈、参考速度、协作负载分配与关节力矩。它不调用 JAS 的 BF、终端滑模面、到达律、回归矩阵或学习律。论文 IV-A 的自适应四元数协议不是本基准；不会将该节的期望速度前馈、动力学补偿或自适应项混入 IV-B。

## 原式与坐标实现

对象坐标取 $x=[p^T,\eta^T]^T$，$\eta=(\phi,\theta,\psi)$ 为项目既有的 `xyz` Euler 角；对象几何速度为 $v=[\dot p^T,\omega^T]^T$。定义 $v=B(\eta)\dot x$，因此原文的表示 Jacobian 是 $J_O=B^{-1}$。代码独立实现此解析映射。局部编码器给出 $x_i=\mathrm{FK}_i(q_i)$、解析 Jacobian $J_{x,i}$ 和 $\dot x_i=J_{x,i}\dot q_i$。两手各自计算，不共享在线状态。

各坐标的姿态和速度性能函数均采用原式正指数包络：

$$\rho_k(t)=(\rho_{k,0}-\rho_{k,\infty})e^{-l_k t}+\rho_{k,\infty},\qquad
\rho_{k,0}>\rho_{k,\infty}>0,\quad l_k>0.$$

式 (25)、(28)–(31) 实现为

$$e_s=x-x_d,\quad \xi_s=\rho_s^{-1}e_s,\quad
\varepsilon_{s,k}=\log\frac{1+\xi_{s,k}}{1-\xi_{s,k}},\quad
r_{s,k}=\frac{2}{1-\xi_{s,k}^2},\quad
v_r=-g_s B(\eta)\rho_s^{-1}r_s\varepsilon_s.$$

式 (31) **没有** $\dot x_d$ 前馈；代码虽为接口一致接收参考函数的三个返回值，但只使用位置/姿态参考。$v_r$ 是当前状态的代数函数，不是动态滤波器，不需要计算 $\dot v_r$，也不估计加速度。

式 (32)–(36) 实现为

$$e_v=v-v_r,\quad\xi_v=\rho_v^{-1}e_v,\quad
\varepsilon_{v,k}=\log\frac{1+\xi_{v,k}}{1-\xi_{v,k}},\quad
r_{v,k}=\frac{2}{1-\xi_{v,k}^2},\quad
u_i=-g_v J_{M_i}\rho_v^{-1}r_v\varepsilon_v.$$

$g_s,g_v$ 均为原式的正常数标量。默认在初始编码器样本根据原文 Step II-a 一次性取所有速度坐标的 $\rho_{v,k,0}=\|e_v(0)\|+\alpha$，之后保持该参数固定；也可在实验配置中预先指定两手共同采用的六维初值。后者是逐轴固定初始化，不能声称逐字采用 Step II-a 的共同 norm 加同一 $\alpha$。原文 V-A 的仿真本身采用按分量的 $|e_{v,k}(0)|$ 选初值，V-B 的实验也采用不同轴的初值 $10,15,7$；因此逐轴初始化采用同类原文性能函数，但其初始域条件仍必须独立验证。不得在接近边界时重置性能函数。姿态和速度变换均严格检查 $|\xi|<1$，越界则记录控制器域失败，代码不裁剪误差。

## 抓取几何、负载分配与力矩

令 $d_i=p_O-p_{E_i}$ 为世界坐标下从掌心到对象质心的已知向量，$S(d)$ 是叉乘矩阵。刚性抓取运动学为

$$v_{E_i}=J_{O_i}v_O,\qquad J_{O_i}=\begin{bmatrix}I&S(d_i)\\0&I\end{bmatrix}.$$

本双臂中心对称抓取的对象坐标偏移为 $(0,-0.20,0)$ 和 $(0,+0.20,0)$ m，控制器初始化检查这些几何量与每个局部运动学模型一致。当前对象旋转矩阵将固定对象偏移变换到世界坐标。对象/掌心姿态用旋转矩阵精确复合，再在同一 Euler 图内表示；这避免将有限三维刚体旋转简单当成 Euler 角逐项相加，是与项目抓取模型一致的坐标适配，不改变 PPC 反馈公式。

使用式 (20) 中的**设计常数** $m_i^*=m^*>0$、$J_i^*=j^*I>0$，不读取实际质量或惯量：

$$m_O^*=2m^*,\quad J_O^*=2j^*I-\sum_i m^*S(d_i)^2,\qquad
J_{M_i}=\begin{bmatrix}\tfrac12 I&m^*(J_O^*)^{-1}S(d_i)\\0&j^*(J_O^*)^{-1}\end{bmatrix}.$$

这里保留原文印刷的矩阵乘法顺序。对本中心对称双抓取与各向同性设计惯量，$(J_O^*)^{-1}$ 与 $S(d_i)$ 可交换，堆叠 $G=[J_{O_1}^T,J_{O_2}^T]^T$、$G_M^+=[J_{M_1}^T,J_{M_2}^T]^T$ 满足 $G^TG_M^+=I_6$。若推广到非对称抓取，应重新审查式 (20) 的条件，当前代码会拒绝非中心对称几何，而不是悄悄换另一种分配律。

掌心几何 Jacobian 为 $J_{E_i}=J_{O_i}BJ_{x,i}$，关节力矩为 $\tau_i=J_{E_i}^Tu_i$。CR7 每臂六个关节，无额外冗余关节零空间控制项。控制器不执行力矩限幅；共同仿真器按相同的物理电机上限记录施加力矩及饱和。

## 相同的 ±5 N 内力任务及其边界

为了给各方法同一个已知任务，使用原文 Remark 5 明确允许的额外项

$$u_{\mathrm{int}}=(I_{12}-G_M^+G^T)\hat f_{\mathrm{int,d}}.$$

这属于已知零空间力任务的适配，不能把原文未设置的 ±5 N 目标默认为其原始实验设置。项目的 `fd` 表示**环境对机器人**、在对象质心的 Euler 共轭广义力，左/右分别为 $(0,-5,0,0,0,0)$ 与其相反数。控制器需要添加的是相反符号的执行器掌心 wrench。对每手，先用

$$\hat f_{\mathrm{int,d},i}=J_{O_i}^{-T}B^{-T}(-f_{d,i})$$

转为世界坐标掌心 wrench，再按 Remark 5 投影，加入 $u_i$。该转换包含物体旋转后保持相同对象质心任务所需的力矩，不能直接把固定世界坐标的纯点力当作相同广义任务。转换前的两个对象质心 wrench 相加为零，因此该向量已经在 $\ker G^T$ 内，投影只带来浮点舍入差。独立检查量 `ppc_allocation_identity_error` 与 `ppc_internal_null_error` 可从 `command` 的 info 中取得。

PPC 没有测量内力的闭环调节器，也不补偿左右不同机器人自重所引起的反力分配。添加已知零空间 wrench 不意味着实际测得内力必然等于 ±5 N；跟踪误差、有限刚度焊接与动力学扰动造成的偏差应作为外部方法的结果如实报告，不能用实际反力反馈修正。

## 可用测量与适用条件

在线输入仅为本臂 $q_i,\dot q_i$、时间和共同的预设参考。局部运动学模型及双手的固定抓取几何在实验前已知；另一只手的在线姿态、速度、力矩、反力、真实惯性参数均不读取。构造函数的 `dynamics`、`payload_mass`、`payload_inertia` 为统一接口占位，控制器既不保存也不使用它们。对象状态由本臂刚性抓取运动学重建，而不是从 MuJoCo 真值读取。

原文保证依赖刚性抓取、各臂运动学非奇异、未知耦合惯量正定、所需光滑有界扰动/参考以及 Euler 图条件。代码检查 $\rho_{s,\theta}(0)+\bar\theta_d<\pi/2$，并检查当前参考不超出所配置的 $\bar\theta_d$。论文为连续时间反馈，未给物理执行器饱和后的域不变保证。项目的有限刚度 weld、采样保持、有限电机力矩和局部重建的不完全一致均须用轨迹数据检查；原定理不能自动覆盖这些数值条件。

## 性能边界与共同评价

外部方法内部使用自己的**对称、单调指数**姿态/速度边界，初始位置误差须在有限初始包络内。它没有 JAS 的全局无穷初始边界或非对称、非单调性能函数。共同实验中的非单调场景只是共同评价要求的变化；不能因此把 JAS 的包络偷偷代入外部控制器。

`info['upper']`、`info['lower']` 为其原生物理姿态边界 $\pm\rho_s$，`barrier_margin=1-|\xi_s|` 为其原生姿态裕量。共同评价器另按统一物理边界计算所有方法的实际对象误差/越界时间；报告需区分两者，尤其共同边界通过不等于原生 PPC 域通过，反之亦然。原生速度域也必须独立检查。

日志 `surface=e_v`、`xi=xi_s` 只是统一数组接口，其定义与 JAS 的 $S,\xi$ 不同，不应跨方法比较这两个数组的数值。`A`、`regressor`、`momentum`、`remainder` 以及全部参数/IE 学习诊断为不适用，记录 NaN；不能当作零误差或已收敛。

## API 与冻结配置

```python
controller = ExternalPPC(kinematics, dynamics, center, payload_mass,
                         payload_inertia, hand_index, settings,
                         nonmonotonic=False, reference=reference,
                         external_settings=config_dict)
torque, info = controller.command(t, q, qd)
controller.update(t, applied_torque, info)  # no-op
configuration = controller.configuration()
```

`external_settings` 允许 `gs`、`gv`、`rho_s_initial`、`rho_s_final`、`rho_s_rate`、`rho_v_initial`、`rho_v_final`、`rho_v_rate`、`velocity_initial_slack`、`desired_pitch_max`、`virtual_mass`、`virtual_inertia`、`grasp_positions_object`。六维包络参数也可传标量。未知键拒绝执行，避免静默漏掉调参项。`configuration()` 返回公开来源、假设、全部配置及初始化后固定的速度包络初值。实例没有 `learner`。

源码的启动默认值是 $g_s=0.01,g_v=10$，姿态初值 $(.08,.08,.08,.22,.20,.22)$，终值 $(.012,.010,.010,.045,.040,.035)$，速度终值 $(.4,.4,.4,.8,.8,.8)$，两类衰减率均为 $0.5$，$\alpha=1$，$m^*=j^*=1$。它们用于同一个名义场景上的有限调参，**正式值以共同实验冻结的 protocol 配置为准**，不是论文实验参数，也不宣称最优。不得按测试场景或未知真值在线调整。

info 同时提供可独立重建原公式的 `ppc_rho_s`、`ppc_rho_v`、`ppc_xi_s`、`ppc_xi_v`、`ppc_epsilon_s`、`ppc_epsilon_v`、`ppc_v_reference`；额外 info 包含对象几何速度、对象反馈 wrench、负载分配/抓取矩阵和掌心 wrench。初始名义/较大误差两手的只读名义 IK 检查中，分配恒等式误差小于 $7\times10^{-19}$，内力零空间误差小于 $7\times10^{-17}$；这些仅验证坐标和公式接线，不代替完整 MuJoCo 仿真及独立验证。

独立审核的 `simulation/tests/test_external_ppc.py` 共 46 项检查通过，包含与独立加权右逆核对式 (20)、$G^TG_M^+$、内力零空间、真实 MuJoCo 掌端几何 Jacobian、式 (25)–(36) 的完整力矩 oracle、内力适配 $\Delta\tau=-J_x^Tf_d$，以及禁止动力学、未知质量/惯量和参考导数读取的 poison 输入检查。测试结论验证实现一致性，不预先判定闭环性能。

只读名义初始构型的对象空间等效惯量对角近似为 $(34.0,26.7,19.1,2.92,0.0244,4.29)$，相同坐标下静态重力 wrench 近似为 $(4.03,9.42,210,0.117,0.505,16.9)$。这些数值由名义链模型和标称对象计算，未读取未知参数。它们说明本任务的 pitch 惯量远小于 roll/yaw，且 yaw 仍需持续重力补偿。小误差时速度反馈斜率为 $4g_v/\rho_{v,k}^2$；因此六个含不同物理单位的速度包络不能仅按同一数字选取。名义有界试跑可据此固定各轴包络，而不能通过加入未公开补偿项修复失败。调参次数、失败试跑和最终参数均由共同实验协议公开保存。
