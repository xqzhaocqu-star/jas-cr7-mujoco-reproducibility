# 最新稿理论与现有正式仿真的算法核对

审查日期：2026-10-18。只读检查论文、控制实现、冻结协议及必要的已存数据；没有改写控制器、模型、论文或轨迹，没有重新运行物理仿真。

## 结论

`results_comparison` 中 **proposed / Full JAS** 所执行的控制与复合学习公式，与本轮用户提供的最新理论一致。本轮修改没有改变 BF、C1 终端滑模面、完整控制律、完整 `−YS+γε` 学习方向、常正对角增益、原始 IE 判据或投影规则，**没有发现因这次理论修改而必须重新运行 Full JAS 的算法差异**。

新增的内力解释、残差相关参数误差指数上界、精确滑模后的盒约束最小二乘极限，属于分析补充。现有仿真可以继续用于报告原方法的数值行为，但不能因此升级为验证全部新增理论结论：数值闭链不是严格刚性等式约束，控制与学习为采样实现，现有机器人数据没有建立精确 `S=0`。其中精确滑模后的指数极限结果尤其不能直接套用于这些曲线。

用户输入的旧仿真文字仍包含二维约化系统、平滑切换、故障和噪声等内容；它与现有双 CR7、原 `sign`、44 组正式实验不符。这需要整体替换仿真叙述，并同步摘要和结论；不意味着已运行的原 JAS 数据因论文文本 SHA 变化而失效。

## 审查版本及证据

以下论文行号均指用户输入的冻结副本：

- `simulation/manuscript_revision/inputs_20261018/bare_jrnl_new_sample4.tex`，SHA256 `2c1c0dfd9a3c0b4372ced31cb1acbe5379447066b0cc2d56d4baa9b94147c7c1`。
- 正式轨迹的 `source_manifest.json` 记录论文 SHA256 为 `2b8d43d6ac154b0718598e9b6db0299767954e73557c6b3ceacc96b9890e30e6`，与现存 `bare_jrnl_new_sample4_before_sim5.tex` 精确匹配。因此本轮可以直接比较真实运行版本与用户修改版本，未用无关旧备份推断变更。
- `jas`、`original_retrial`、`comparison` 中正式 manifest 记录的 15 个 Python 模块，当前 SHA 均与正式 manifest 一致。
- 对两个论文版本的带标签理论公式环境去除空白和注释后比较。下表列出的核心操作公式均完全相同；内力分解相关公式主要是等价改写；新增标签为 `giinternal`、`iforceerr`、`parambound`、`boxls`、`projectedlimit`、`boxlsbias`。

保留原始 manifest 才能准确记录运行来源；当前稿理论的语义兼容性由本审查说明，不应回写轨迹 manifest 来制造与新稿字节一致的历史。

## 逐式对应

| 理论、冻结输入行号 | 实际执行位置 | 结论 |
|---|---|---|
| `yundong`，315；Assumptions 1–3，321–331 | `original_retrial/controller.py:45`；`jas/geometry.py` | 用本臂编码器重建对象位姿及 `xd=J@qd`。每臂六关节、任务六维，`N_i=k_o`，不需要额外零空间控制。非奇异配置与 Euler 坐标有效域需由实验检查，不能由跟踪性能推出。 |
| `taui` 及其 `D,Q,W` 定义，404–425；Property 6，659–671 | `jas/dynamics.py:20`、`:64`、`:74`；`original_retrial/controller.py:61` | 六个连杆密度比例、一个黏性摩擦比例、一个对象密度比例构成八参数精确线性族。COM 与归一化惯量形状为已知；不代表任意 COM/任意独立惯量未知。双方负载份额固定为 1/2。 |
| `errortransfor`、`barrier`、`a`、`b`，500、532、562–563 | `jas/barrier.py:24` | 使用 `ell*tanh(chi*e)`、原有双因子分母及其解析一阶/二阶链式导数，没有误差裁剪或替换 BF。 |
| BF 的 C2、正性、`mu(0)=ell` 条件，535 | `jas/reference.py:38` | Full 使用原非对称包络；非单调工况的 C2 bump 位于 3–7 s。初始物理界为无穷，符合原 BF 的任意有限初始误差机制。另设的共同有限评估界不反馈给 Full。 |
| `dotq`、`ddotq`、`xinxitong`，582–599 | `original_retrial/controller.py:53` | `A=J^{-1}diag(a^{-1})`；`Adot=−J^{-1}Jdot A−A diag(adot/a)`，保留 `Adot`、参考及包络的导数项。此处数组变量 `a` 实际表示矩阵 `A`，而 `bs.a` 表示 BF 对角系数；只是代码命名差别。 |
| `huamomian`、`huamohanshu`、`xir`，734–759 | `jas/barrier.py:50`；`original_retrial/controller.py:51` | C1 内支和外支在 `±delta` 的值与导数匹配；达到滑模面后也不更换函数。标量 `lam1=lam2=.8` 对应合法特例 `Lambda1=Lambda2=.8I`。 |
| 参考回归器 `wuchafangcheng2`，780–785 | `original_retrial/controller.py:59`–`:64` | `D,Q,W` 的动力学系数用实际 `q,qd` 与对象 `x,xd`，再将参考 `qr,qrr` 代入向量乘积；没有用参考速度替换 Coriolis 系数中的实际速度。 |
| `kongzhiqi` 及展开式，788–807 | `comparison/controller.py:106`；`original_retrial/controller.py:65`–`:71` | proposed 直接进入 `OriginalController.command`。完整包含 `Y^T theta_hat`、`Ka+rho I`、两幂次到达项、`Y^T R sign(YS)` 和 `−J^T fd`。使用原 `sign`，未使用可选 boundary-layer 模式。 |
| `hatrho`，647–650；`priorbox`，677–701 | `original_retrial/controller.py:66`–`:67`；`jas/controller.py:53` | `rho=1e−8*||A||_2^2>0`；固定份额使 `L=0`，正的该上界合法。先验 `[.85,1.15]` 给 `r=.15`，鲁棒系数 `R=.30 I=2r I`，未缩小先验鲁棒项。 |
| `fuhetao`、`lvbohou`，834–876 | `jas/controller.py:101`；`jas/learning.py:27` | 以 `A^T(t)(tau_applied+J^T(t)fd)` 为已知信号，无反力反馈、真参数或实测加速度输入。保留零状态滤波和已知初值。 |
| `fuhewuchaget`、`fuhewucha`、`theta1`、`epsilon`，880–928 | `jas/learning.py:48`–`:64` | 原始 `YfYf^T` 和 `Yf*tauf` 积分；首个 `t>=window` 且原始最小特征值达 `omega` 的本地时刻冻结，无对角正则或伪造 IE。 |
| `gengxin`、`pr`，937–958 | `original_retrial/controller.py:35`；`jas/learning.py:64`–`:71` | 正式协议选择 `explicit_euler`，进入原 `CompositeLearner`；完整保留 `−Y S+gamma epsilon`，乘固定正对角 `Upsilon`，阻断盒面向外方向，并对有限 Euler 步作数值裁投影。没有换成历史项单通道或预条件逆 Gram 学习。 |
| `fFi`，375–381；`fuhetao` 的局部残差，834–863 | `run_comparison.py:124`–`:131`，`:208`–`:217` | 交互力先从约束广义力映射到对象坐标，再用 `Fcei−.5*sum(Fcej)` 计算内力。反力仅在控制/学习结束后的评估块读取；真参数在整段 rollout 结束后读取。 |

正式配置来自 `results_comparison/protocol.json`，而非各基类的默认值。尤其 `OriginalSettings.learning_integrator` 的类默认值是 `projected_implicit`，但本批正式协议明确覆盖为 `explicit_euler`，且比较控制器入口拒绝其他值。不能因看见类默认值就把本批结果描述为隐式学习更新。

正式学习参数为 `filter_rate=16`、`window=6 s`、原始 `omega=.005`、`gamma=1`、固定正对角 `Upsilon`（完整八维值保存在 protocol），初始估计为 `.9`。控制参数为 `alpha=.5,beta=1.5,delta=.0005,upsilon1=1.5,upsilon2=.5`；它们满足 Theorem 2 的成对幂次特例 `varphi=2,psi=4`。`Ka,K2,K3` 都是正对角矩阵，符合最新稿 807 行的条件。

## 本轮理论变更的实质与适用性

| 最新变更 | 与旧版、实际实现的关系 | 是否要求更换算法或重跑 |
|---|---|---|
| 内力分解 `del/Fcei/fFi/wutifenpei`，354–400 | 新稿把 `sum fMi=sum Fcei` 写明，并由此写 `fIi=Fcei−theta_i sum Fcej`。旧版用 `sum fMj` 的形式在理想模型下等价；当前诊断正采用新稿的直接形式。负载份额只定义分解，不等于强制实际受力分配。 | 否；仿真须报告实际受力误差，不能把 ±5 N 前馈直接称为达到目标。 |
| 常参数回归与时变份额的澄清，666–671 | 已知份额是回归系数；未知物理参数保持常数。本批固定 1/2，符合此条件。当前八参数的对象参数是对象整体密度比例，1/2 是已知系数。 | 否。本批没有验证时变负载分配；将来改变份额需同步推导其 `L` 项及动量滤波余项。 |
| 共同 BF/滑模参数、共同 `S,a`，751–756、1101–1102 | 当前两臂使用同参考、同 BF/TSM 参数。新稿将原先证明中使用的共同量前提显式化。理想刚性抓持下成立；数值 weld 只有近似成立。 | 不因该文字澄清必须重跑；须说明数值闭链与理想共同 `S` 的差别。 |
| 局部残差不因合力平衡消失，850–863、931–933；Remark 5，1015–1034 | 当前采用两个独立本地学习器，未把双方回归相加消残差。非零历史残差在冻结后继续存在，正是原方法的模型内容。 | 否；不能以 IE 阳性宣称真值参数收敛，也不能把 `G^{-1}p` 一概称为当前投影闭环极限。 |
| 内力恒等式 `giinternal/iforceerr` 与前馈对应解，1036–1093 | 从已有 `dsdot` 与共同运动量消元得到，无新增在线补偿/反力反馈。说明内力误差同时依赖动力学、估计及反馈项。 | 否；未做改变前馈目标的配对实验，不应称验证了精确的前馈不变性。 |
| 显式参数误差包络 `parambound`，1300–1309 | 来自已有 `Vdot<=−varpi V+Czeta`，是原 Lyapunov 估计的展开。包络允许残差相关半径，且不能确定极限点。 | 否；有界曲线与非零最终误差只说明相容，不能单独证明非平凡指数包络。 |
| `boxls/projectedlimit/boxlsbias`，1484–1566 | 在精确滑模 `S=0` 且历史冻结后，原学习律才化为带固定对角度量的投影梯度流。新结论到盒约束最小二乘点，通常有偏。 | 不改变 Full 算法；现有机器人数据不能直接验证该条件性新结论。若要专门实证此结论，需另行构造满足其前提的实验，不能事后强设 `S=0`。 |

对新增参数极限推导作了代数核对：以盒内最小点为比较点，投影有 `e_star^T(Pr(mu)−mu)<=0`；加最小点 VI 后得到 `V_star_dot<=−2*gamma*omega*lambda_min(Upsilon)*V_star`，因此加权范数指数率为 `sigma=gamma*omega*lambda_min(Upsilon)`。`boxlsbias` 的 `||zeta||/omega` 来自真参数也在盒中的 VI，不是宣称一般残差为零。这些条件性推导本身不要求删除原 `−YS`；该项是在 `S=0` 的前提下自然为零。

## 数值实现与理想模型的边界

1. **有限刚度闭链。** Theorem 1/2 与新增 `iforceerr/projectedlimit` 都针对 Assumption 2 的精确共同对象运动。MuJoCo weld 有小的位姿/速度残差，本地重建量与真对象量并非逐位一致；因此 `sum(S_i/a_i)^T(fIi−fdi)` 不保证精确抵消。动力学回归的代数形式相符，不等于有限 weld 的物理轨迹严格满足理想约束恒等式。
2. **连续公式的采样实现。** 物理 RK4 步长 `.000125 s`、控制与学习 `.0005 s`，电机命令在控制周期内保持。命令使用本 tick 学习更新前的 `theta_control`；随后用当前 `YS/G/p` 推进一次 Euler 学习，后验 `theta` 从下一 tick 进入命令。它是一阶交错离散实现，不是连续自适应解或同一时刻的隐式联立求解。数据同时存储两种 theta，不能用后验字段重构先前已施加的扭矩。
3. **因果滤波。** `update` 用上一周期保持扭矩形成当前区间左极限，以新命令保存下一区间右极限。这样不会跨反馈跳变作线性插值。随状态变化的 `A,J` 在滤波采样点计算，滤波源在样本间线性近似；仍有有限采样积分误差。
4. **动量积分分部。** 本批 `L=0`。每个参数基有 `m_k=A^T D_k qdot`、`r_k=A^T(W_k−Q_k^T qdot)−Adot^T D_k qdot`，使实际轨迹回归行为 `dm_k/dt+r_k`。代码实现 `Yf_k=jmath*(m_k−exp(−jmath*t)m_k(0))+filter(r_k−jmath*m_k)`，没有遗漏 `Adot` 或初值项。该等价式没有把反力补偿喂回学习。
5. **配置导数与滑模求解。** `M` 的配置导数及 `Jdot` 用小配置扰动差分，Coriolis 基随后使用实际速度。点值 `np.sign` 加保持输入/RK4 不构成 Filippov 等价控制求解器；即使选 `paper_sign` 也不能据此声称数值轨迹精确停留在滑模面。
6. **共同评估界不是改动 Full 原 BF。** `comparison/controller.py:28` 定义前 1 s 的有限 C2 共同评估界，用于让 log BF 与其他方法在共同物理域接受评估；Full 内部仍用 `mu(0)=ell`。从 1 s 起两者相同。前 1 s 的共同有限界是额外数值检查，不是 Full 定理直接给出的任意初值包络。
7. **有限物理可行域。** 非奇异、关节限位、Euler chart、无接触以及执行器限幅是否激活都须按实际记录分别报告。有限初值扫描不覆盖任意机械初始配置；浮点 `tanh` 也不能实现所有任意大的有限误差。

## 对本批结果允许作出的结论

本轮只读再次取 main 三工况的 Full 原始数据和 summary，确认两臂首次 IE 均为 6 s，全部记录的参数估计在 `[.85,1.15]`。最终真值误差 L2 范数依次为：

| 工况 | 左臂 | 右臂 | 22–24 s 的 `max(abs(S))`，两臂全部坐标 |
|---|---:|---:|---:|
| nominal | 0.188578 | 0.260196 | 0.0208906 |
| nonmonotonic | 0.182785 | 0.258353 | 0.0212964 |
| large_initial_error | 0.234613 | 0.281373 | 0.0176195 |

这些数据支持“原 JAS 方法在本批有限采样任务中保持先验内的参数估计，并产生非零最终参数误差”的描述；不支持参数真值误差已指数归零，也不支持已验证精确滑模后的盒最小二乘指数极限。`S` 的诊断容差为 `.001`，上述末段最大值不是精确滑模证据。

原 `−YS` 在联合 Lyapunov 分析中用于消交叉项，是本批原复合学习律的一部分；不能把删去它视为修复最新论文所必需的步骤。当前主结果也没有采用先前工程扩展的测力补偿、预条件历史学习或平滑鲁棒切换。

六种比较方法中只有 `proposed` 是完整原 JAS；`traditional_bf/linear_surface/linear_reaching/no_history` 是预先定义的结构消融；`external_ppc` 是其自身控制架构。消融改动不能误写成 Full 的实现偏差，外部方法也不能被描述为共享 JAS BF/TSM 的学习律对照。

## 论文文字必须同步的范围

- 冻结输入摘要 56 行、旧仿真节 1621–2388 行及结论 2391 行仍有二维约化模型、七参数、平滑鲁棒项、执行器故障和噪声实验等描述；不能迁移为本批双 CR7 八参数、原 sign 比较的事实。
- 1564–1565 行的“smoothed-feedback simulations”交叉说明应改成与当前采样实验一致的范围说明：精确滑模后理论不直接由这些有限步长机器人结果验证。保留“对平滑替换不直接适用”的数学区分本身合法，但不应继续暗示本文下面实际运行了平滑 Full。
- 最新表述“internal-force feedforward”与实际 `−J^T fd` 一致。应报告目标 ±5 N 下的实际误差、波动和力矩，不能写为已证明内力误差收敛或所有受力分量均改善。
- 原始首次 IE、非零冻结残差、参数投影有界、固定时间机理标量实验和机器人有限初值扫描，应分开解释。44 组数值校验通过属于产物/实现核验，不替代连续模型定理的前提或证明。

仿真节更新可沿用现有冻结 raw、协议、验证和图表；保留版本来源说明。没有必要仅为了使论文文件哈希等于新文本而重新积分同一控制算法。
