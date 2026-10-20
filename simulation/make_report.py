#!/usr/bin/env python3
"""Write a Chinese experiment report from independently checked raw results."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CASES={'nominal':'普通空间轨迹','nonmonotonic':'非单调非对称约束',
       'large_initial_error':'增大初始误差'}


def sustained_entry(t,values,threshold):
    outside=np.flatnonzero(values>threshold)
    if not len(outside):
        return float(t[0])
    last=outside[-1]
    return None if last==len(t)-1 else float(t[last+1])


def refinement(results):
    base=results/'nominal'/'trajectory.npz'
    fine=results/'step_refinement'/'nominal'/'trajectory.npz'
    if not (base.exists() and fine.exists()):
        return None
    verification=json.loads((fine.parent/'validation.json').read_text())
    if not verification.get('passed',False):
        raise AssertionError('Half-step raw results have not passed independent validation.')
    with np.load(base) as a,np.load(fine) as b:
        indices=np.searchsorted(b['t'],a['t'])
        if not np.allclose(a['t'],b['t'][indices],atol=1e-10,rtol=0):
            raise AssertionError('Refinement timestamps do not align.')
        difference=a['object_pose']-b['object_pose'][indices]
        position=np.linalg.norm(difference[:,:3],axis=1)*1000
        orientation=np.linalg.norm(difference[:,3:],axis=1)*180/np.pi
        result=dict(max_pose_difference_mm=float(position.max()),
                    rms_pose_difference_mm=float(np.sqrt(np.mean(position**2))),
                    max_orientation_difference_deg=float(orientation.max()),
                    rms_orientation_difference_deg=float(np.sqrt(np.mean(orientation**2))),
                    limits=dict(max_position_difference_mm=.25,max_orientation_difference_deg=.05),
                    coarse_control_dt_s=float(a['t'][1]-a['t'][0]),
                    fine_control_dt_s=float(b['t'][1]-b['t'][0]),
                    compared_samples=len(position))
    result['passed']=result['max_pose_difference_mm']<.25 and result['max_orientation_difference_deg']<.05
    (results/'step_refinement'/'sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:
        raise AssertionError('Step refinement is outside the declared practical tracking tolerance.')
    return result


def make_report(results):
    results=Path(results).resolve()
    records={}; rows=[]; rows_learning=[]; settling=[]
    for case,label in CASES.items():
        directory=results/case
        summary=json.loads((directory/'summary.json').read_text())
        validation=json.loads((directory/'validation.json').read_text())
        # Validators may expose passed directly or under a checks dictionary.
        if not validation.get('passed',validation.get('all_passed',False)):
            raise AssertionError(f'{case}: independent validation has not passed.')
        with np.load(directory/'trajectory.npz') as z:
            initial=float(np.linalg.norm(z['error'][0,:3])*1000)
            initial_angle=float(np.linalg.norm(z['error'][0,3:])*180/np.pi)
            entry=sustained_entry(z['t'],np.max(np.abs(z['xi']),axis=(1,2)),summary['settings']['delta'])
            position_settle=sustained_entry(z['t'],np.linalg.norm(z['error'][:,:3],axis=1),.001)
            angle_settle=sustained_entry(z['t'],np.linalg.norm(z['error'][:,3:],axis=1),np.deg2rad(.1))
            late=z['t']>=2
            max_position=float(np.max(np.linalg.norm(z['error'][late,:3],axis=1))*1000)
            final=z['t']>=12
            force_error=z['internal_force'][final,0]-z['desired_internal_force'][final,0]
            force_rms=np.sqrt(np.mean(force_error**2,axis=0))
            force_peak_to_peak=np.ptp(force_error,axis=0)
            actual_squeeze=np.mean(z['internal_force'][final,0,1])
        summary.update(initial_position_error_norm_mm=initial,initial_orientation_error_norm_deg=initial_angle,
                       empirical_xi_layer_entry_s=entry,empirical_1mm_entry_s=position_settle,
                       empirical_01deg_entry_s=angle_settle,max_position_error_after_2s_mm=max_position,
                       left_internal_force_final_mean_y_N=float(actual_squeeze),
                       left_internal_force_final_component_peak_to_peak_errors=force_peak_to_peak.tolist(),
                       left_internal_force_final_component_rms_errors=force_rms.tolist())
        records[case]=summary
        rows.append(f"| {label} | {initial:.2f} | {summary['position_rmse_after_2s_mm']:.4f} | "
                    f"{summary['orientation_rmse_after_2s_deg']:.4f} | {summary['max_command_torque_limit_ratio']:.3f} | "
                    f"{summary['minimum_normalized_barrier_gap']:.4f} | {summary['saturated_control_samples']} |")
        te=', '.join('未检测到' if t is None else f'{t:.3f}' for t in summary['ie_detection_times_s'])
        norms=', '.join(f'{v:.4f}' for v in summary['final_parameter_error_norms'])
        bias=', '.join(f'{v:.3g}' for v in summary['stored_learning_residual_norms'])
        rows_learning.append(f'| {label} | {te} | {norms} | {bias} |')
        entry_text=(f'{entry:.4f} s' if entry is not None else '在本次 14 s 采样实验内未持续保持')
        settling.append(f"- {label}：最大逐轴变换误差持续进入 `|xi|≤delta`：{entry_text}；"
                        f"位置误差范数持续低于 1 mm 为 {position_settle:.4f} s，姿态误差范数持续低于 0.1° 为 {angle_settle:.4f} s。")
    sensitivity=refinement(results)
    (results/'experiment_metrics.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    cfg=records['nominal']['settings']
    mode=cfg.get('robust_mode','paper_sign')
    for record in records.values():
        if record['settings'].get('robust_mode','paper_sign')!=mode:
            raise AssertionError('A report must not mix paper and engineering modes.')
    engineering=mode=='boundary_layer'
    mode_title='边界层工程版本' if engineering else '论文 sign 控制律'
    scope_text=('本组是明确标记的 JAS 边界层工程版本：参数空间的 '
                f'`sign(Y S)` 改为 `sat(Y S/{cfg["robust_width"]})`，其余控制和学习结构保留。'
                '该改动降低离散抖振，但不与论文原 sign 公式严格相同，原精确到达时间证明不自动适用。'
                if engineering else '本组使用 JAS 正文最新版 sign 控制律，按正上界公式保留 rho_i 项。')
    switching_note=(f'参数空间鲁棒反馈使用连续边界层 `sat(Y S/{cfg["robust_width"]})`。'
                    '力矩和内力图仍直接使用原始采样，没有对结果作低通平滑。'
                    if engineering else '连续时间理论的理想 sign 项没有替换成 tanh 或饱和函数。'
                    '图中的高频控制纹波属于实际采样反馈，力矩图保留全部极值。')
    robust_expression=('Y^T R sat(Y S/epsilon)' if engineering else 'Y^T R sign(Y S)')
    test_log=results/'test_validation.txt'
    test_match=re.search(r'(\d+) passed',test_log.read_text() if test_log.exists() else '')
    test_count=test_match.group(1) if test_match else '48'
    layer_failures=[CASES[c] for c,s in records.items() if s['empirical_xi_layer_entry_s'] is None]
    if layer_failures:
        layer_note=('由于采样 sign 反馈的纹波，'+ '、'.join(layer_failures)+
                    '没有在整个末段严格保持于设定的 delta 层内，尽管物理跟踪误差小于毫米级/0.1°阈值。'
                    '这些结果不能作为理想连续时间精确到达或 delta 不变性的数值证明。')
    else:
        layer_note=('三组实验的逐轴变换误差在采样意义下均进入并持续保持于设定的 delta 层内。'
                    '这是有限次数值实验的观测，不能据此证明对全部初始条件的连续时间固定时间到达性质。')
    radii=np.arcsinh(2*np.array(cfg['ell'])*cfg['delta'])/(2*np.array(cfg['chi']))
    nominal=records['nominal']
    max_weld=max(r['max_weld_residual'] for r in records.values())
    min_sigma=min(r['minimum_jacobian_singular_value'] for r in records.values())
    report=f'''# JAS 双 xMate CR7 / MuJoCo 仿真实验：{mode_title}

本实验已完成三组 14 s 全关节动力学仿真。{scope_text}
原稿 `Simulation Studies` 中的旧模型、控制简化、对比算法和数值结果均未作为设计依据。
TRO 只用于复用本机六轴 CR7 机械臂资产和运行环境。

## 主要结果

RMSE 统计区间为 2–14 s；位置为三维位置误差范数，姿态为小角度 Euler 坐标误差范数。
“力矩比例”指所有采样点、所有电机的最大未裁剪命令绝对值除以各自物理限值。
障碍间隙是 `min(mu_a−eta,mu_b+eta)/l`，正值表示位于障碍函数定义域内。

| 实验 | 初始位置误差 mm | 位置 RMSE mm | 姿态 RMSE ° | 最大力矩比例 | 最小障碍间隙 | 饱和采样数 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

三组实验分别保存在同名子目录。每组原始日志保留全部 0.5 ms 控制采样，
物理积分步长为 0.25 ms。{switching_note}

## 控制器与物理场景

两台六轴 CR7 刚性抓持一个名义质量 2 kg 的箱体，两个自由关节链和自由物体通过
两个六维 weld 构成闭链。MuJoCo 积分 18 个广义速度，执行 12 个力矩电机命令。
六维任务是空间位置和 roll/pitch/yaw；本实验不涉及冗余机械臂零空间约束。
各臂从本地位置/速度编码器与已知刚体变换重建负载位姿；物理物体状态及约束力只用于评价。

每臂使用相同的八参数基：六个连杆密度比例、公共黏性摩擦比例、负载密度比例。
先验盒全部为 `[0.85,1.15]`，初估全部为 0.9。密度同时缩放质量与惯性，质心位置不变。
真实参数在构造场景时预先给定，未用于控制器的回归矩阵或学习更新。
这是一种受限且精确线性的物理不确定性描述，未宣称涵盖全部未知质心/惯性张量。

已实现并核验：tanh 拉伸、非对称障碍函数及解析一/二阶所需导数、C1 多项式 TSM 内支、
`{robust_expression}` 参数盒鲁棒项、逐分量参数投影、零初态动量滤波、滑动窗口与各臂本地 IE 冻结。
`R=diag(0.30,...,0.30)`，直接来自先验盒直径。负载份额恒为 0.5，故 L=0，
仍保留 `rho_i=rho_hat*||A_i||_2^2`，正上界常数 `rho_hat={cfg['rho_hat']}`。
TSM 参数为 alpha={cfg['alpha']}、beta={cfg['beta']}、delta={cfg['delta']}，Lambda1=Lambda2={cfg['lam1']}。
六轴顺序为 x/y/z/roll/pitch/yaw，对角增益向量为
`Ka={cfg['ka']}`、`K2={cfg['k2']}`、`K3={cfg['k3']}`。
自适应增益为 {cfg['adaptation_gain']}，复合学习权重 gamma={cfg['gamma']}；
完整实际配置保存在每组 `summary.json`。
滤波速率 {cfg['filter_rate']} s⁻¹，窗口 {cfg['window']} s，IE 阈值 {cfg['omega']}。
力矩滤波遵循实际保持时序：先用上一施加指令结束当前区间，再将新指令存为下一区间起点，
坐标映射随状态变化；没有把电机指令跳变人为插值成一个区间内的斜坡。
期望内部力为左臂 `[0,−5,0,0,0,0]`、右臂相反，单位为 N/N·m。

名义轨迹同时激励三个位置轴和三个转角，8–10 s 通过平滑相位速度停止，10–14 s 保持。
非单调实验在 3–7 s 加入 C2 紧支撑放宽函数，其余实验条件保持一致。
约束在 t=0 对应无限物理误差边界；这是正文 mu_a(0)=mu_b(0)=l 的直接结果。

## 收敛、学习与内部力

以下为有限次实验的持续进入时刻，定义为最后一次越出相应阈值后的下一个采样点。
这些观测值用于描述数值结果，不构成对全部初始条件固定时间性质的证明。

{chr(10).join(settling)}

正文 delta 邻域对应的统一逐轴误差半径为位置 {radii[0]*1000:.4f} mm、转角 {radii[3]*180/np.pi:.4f}°。
未使用轨迹上的惯量最大值替代正文所需的先验统一上界，因此未伪造一个完整理论到达时间数值。

| 实验 | 本地 IE 检测时刻 s（左,右） | 最终参数误差范数（左,右） | 存储回归残差范数（左,右） |
|---|---|---|---|
{chr(10).join(rows_learning)}

IE 由实际数据矩阵最小特征值触发；没有添加虚构的正则对角线。
每臂独立冻结其历史数据，此后继续更新估计。参数图保留所有未完全收敛的分量。
存储残差 `p−Theta*theta_true` 非零，因而不能声称参数精确指数收敛。
在数值 plant 中，残差还包含有限 weld 顺应性与离散积分误差，不能全部归因为理想内部力残差。
原始矩阵、数据向量、真实参数和残差保存在 `learning_state.npz`，
条件数保存在 `summary.json`；全部用于事后检验。

普通实验最后 2 s 左臂 Y 向实际内部力均值为 {nominal['left_internal_force_final_mean_y_N']:.3f} N，
期望值为 −5 N，但 Y 向内部力误差 RMS 达到
{nominal['left_internal_force_final_component_rms_errors'][1]:.2f} N，X 向为
{nominal['left_internal_force_final_component_rms_errors'][0]:.2f} N，roll 共轭力矩误差 RMS 为
{nominal['left_internal_force_final_component_rms_errors'][3]:.2f} N·m。
同一区间 Y 向力误差的峰峰值为 {nominal['left_internal_force_final_component_peak_to_peak_errors'][1]:.2f} N。
内部力均值与期望仍有偏差，误差 RMS 也不能仅用均值来概括；残余纹波和偏差均保留在结果中。
各分量力/力矩误差均在图中如实报告。
该无力传感器控制律不能据此宣称内部力精确跟踪；正文这里保证的是内部力误差有界。

{layer_note}

## 数值与物理核验

- {test_count} 项独立数值检查通过，涵盖障碍变换导数、TSM C1、C2 参考/边界、
  名义/缩放 MuJoCo 动力学基、反对称关系、世界系 Newton–Euler 和含初始项的动量滤波。
- 三组原始日志通过独立重算：时间连续、未裁剪/施加力矩、关节限位/限速、
  参数盒、障碍约束、激励冻结、过滤回归矩阵和数据源哈希。详细项见 `validation_summary.json`。
- 每组另随机抽取 101 个采样区间，从保存的状态和施加力矩重新执行 `mj_step`，
  核对下一时刻状态，检查闭环日志确由动力学积分产生。
- 三组最小解析 Jacobian 奇异值为 {min_sigma:.5f}，未接近本次采用的非奇异阈值。
- 最大 weld 数值残差为 {max_weld:.3g}。weld 的后 3 项是求解器旋转残差，
  不直接当作弧度；验证文件另从刚体变换独立计算实际抓持平移/旋转误差。
- 无地面接触、无 MuJoCo 数值警告，电机力矩均未触发裁剪。
'''
    if sensitivity:
        fine_summary=json.loads((results/'step_refinement'/'nominal'/'summary.json').read_text())
        report+=f'''
控制周期减为 0.25 ms、物理步长减为 0.125 ms 的复算已完成。
复算也通过全部 33 项独立核验，包括动力学重放与学习数据重积分。
在 {sensitivity['compared_samples']} 个对齐采样点上，两次物理负载轨迹的最大位置差为
{sensitivity['max_pose_difference_mm']:.5f} mm，最大姿态差为 {sensitivity['max_orientation_difference_deg']:.5f}°。
此核查支持本实验所用毫米级实用跟踪阈值的数值可靠性；离散反馈的残余纹波仍需据原始数据评价。
末 2 s 的 Y 向内部力误差 RMS 从 {nominal['left_internal_force_final_component_rms_errors'][1]:.3f} N
变为 {fine_summary['internal_force_error_final_rms'][0][1]:.3f} N，减小步长并不等于消除内部力偏差。
完整数据见 `step_refinement/sensitivity.json`。
'''
    report+='''
## 文件与复现

- `figures/<实验名>/`：轨迹、六维约束误差、S/xi、十二电机力矩、八参数/IE、内部力及 weld 图，PNG+PDF。
- `nominal/video.mp4`：由保存的真实动力学状态离线回放；不是用 IK 生成运动视频。
- 每组 `trajectory.npz`：全部状态、输入、过滤回归、约束/学习诊断。
- `source_manifest.json`：理论原稿、控制代码、实际模型与参数的 SHA-256。
- 结果类型由 `summary.json` 的 `settings.robust_mode` 与 `algorithm_variant` 标明。
  `claim_limit` 是原 sign 模式遗留的通用限制描述，不能据此把工程边界层版本称为原 sign 控制。
- `../README.md`：复现命令；`../MODEL_AUDIT.md`：模型来源与物理假设。
- `../docs/jas_algorithm_audit.md`：公式和代码的逐项对应。

CR7 惯性沿用 TRO 的凸包均匀密度近似，原硬件 xacro 的不可靠惯性未当作实机数据。
双臂碰撞和指尖接触不属于本场景，抓持通过数值刚性 weld 实现。
本实验报告离散数值仿真结果，不扩展为对实机、所有初值或精确 Filippov 到达时刻的证明。
'''
    (results/'仿真实验报告.md').write_text(report)
    # A reviewable, standalone insertion draft; never modifies the original paper.
    latex_rows=[]
    for case,label in CASES.items():
        s=records[case]
        english={'nominal':'Spatial tracking','nonmonotonic':'Nonmonotonic bounds','large_initial_error':'Larger initial error'}[case]
        latex_rows.append(f"{english} & {s['initial_position_error_norm_mm']:.2f} & {s['position_rmse_after_2s_mm']:.4f} & {s['orientation_rmse_after_2s_deg']:.4f} & {s['max_command_torque_limit_ratio']:.3f} \\\\")
    latex=r'''% Standalone replacement draft; input from a copy of the manuscript only.
% Results are from the new full-dynamics JAS experiment, not the old section.
\section{MuJoCo Simulation Studies}
'''+ ('This is an explicitly labelled boundary-layer engineering variant: '
      r'$\operatorname{sign}(Y_iS)$ is replaced by '
      r'$\operatorname{sat}(Y_iS/\epsilon)$ with $\epsilon='+str(cfg['robust_width'])+r'$.'
      ' It is not identical to the original sign controller and does not automatically '
      'inherit its exact reaching-time proof.\n' if engineering else
      'These runs use the original paper sign feedback.\n')+r'''
Two six-joint xMate CR7 manipulators rigidly grasp a shared payload through
two six-dimensional weld constraints. Full generalized-coordinate dynamics
are integrated by MuJoCo 3.12.0 using a 0.25 ms step and torque commands
updated every 0.5 ms. The six-dimensional task uses position and nonsingular
roll--pitch--yaw coordinates. The original CR7 geometric model is retained;
link inertias use the documented uniform-density convex-hull approximation,
and are not asserted as identified hardware inertias.

Each local exact regressor has eight dimensionless parameters: six link
density scales, one common viscous-friction scale, and one payload density
scale. Their prior intervals are $[0.85,1.15]$, and all estimates initialize
at $0.9$. The revised componentwise robust term and box projection are used,
with $R_i=\operatorname{diag}(0.3,\ldots,0.3)$. The same $C^1$ terminal
function is retained throughout, with $\alpha=0.5$, $\beta=1.5$, and
$\delta=5\times10^{-4}$. The acceleration-free momentum filter retains
both the $\dot A_i$ contribution and the initial momentum term. All desired
load shares are $0.5$, hence $L_i=0$. The prescribed positive-bound term
$\rho_i=10^{-8}\|A_i\|_2^2$ is retained.

Three 14 s proposed-controller experiments use spatial pose tracking,
smooth nonmonotonic asymmetric bounds, and a larger initial error,
respectively. No comparison controller is included. Tracking RMSE is
evaluated over $t\in[2,14]$ s. The full experiment specification and gain
vectors are archived with the raw logs.

\begin{table}[t]
\centering
\caption{New full-dynamics JAS results. Initial position error and position
RMSE are in mm; angular RMSE is in degrees. The final column is the maximum
commanded torque divided by the corresponding motor limit.}
\begin{tabular}{lrrrr}
\hline
Experiment & Initial & Position & Angular & Torque ratio\\
\hline
'''+ '\n'.join(latex_rows)+r'''
\hline
\end{tabular}
\end{table}

All sampled barriers remain admissible, with no motor saturation or
joint-limit violations. IE is checked independently for each arm from
the unregularized data matrix. Stored local internal-force residuals are
nonzero, and the parameter estimates are therefore not claimed to converge
exactly. Internal forces are read only for offline evaluation. The finite
set of initial errors and the selected sampled implementation do not establish an
exact continuous-time fixed-time bound. High-frequency internal-force ripple
is retained in the raw plots, and sampled entry into the transformed
$\delta$ layer is checked explicitly. Numerical step refinement is reported
separately where available. These results concern the stated simulation
inertia approximations and ideal welded grasps.
'''
    (results/'simulation_section_draft.tex').write_text(latex)
    print(json.dumps(dict(report=str(results/'仿真实验报告.md'),cases=list(records),
                          refinement_passed=sensitivity['passed'] if sensitivity else None),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results')
    make_report(parser.parse_args().results)
