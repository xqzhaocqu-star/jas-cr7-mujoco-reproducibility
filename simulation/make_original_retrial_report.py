#!/usr/bin/env python3
"""Report the ORIGINAL JAS retrial, including unsuccessful identification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = ('nominal', 'nonmonotonic', 'large_initial_error')


def case_metrics(directory):
    summary = json.loads((directory/'summary.json').read_text())
    validation = json.loads((directory/'validation.json').read_text()) if (directory/'validation.json').exists() else {}
    with np.load(directory/'trajectory.npz') as raw:
        times, theta = raw['t'], raw['theta']
        final = times >= times[-1]-2.
        true = np.asarray(summary['true_parameters_offline_only'])
        errors = theta-true
        initial_norm, final_norm = np.linalg.norm(errors[0], axis=1), np.linalg.norm(errors[-1], axis=1)
        tail = theta[final]
        time_zero = times[final]-np.mean(times[final])
        slopes = np.einsum('n,nak->ak', time_zero, tail-np.mean(tail, axis=0))/np.sum(time_zero**2)
        force_error = raw['internal_force'][final]-raw['desired_internal_force'][final]
        force_ac = force_error-np.mean(force_error, axis=0)
        final_drive = raw['learning_drive'][final]
        fixed_point = float(np.max(raw['update_fixed_point_error']))
        theta_rms = np.sqrt(np.mean(errors[final]**2, axis=0))
        last_xi = np.max(np.abs(raw['xi'][final]), axis=(0, 2))
        final_theta_drift = np.linalg.norm(tail[-1]-tail[0], axis=1)
        lower, upper = summary['settings']['prior_lower'], summary['settings']['prior_upper']
        final_face_components = ((np.abs(theta[-1]-lower) <= 1e-9)
                                 | (np.abs(theta[-1]-upper) <= 1e-9))
    with np.load(directory/'learning_state.npz') as saved:
        gram, p, residual = saved['gram'], saved['p'], saved['stored_residual']
    upsilon = np.asarray(summary['settings']['adaptation_gain'])
    sqrt_u = np.sqrt(upsilon)
    eigen = np.linalg.eigvalsh(sqrt_u[None, :, None]*gram*sqrt_u[None, None, :])*summary['settings']['gamma']
    return dict(case=directory.name, summary=summary,
                validation_passed=validation.get('passed', False),
                validation_check_count=len(validation.get('checks', {})), validation_failures=validation.get('failures', []),
                raw_sha256=hashlib.sha256((directory/'trajectory.npz').read_bytes()).hexdigest(),
                initial_truth_error_norms=initial_norm.tolist(), final_truth_error_norms=final_norm.tolist(),
                final_error_to_initial_ratios=(final_norm/initial_norm).tolist(),
                final_estimates=theta[-1].tolist(), true_parameters_offline_only=true.tolist(),
                final_component_errors=errors[-1].tolist(), final_two_second_component_error_rms=theta_rms.tolist(),
                final_prior_face_components=final_face_components.tolist(),
                final_two_second_parameter_vector_changes=final_theta_drift.tolist(),
                final_two_second_maximum_absolute_component_slopes=np.max(np.abs(slopes), axis=1).tolist(),
                raw_gram_eigenvalues=np.linalg.eigvalsh(gram).tolist(),
                history_homogeneous_eigenvalues_s_inv=eigen.tolist(),
                history_homogeneous_rate_scope='Only the fixed-history interior linear channel; not the complete projected adaptive closed-loop rate.',
                frozen_total_regression_residual_norms=np.linalg.norm(residual, axis=1).tolist(),
                final_learning_tracking_drive_rms=np.sqrt(np.mean(final_drive**2, axis=0)).tolist(),
                maximum_logged_implicit_fixed_point_error=fixed_point,
                implicit_fixed_point_metric_applicable=summary['settings']['learning_integrator'] == 'projected_implicit',
                final_two_second_internal_force_error_rms=np.sqrt(np.mean(force_error**2, axis=0)).tolist(),
                final_two_second_internal_force_ac_rms=np.sqrt(np.mean(force_ac**2, axis=0)).tolist(),
                final_two_second_maximum_absolute_xi=last_xi.tolist())


def fmt_pair(values, digits=6):
    return ' / '.join(f'{float(v):.{digits}g}' for v in values)


def generate(results):
    cases = [case_metrics(results/case) for case in CASES]
    fine_path = results/'step_refinement/nominal'
    fine = case_metrics(fine_path) if (fine_path/'source_manifest.json').exists() else None
    first = cases[0]['summary']
    cfg = first['settings']
    diagnoses = {case:json.loads((results/case/'residual_diagnosis.json').read_text())
                 for case in CASES if (results/case/'residual_diagnosis.json').exists()}
    fine_diagnosis = (json.loads((fine_path/'residual_diagnosis.json').read_text())
                      if (fine_path/'residual_diagnosis.json').exists() else None)
    fine_validation = (json.loads((fine_path/'validation.json').read_text())
                       if (fine_path/'validation.json').exists() else {})
    aggregate = dict(method='Original JAS torque and original projected composite learner', cases=cases,
                     offline_residual_diagnoses=diagnoses,
                     step_refinement_offline_residual_diagnosis=fine_diagnosis,
                     step_refinement_comparison=fine_validation.get('step_comparison', {}),
                     step_refinement_learning_comparison=fine_validation.get('original_learning_step_comparison', {}),
                     step_refinement=fine, all_validations_passed=all(c['validation_passed'] for c in cases)
                     and (fine is None or fine['validation_passed']),
                     interpretation='Numerical evidence for the stated experiments; not a universal theorem proof.')
    (results/'original_method_metrics.json').write_text(json.dumps(aggregate, indent=2, ensure_ascii=False)+'\n')
    if cfg['learning_integrator'] == 'explicit_euler':
        integration_note = '正式实验直接调用原 `jas.learning.CompositeLearner` 的显式投影更新，完整保留 `-YS+gamma*epsilon`；新目录没有替代正式学习器。高增益试跑采用隐式步处理刚性，但只作为参数诊断，未作为本轮正式学习更新。正式半步长复算检查显式离散与物理积分的敏感性。'
    else:
        integration_note = '投影学习方程采用盒约束隐式 Euler 数值积分，完整求解其耦合变分不等式；这只是原连续方程的一阶离散，不是把历史通道替换成纯历史指数流。半步长复算另报，不声称数值积分精确等于连续 Filippov 解。'
    lines = [
        '# 原 JAS 控制与复合学习律：重新仿真报告', '',
        '本次正式结果只使用论文理论部分的原控制律和原复合学习律。未采用论文旧仿真章节的算法简化；未使用此前测力补偿/纯历史预条件辨识扩展。', '',
        '本轮按论文的一般结论评价：运动和约束保持、参数有界，以及 IE 后残差相关的指数最终界。非零权重误差不是失败判据；到真值的误差只作为辨识准确性指标，真值指数趋零需要额外的零冻结残差条件。有限数值轨迹不替代完整理论指数包络证明。', '',
        f'三组 {first["duration_s"]:g} s 正式实验最终八维参数真值误差（左/右）分别为：'+
        '；'.join(f'{item["case"]} {fmt_pair(item["final_truth_error_norms"])}' for item in cases)+'。最终误差与后期漂移需要和跟踪、IE、冻结残差分开判断。', '',
        '## 方法一致性', '',
        r'原学习律为 $\dot{\hat\theta}_i=\Upsilon_i\operatorname{Pr}_{\Omega_i}[-Y_iS+\gamma_i(p_i-\check\Theta_i\hat\theta_i)]$。跟踪误差项 `-YS` 与历史项都保留，`Upsilon` 在运行前固定且为正对角矩阵；没有在线 `G^-1` 学习增益、历史目标门控、额外反力反馈或学习律替换。', '',
        '控制器保留原 `sign(Y S)`、先验盒直径鲁棒项 `R=diag(0.3)`、正 rho、同一 C1 终端滑模内支。先验仍为 `[0.85,1.15]^8`，初始估计仍为 `0.9`；真实参数只在整个物理积分结束后读取用于评价。', '',
        integration_note, '',
        '## 固定配置', '',
        f'- 每组时长 {first["duration_s"]:g} s；控制周期 {first["control_dt_s"]:g} s，物理步长 {first["physics_dt_s"]:g} s，MuJoCo {first["integrator"]}。',
        f'- 六维 `{cfg["excitation_profile"]}` 多正弦，{cfg["excitation_duration"]-2:g}–{cfg["excitation_duration"]:g} s 平滑停止后保持；三工况使用相同控制和学习增益。',
        f'- 滤波率 {cfg["filter_rate"]:g} /s，窗口 {cfg["window"]:g} s，原始 IE 阈值 {cfg["omega"]:g}；严格按首次 `t>=window && lambda_min(G)>=omega` 冻结各臂历史。',
        f'- gamma = {cfg["gamma"]:g}；Upsilon 对角 = `{cfg["adaptation_gain"]}`。',
        '- 调参候选与名义设计来源记录于 `tuning_summary.json`（如存在）。若常对角增益按名义 Gram 列尺度设置，它仍在试验前固定，不随实际数据改变。', '',
        '## 正式数值结果', '',
        '| 工况 | 末2s位置RMSE mm | 末2s姿态RMSE deg | 初始参数误差 左/右 | 最终参数误差 左/右 | IE时间 左/右 s | 冻结总残差范数 左/右 |',
        '|---|---:|---:|---|---|---|---|',
    ]
    for item in cases:
        s = item['summary']
        ie = ' / '.join('未发生' if v is None else f'{v:g}' for v in s['ie_detection_times_s'])
        lines.append(f'| {item["case"]} | {s["final_position_rmse_mm"]:.6g} | {s["final_orientation_rmse_deg"]:.6g} | {fmt_pair(item["initial_truth_error_norms"])} | {fmt_pair(item["final_truth_error_norms"])} | {ie} | {fmt_pair(item["frozen_total_regression_residual_norms"])} |')
    lines += ['', '参数误差指八维估计减真值的 Euclidean 范数。冻结残差是变换后回归量的积分残差，不是 N 或 N·m。IE 正定不能推出冻结回归无偏。', '',
              '| 工况 | 末2s参数向量变化 左/右 | 最大力矩/限额 | 最大速度/限额 | 独立验证 |',
              '|---|---|---:|---:|---|']
    for item in cases:
        s = item['summary']
        status = 'PASS' if item['validation_passed'] else 'FAIL: '+', '.join(item['validation_failures'])
        lines.append(f'| {item["case"]} | {fmt_pair(item["final_two_second_parameter_vector_changes"])} | {s["max_command_torque_limit_ratio"]:.6g} | {s["max_joint_speed_limit_ratio"]:.6g} | {status} ({item["validation_check_count"]} checks) |')
    lines += ['', '## 如何判断原方法', '',
              '1. 运动控制：依据全部保存时刻的真实负载/本地误差、约束、Jacobian、关节/扭矩限制，以及完整动力学短区间重放判断；只说明这些工况的数值表现，不以有限仿真证明所有初值的固定时间定理。',
              '2. 权重辨识：直接比较八维真值误差和末段漂移；预测误差下降、触发 IE 或权重曲线变平都不能单独说明收敛真值。先验盒鲁棒项允许在参数估计尚不精确时保持跟踪，因此跟踪精度不能替代辨识精度。原律还包含持续的 `-YS` 驱动，不把它当成指数趋向 `G^-1 p` 的纯历史系统。',
              '3. 理论条件：在 IE 及其余理想模型假设成立时，论文 Theorem 1 给出残差相关指数界；所有实际冻结残差为零时给出真值指数收敛。非零残差本身不足以排除任意闭环中的 YS 抵消；但若同时采用原理想定理的 S 趋零、Y 有界结论，且真值严格在盒内，则 YS 趋零，非零冻结残差与参数趋向真值不相容。这是条件性的连续律分析，不能将有限采样轨迹直接当作极限证明。详见 docs/parameter_convergence_diagnosis.md。',
              '4. 内力与数值近似：原局部回归含未测内部力误差，有限数值 weld/采样滤波也贡献总残差。下节用实际数据作离线分解；没有向在线学习器补偿这些量。原理论保证内部力误差有界，没有内部力误差归零定理。', '',
              '## 历史残差的独立离线分解', '']
    if diagnoses:
        lines += [r'逐时刻以保持电机输入的左右极限重建实际反力，构造 $d=a^{-1}(f_I-f_d)$，从零初值作同一滤波；在原首次 IE 接受的窗口积分 $\zeta_{force}=-\int Y_f d_f\,dt$。这是事后诊断，原学习输入未被改变。', '',
                  '| 工况 | 总残差范数 左/右 | 内力残差向量范数 左/右 | 相减后剩余范数 左/右 |',
                  '|---|---|---|---|']
        for case, diagnosis in diagnoses.items():
            arms = diagnosis['arms']
            lines.append(f'| {case} | {fmt_pair([a["total_residual_norm"] for a in arms])} | {fmt_pair([a["internal_force_residual_norm"] for a in arms])} | {fmt_pair([a["remaining_residual_norm"] for a in arms])} |')
        remaining_ratios = [a['remaining_norm_divided_by_total_norm']
                            for d in diagnoses.values() for a in d['arms']
                            if a['remaining_norm_divided_by_total_norm'] is not None]
        if remaining_ratios and max(remaining_ratios) < .01:
            lines += ['', f'本轮分解后的剩余/总残差范数为 {min(remaining_ratios):.3g}–{max(remaining_ratios):.3g}。这给出了本轮冻结回归残差主要来自未测内力误差的直接证据；并非仅凭总残差范数推测。']
        lines += ['', '应比较残差向量相减后的剩余，不能把不同向量范数相加当作污染比例。剩余同时含有限 weld、局部坐标近似和离散滤波/积分误差；这些来源未单独辨识。该分解可以解释本轮历史数据为何有偏，但不单独证明完整闭环的参数极限。', '']
    else:
        lines += ['尚无完成的离线残差分解，不能将总残差预先全部归因内力。', '']
    lines += ['## 步长复核', '']
    if fine is not None:
        s = fine['summary']
        v = fine_validation
        lines += [f'名义半步长：控制 {s["control_dt_s"]:g} s、物理 {s["physics_dt_s"]:g} s，最终参数误差 左/右 {fmt_pair(fine["final_truth_error_norms"])}，独立验证 {"PASS" if fine["validation_passed"] else "FAIL"}。', '',
                  '与主步长的实际位姿差异及参数差异分别保存在 `validation.json` 的 `step_comparison` 和 `original_learning_step_comparison`。下列数值只描述离散敏感性，不是连续收敛证明。', '']
        pose_comparison = v.get('step_comparison', {})
        learning_comparison = v.get('original_learning_step_comparison', {})
        if pose_comparison:
            lines += [f'- 2 s 后共同区间：位置差 RMS {pose_comparison["position_difference_rms_mm"]:.6g} mm，姿态坐标差 RMS {pose_comparison["orientation_coordinate_difference_rms_deg"]:.6g} deg。']
        if learning_comparison:
            final_difference = np.linalg.norm(learning_comparison['final_common_parameter_difference'], axis=1)
            max_difference = np.max(np.abs(learning_comparison['maximum_absolute_parameter_trace_difference']), axis=1)
            lines += [f'- 共同终点参数向量差范数 左/右 {fmt_pair(final_difference)}；全程最大单分量参数差 左/右 {fmt_pair(max_difference)}。',
                      f'- 冻结 Gram 差的相对指标 左/右 {fmt_pair(learning_comparison["frozen_gram_relative_difference"])}，冻结 p 差的相对指标 左/右 {fmt_pair(learning_comparison["frozen_p_relative_difference"])}（归一化分母均为 `1+主步长范数`）。']
        if fine_diagnosis:
            arms = fine_diagnosis['arms']
            lines += [f'- 半步长总历史残差范数 左/右 {fmt_pair([a["total_residual_norm"] for a in arms])}；事后分解内力项后的剩余为 {fmt_pair([a["remaining_residual_norm"] for a in arms])}。数值剩余减小，但主要历史偏差仍存在。']
    else:
        lines += ['尚未有完整半步长结果。']
    lines += ['', '## 查看结果', '',
              '- `figures/<case>/01_trajectory.pdf`：轨迹。',
              '- `figures/<case>/02_errors.pdf`：实际误差和非对称边界。',
              '- `figures/<case>/05_learning.pdf`：八参数、真实值、先验边界和原始 IE。',
              '- `figures/<case>/08_parameter_error_diagnostics.pdf`：到真值的误差、末段变化；不伪造指数拟合。',
              '- `figures/<case>/06_internal_force.pdf`：原始内力误差及纹波。',
              '- `nominal/video.mp4`：真实 MuJoCo 积分日志离线回放。',
              '- `validation_summary.json`、各组 `validation.json`：独立验证。', '',
              '原 `jas/` 源代码、旧原式结果和此前工程扩展均未被本轮覆盖。']
    (results/'原方法重新仿真报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(results=str(results), validations=aggregate['all_validations_passed'],
                          final_parameter_errors={item['case']:item['final_truth_error_norms'] for item in cases}), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE/'results_original_retrial')
    args = parser.parse_args()
    generate(args.results.resolve())
