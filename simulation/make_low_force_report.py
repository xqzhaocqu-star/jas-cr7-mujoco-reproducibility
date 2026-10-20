#!/usr/bin/env python3
"""Report fixed-parameter tuning of the unchanged original JAS method."""
import argparse
import json
from pathlib import Path

import numpy as np

from make_original_retrial_report import CASES, HERE, case_metrics, fmt_pair


def force_metrics(force, desired):
    error = force - desired
    bias = np.mean(error, axis=0)
    ac = error - bias
    return dict(
        actual_mean=np.mean(force, axis=0).tolist(),
        desired_mean=np.mean(desired, axis=0).tolist(),
        error_mean=bias.tolist(),
        error_component_rms=np.sqrt(np.mean(error**2, axis=0)).tolist(),
        ac_component_rms=np.sqrt(np.mean(ac**2, axis=0)).tolist(),
        error_component_peak_to_peak=np.ptp(error, axis=0).tolist(),
        actual_force_vector_rms_N=np.sqrt(np.mean(np.sum(force[..., :3]**2, axis=-1), axis=0)).tolist(),
        force_error_vector_rms_N=np.sqrt(np.mean(np.sum(error[..., :3]**2, axis=-1), axis=0)).tolist(),
        force_ac_vector_rms_N=np.sqrt(np.mean(np.sum(ac[..., :3]**2, axis=-1), axis=0)).tolist(),
        force_error_vector_peak_N=np.max(np.linalg.norm(error[..., :3], axis=-1), axis=0).tolist(),
        moment_error_vector_rms_Nm=np.sqrt(np.mean(np.sum(error[..., 3:]**2, axis=-1), axis=0)).tolist(),
        moment_ac_vector_rms_Nm=np.sqrt(np.mean(np.sum(ac[..., 3:]**2, axis=-1), axis=0)).tolist(),
        moment_error_vector_peak_Nm=np.max(np.linalg.norm(error[..., 3:], axis=-1), axis=0).tolist())


def read_case(path):
    metrics = case_metrics(path)
    cfg = metrics['summary']['settings']
    with np.load(path/'trajectory.npz', allow_pickle=False) as raw:
        t, theta = raw['t'], raw['theta']
        truth = np.asarray(metrics['true_parameters_offline_only'])
        masks = dict(full=np.ones(len(t), dtype=bool),
                     moving=(t >= 2.) & (t <= min(t[-1], cfg['excitation_duration'])),
                     final_two_seconds=t >= t[-1]-2.)
        windows = {}
        for name, mask in masks.items():
            windows[name] = force_metrics(raw['internal_force'][mask], raw['desired_internal_force'][mask])
            windows[name]['interval_s'] = [float(t[mask][0]), float(t[mask][-1])]
        norms = np.linalg.norm(theta-truth, axis=-1)
        metrics.update(force=windows, parameter_component_minimum=theta.min(axis=0).tolist(),
                       parameter_component_maximum=theta.max(axis=0).tolist(),
                       maximum_truth_error_norms=norms.max(axis=0).tolist(),
                       maximum_parameter_single_step=np.max(np.abs(np.diff(theta, axis=0)), axis=(0, 2)).tolist(),
                       finite_weights=bool(np.isfinite(theta).all()),
                       all_weights_in_prior=bool(np.all(theta >= cfg['prior_lower']-1e-12)
                                                 and np.all(theta <= cfg['prior_upper']+1e-12)),
                       prior_error_norm_bound=float(np.sqrt(theta.shape[-1])*(cfg['prior_upper']-cfg['prior_lower'])))
    return metrics


def generate(results, baseline):
    current = {name:read_case(results/name) for name in CASES}
    previous = {name:read_case(baseline/name) for name in CASES}
    fine_path = results/'step_refinement/nominal'
    fine = read_case(fine_path) if (fine_path/'source_manifest.json').exists() else None
    summary = current['nominal']['summary']
    cfg, old_cfg = summary['settings'], previous['nominal']['summary']['settings']
    changed = {key:dict(before=old_cfg[key], after=value) for key, value in cfg.items()
               if key in old_cfg and value != old_cfg[key]}
    same_task = all(current[name]['summary']['settings']['internal_force']
                    == previous[name]['summary']['settings']['internal_force']
                    and current[name]['summary']['settings']['excitation_profile']
                    == previous[name]['summary']['settings']['excitation_profile']
                    and current[name]['summary']['settings']['excitation_duration']
                    == previous[name]['summary']['settings']['excitation_duration'] for name in CASES)
    all_cases = list(current.values()) + ([fine] if fine else [])
    independent_path = results/'performance_assessment.json'
    independent = json.loads(independent_path.read_text()) if independent_path.exists() else None
    diagnoses = {name:json.loads((results/name/'residual_diagnosis.json').read_text())
                 for name in CASES if (results/name/'residual_diagnosis.json').exists()}
    aggregate = dict(goal='Original paper method; lower internal-force error and ripple; bounded weights.',
                     cases=current, baseline=previous, step_refinement=fine,
                     changed_fixed_parameters=changed, force_target_and_reference_unchanged=same_task,
                     core_validations_passed=all(item['validation_passed'] for item in all_cases),
                     finite_weights_in_prior=all(item['finite_weights'] and item['all_weights_in_prior'] for item in all_cases),
                     independent_performance_assessment=independent,
                     offline_residual_diagnoses=diagnoses,
                     full_theoretical_exponential_envelope_verified=False,
                     recommend_uniform_parameter_replacement=False,
                     interpretation='Correct original-method parameter trial. Force improves in some cases, but large-initial-error force and moments worsen. Retain the earlier original-method baseline as the general reference.')
    (results/'low_force_metrics.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 原 JAS 方法：内力与权重有界性复核', '',
             '本轮按论文理论部分的原控制律与完整复合学习律重新仿真。目标是减小同一内力任务下的误差和波动，并保持权重有界；不要求权重误差指数趋于零。', '',
             '这是原方法内的固定参数折中：正式结果必须逐工况同时看力、力矩、峰值和跟踪。不能把先导中较低的力RMS或更平稳的权重，转述为所有受力指标都改善。', '',
             '完整结果中，名义与非单调工况的力误差RMS下降，但大初值工况的力误差以及三组力矩误差RMS上升。因此不建议把这组参数统一替换既有正式基线。新结果作为忠实原法的调参复核保留；既有 results_original_retrial/ 也满足投影有界，并非权重发散。', '',
             '## 方法和任务', '',
             r'原学习律仍为 $\dot{\hat\vartheta}_i=\Upsilon_i\operatorname{Pr}_{\Omega_i}[-Y_iS+\gamma_i(p_i-\check\Theta_i\hat\vartheta_i)]$。原 sign、先验盒直径鲁棒项 R=diag(0.3)、正 rho、C1 终端滑模内支、无加速度动量滤波和各臂首次原始 IE 后冻结全部保留。', '',
             '控制器与学习器沿用原源码；反力仅作事后诊断，真实参数在整段物理积分结束后才用于评价。没有测力反馈、额外电机滤波、逆 Gram 预条件、删除 -YS 或更换学习律。', '',
             f'目标内力仍为左 Fy=-{cfg["internal_force"]:g} N、右 Fy=+{cfg["internal_force"]:g} N，其余分量目标为零。运动轨迹、初始状态、物理模型和先验与原24s实验相同。表中的“降低”比较相同任务与相同时段，没有靠降低力目标取得。', '',
             '## 固定参数', '']
    for key, change in changed.items():
        lines.append(f'- `{key}`：`{change["before"]}` → `{change["after"]}`。')
    lines += ['', f'每组 {summary["duration_s"]:g} s；控制周期 {summary["control_dt_s"]:g} s，物理步长 {summary["physics_dt_s"]:g} s，{summary["integrator"]}；滤波率 {cfg["filter_rate"]:g}/s、窗口 {cfg["window"]:g}s、原始IE阈值 {cfg["omega"]:g}。全部参数在运行前固定，三工况使用同一套参数。', '',
              '增大 delta 会放宽论文的理论跟踪邻域，降低到达增益会延长理论时间界；实际跟踪仍须独立验收。delta 是原终端滑模参数，未用于平滑鲁棒 sign。', '',
              '## 同任务的内力结果', '',
              '以下为末2s的左侧统计；两侧内部力按零和分解，误差范数相同。AC指去均值后的全部变化，包含缓慢漂移，不把它全部称为高频噪声。力与 Euler 共轭广义力矩分开计量。', '',
              '| 工况 | Fy误差RMS 原→新 N | 三维力误差RMS 原→新 N | 三维AC RMS 原→新 N | 采样力误差峰值 原→新 N | 广义力矩误差RMS 原→新 Nm |',
              '|---|---:|---:|---:|---:|---:|']
    for name in CASES:
        a, b = previous[name]['force']['final_two_seconds'], current[name]['force']['final_two_seconds']
        pair = lambda key:f'{a[key][0]:.5g} → {b[key][0]:.5g}'
        lines.append(f'| {name} | {a["error_component_rms"][0][1]:.5g} → {b["error_component_rms"][0][1]:.5g} | {pair("force_error_vector_rms_N")} | {pair("force_ac_vector_rms_N")} | {pair("force_error_vector_peak_N")} | {pair("moment_error_vector_rms_Nm")} |')
    lines += ['', '| 工况 | 新实际力向量RMS N | 新力误差均值 xyz N | 运动段三维力误差RMS 原→新 N | 全程采样力误差峰值 原→新 N |',
              '|---|---:|---|---:|---:|']
    for name in CASES:
        a, b = previous[name], current[name]
        tail = b['force']['final_two_seconds']
        bias = ', '.join(f'{v:.5g}' for v in tail['error_mean'][0][:3])
        lines.append(f'| {name} | {tail["actual_force_vector_rms_N"][0]:.5g} | [{bias}] | '
                     f'{a["force"]["moving"]["force_error_vector_rms_N"][0]:.5g} → {b["force"]["moving"]["force_error_vector_rms_N"][0]:.5g} | '
                     f'{a["force"]["full"]["force_error_vector_peak_N"][0]:.5g} → {b["force"]["full"]["force_error_vector_peak_N"][0]:.5g} |')
    lines += ['', '原 sign 鲁棒项仍会切换，本轮不声称消除了内力误差或抖振。不同工况、分量和峰值不一定同步改善；全部原始采样均保留，未平滑曲线或隐去峰值。统计峰值是保存的控制时刻峰值，非连续时间全局峰值。', '',
              '## 权重、跟踪及独立验证', '',
              '| 工况 | 权重最小/最大 | 全程最大真值误差 左/右 | 最终真值误差 左/右 | 末2s权重变化 左/右 | 末2s位置RMSE mm | 验证 |',
              '|---|---|---|---|---|---:|---|']
    for name, item in current.items():
        s = item['summary']
        lo, hi = np.min(item['parameter_component_minimum']), np.max(item['parameter_component_maximum'])
        lines.append(f'| {name} | {lo:.6g} / {hi:.6g} | {fmt_pair(item["maximum_truth_error_norms"])} | '
                     f'{fmt_pair(item["final_truth_error_norms"])} | {fmt_pair(item["final_two_second_parameter_vector_changes"])} | '
                     f'{s["final_position_rmse_mm"]:.5g} | {"PASS" if item["validation_passed"] else "FAIL"} ({item["validation_check_count"]}) |')
    lines += ['', '先验与原投影给出每个分量始终在 [0.85,1.15]，故真实参数也在盒内时，八维误差范数不超过 0.3√8≈0.848528。这个机制保证不发散；表中的实际误差、末段变化及投影活动另行报告，不能用盒内有界代替学习精度。', '',
              '各组 validation.json 独立重建完整 -YS 与历史学习步、原始滤波/IE、控制力矩，并重放实际 MuJoCo 物理区间。另由各组 low_force_assessment.json 和总表 performance_assessment.json 审查参数正定、原任务一致性、权重/投影与受力性能。', '',
              '## 与论文指数界的关系', '',
              r'论文在全部 IE 及其他理想假设下给出 $V(t)\le e^{-\varpi(t-T_e)}V(T_e)+(C_\zeta/\varpi)(1-e^{-\varpi(t-T_e)})$。本轮接受残差相关最终半径，不把非零或局部上升的权重误差当作理论失效。IE时间与冻结残差保留在 summary 和离线诊断中。', '',
              '有限仿真中的有界轨迹与该结论相容，但本轮没有据此宣布验证了所有初值的完整指数包络。MuJoCo weld 的有限顺从性、采样保持输入和显式学习离散是理想刚性/连续 Filippov 模型的数值近似。', '',
              '## 半步长复核', '']
    ie_lines = ['## IE及历史残差', '',
                '| 工况 | IE时刻 左/右 s | 冻结总残差范数 左/右 |',
                '|---|---|---|']
    for name, item in current.items():
        times = item['summary']['ie_detection_times_s']
        ie = ' / '.join('未检测' if value is None else f'{value:g}' for value in times)
        ie_lines.append(f'| {name} | {ie} | {fmt_pair(item["frozen_total_regression_residual_norms"])} |')
    ie_lines += ['', '冻结残差不是N或Nm，而是变换回归后的历史积分量；非零时应使用论文的残差相关最终界。各组residual_diagnosis.json以实际保持力矩左右极限事后重建反力贡献，没有将这些反力输入学习器。', '']
    half_index = lines.index('## 半步长复核')
    lines[half_index:half_index] = ie_lines
    if fine:
        fine_v = json.loads((fine_path/'validation.json').read_text())
        pose = fine_v.get('step_comparison', {})
        learning = fine_v.get('original_learning_step_comparison', {})
        a, b = current['nominal']['force']['final_two_seconds'], fine['force']['final_two_seconds']
        lines += [f'名义半步长控制周期 {fine["summary"]["control_dt_s"]:g}s，物理步长 {fine["summary"]["physics_dt_s"]:g}s，独立验证 {"PASS" if fine["validation_passed"] else "FAIL"}。', '',
                  f'- 三维力误差RMS：主步长 {a["force_error_vector_rms_N"][0]:.6g} N，半步长 {b["force_error_vector_rms_N"][0]:.6g} N。',
                  f'- 三维力AC RMS：主步长 {a["force_ac_vector_rms_N"][0]:.6g} N，半步长 {b["force_ac_vector_rms_N"][0]:.6g} N。',
                  f'- 广义力矩误差RMS：主步长 {a["moment_error_vector_rms_Nm"][0]:.6g} Nm，半步长 {b["moment_error_vector_rms_Nm"][0]:.6g} Nm。']
        if pose:
            lines.append(f'- 共同区间位置差RMS {pose["position_difference_rms_mm"]:.6g} mm，姿态差RMS {pose["orientation_coordinate_difference_rms_deg"]:.6g} deg。')
        if learning:
            lines.append(f'- 终点权重向量差范数 左/右 {fmt_pair(np.linalg.norm(learning["final_common_parameter_difference"], axis=1))}。')
        lines += ['', 'sign 受力波动可能对控制采样周期敏感；步长复核的实际差异全部保留，不声称减小步长必定消除纹波。']
    else:
        lines.append('半步长结果尚未完成。')
    lines += ['', '## 文件', '',
              '- `figures/<case>/02_errors.pdf`：跟踪误差与非对称上下边界。',
              '- `figures/<case>/05_learning.pdf`：权重、真值、先验盒和原始IE。',
              '- `figures/<case>/06_internal_force.pdf`：内力误差及真实波动。',
              '- `figures/<case>/08_parameter_error_diagnostics.pdf`：权重误差及末段变化。',
              '- `nominal/video.mp4`：24s实际物理轨迹离线回放。',
              '- `validation_summary.json`、`performance_assessment.json`：独立公式/物理与性能审查。',
              '- `tuning_summary.json`：固定参数候选及选择依据；先导仅保留小摘要，正式数据完整保留。',
              '- `../configs/original_low_force.json`：本轮正式参数。',
              '- `../docs/original_low_force_audit.md`：逐公式一致性审查。']
    (results/'原方法一致性与内力复核报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key:aggregate[key] for key in ('core_validations_passed','finite_weights_in_prior','force_target_and_reference_unchanged')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE/'results_original_low_force')
    parser.add_argument('--baseline', type=Path, default=HERE/'results_original_retrial')
    args = parser.parse_args()
    generate(args.results.resolve(), args.baseline.resolve())
