#!/usr/bin/env python3
"""Report actual truth errors and data-driven exponential identification.

Truth is loaded exclusively from saved offline evaluation records. This
module neither changes parameters nor feeds information into a rollout.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = ('nominal', 'nonmonotonic', 'large_initial_error')
NAMES = {'nominal': '名义跟踪', 'nonmonotonic': '非单调边界',
         'large_initial_error': '大初始误差'}


def read_case(directory):
    summary = json.loads((directory/'summary.json').read_text())
    with np.load(directory/'trajectory.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    with np.load(directory/'learning_state.npz', allow_pickle=False) as saved:
        learning = {key: saved[key] for key in saved.files}
    validation = json.loads((directory/'validation.json').read_text())
    times, theta = arrays['t'], arrays['theta']
    truth = learning['true_parameters_offline_only']
    rate = float(summary['settings']['identification_rate'])
    metrics = dict(case=summary['case'], validation_passed=validation['passed'],
                   validation_checks=len(validation['checks']), rate_s_inv=rate,
                   initial_parameter_error_norms=np.linalg.norm(theta[0]-truth, axis=1).tolist(),
                   final_parameter_error_norms=np.linalg.norm(theta[-1]-truth, axis=1).tolist(),
                   final_maximum_parameter_component_errors=np.max(np.abs(theta[-1]-truth), axis=1).tolist(),
                   final_maximum_relative_parameter_component_error_percent=float(
                       np.max(np.abs((theta[-1]-truth)/truth))*100),
                   data_target_error_norms=np.linalg.norm(learning['data_target']-truth, axis=1).tolist(),
                   numerical_truth_accuracy_target_l2=1e-4,
                   numerical_truth_accuracy_met=bool(np.all(np.linalg.norm(theta[-1]-truth, axis=1) <= 1e-4)),
                   post_freeze_flow_maximum_errors=[], post_freeze_fitted_rates_s_inv=[],
                   truth_error_envelope_excess=[], first_stays_below_1e_3_s=[])
    truth_error = np.linalg.norm(theta-truth, axis=2)
    for arm, freeze in enumerate(summary['ie_detection_times_s']):
        if freeze is None:
            raise ValueError(f'{directory}: IE was never observed.')
        active = times >= freeze
        decay = np.exp(-rate*(times[active]-freeze))
        target = learning['data_target'][arm]
        initial = learning['theta_at_freeze'][arm]
        expected = target+decay[:, None]*(initial-target)
        metrics['post_freeze_flow_maximum_errors'].append(float(np.max(np.abs(theta[active, arm]-expected))))
        target_error = np.linalg.norm(theta[active, arm]-target, axis=1)
        fit = (target_error > 1e-8) & (target_error < .95*np.linalg.norm(initial-target))
        metrics['post_freeze_fitted_rates_s_inv'].append(float(-np.polyfit(times[active][fit],
                                                                np.log(target_error[fit]), 1)[0]))
        envelope = (decay*np.linalg.norm(initial-truth[arm])
                    +(1-decay)*np.linalg.norm(target-truth[arm]))
        metrics['truth_error_envelope_excess'].append(float(np.max(truth_error[active, arm]-envelope)))
        suffix = np.maximum.accumulate(truth_error[::-1, arm])[::-1]
        indices = np.flatnonzero((times >= freeze) & (suffix <= 1e-3))
        metrics['first_stays_below_1e_3_s'].append(float(times[indices[0]]) if len(indices) else None)
    final = times >= times[-1]-2.
    force_error = arrays['internal_force'][final]-arrays['desired_internal_force'][final]
    centered = force_error-force_error.mean(axis=0)
    metrics['internal_force_final_2s_bias'] = force_error.mean(axis=0).tolist()
    metrics['internal_force_final_2s_rms'] = np.sqrt(np.mean(force_error**2, axis=0)).tolist()
    metrics['internal_force_final_2s_ac_rms'] = np.sqrt(np.mean(centered**2, axis=0)).tolist()
    metrics['internal_force_final_2s_peak_to_peak'] = np.ptp(force_error, axis=0).tolist()
    metrics['final_2s_position_rmse_mm'] = float(np.sqrt(np.mean(np.sum(arrays['error'][final, :3]**2, axis=1)))*1000)
    metrics['final_2s_orientation_rmse_deg'] = float(np.sqrt(np.mean(np.sum(arrays['error'][final, 3:]**2, axis=1)))*180/np.pi)
    metrics['artifacts_sha256'] = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                  for name in ('trajectory.npz', 'learning_state.npz', 'summary.json', 'source_manifest.json')}
    return summary, learning, metrics


def pair(values, precision=6):
    return ' / '.join(f'{v:.{precision}g}' for v in values)


def make_report(results):
    results = Path(results).resolve()
    cases = {name: read_case(results/name) for name in CASES}
    if not all(case[2]['validation_passed'] for case in cases.values()):
        raise ValueError('Every main case must pass independent validation before reporting success.')
    s0, l0, m0 = cases['nominal']
    settings = s0['settings']
    lines = ['# 权重指数辨识实验：测力补偿与历史数据预条件扩展', '',
             '**这是一项明确修改学习结构和观测条件的扩展，不是仅调增益，也不是原论文无力传感学习律的逐字复现。**', '',
             '保留双 CR7、真实 MuJoCo 积分、八参数物理含义、先验盒及现有 JAS 运动控制结构。'
             '新增理想虚拟腕端反力和实际对象位姿/速度观测；两臂共享对象动量观测。'
             '辨识回归由机器人七参数与对象单参数分块组成。'
             '冻结后用纯历史 prediction 与正定数据预条件更新，替换原学习式的跟踪驱动项 `−YS`。', '',
             '## 可证明的收敛关系', '',
             '由实测动量滤波生成 Φ、y，计算真实历史矩阵 `G=∫ΦΦᵀdt`、`p=∫Φy dt`。'
             'IE 检查不添加正对角、不截断奇异值；同时检查由数据解出的目标位于先验盒内。冻结后使用：', '',
             '```text', 'dot(theta_hat) = k * solve(G, p-G*theta_hat)',
             'theta_data = solve(G,p)',
             'theta_hat(t)-theta_data = exp[-k(t-Te)] * [theta_hat(Te)-theta_data]',
             '```', '',
             f'本次 `k={settings["identification_rate"]:g} s⁻¹`，参数误差时间常数为 '
             f'{1/settings["identification_rate"]:.6g} s。无偏观测与精确连续回归下，'
             '`p=G theta_true`，因而上述关系就是到真实参数的指数收敛。'
             '这里“无偏”指该冻结窗的总存储残差精确为零；仅有噪声统计均值为零，不能保证单次有限窗满足该等式。'
             '有测量/数值偏差 ζ 时，`theta_data−theta_true=solve(G,ζ)`；预条件不会消除偏差。'
             '本报告分别给出到数据目标的指数收缩和到真值的剩余误差，不能将两者混称。', '',
             '严格证明、传感假设及与原方法的区别见 `../docs/exponential_identification_design.md`。'
             '生产学习器不读取真值、真实质量/惯量或 qacc；真值首次在物理试验结束后用于评估。', '',
             '## 固定实验配置', '',
             f'- 三组均运行 {s0["duration_s"]:g} s；六维多正弦参考 `moderate`，'
             f'0–2 s 平滑起动，{settings["excitation_duration"]-2:g}–{settings["excitation_duration"]:g} s 平滑停止，随后保持。',
             f'- 物理积分器 `{s0["integrator"]}`，控制周期 {s0["control_dt_s"]:g} s，'
             f'物理步长 {s0["physics_dt_s"]:g} s；没有用参考轨迹覆盖真实状态。',
             f'- 动量滤波率 {settings["filter_rate"]:g} s⁻¹；历史窗 {settings["identification_window"]:g} s；'
             f'归一化真实 Gram 最小特征值阈值 {settings["identification_threshold"]:g}。',
             f'- 初估全为 0.9；先验盒 [{settings["prior_lower"]:g},{settings["prior_upper"]:g}]。'
             '初估及合格数据目标均在盒内，指数流保持在盒内。',
             f'- 沿用边界层降抖运动反馈，epsilon={settings["robust_width"]:g}；'
             '不继承原 sign 控制的精确固定时间到达结论。', '',
             '参数记录继承了原控制器的配置字段，但旧 `window/omega/adaptation_gain/gamma` '
             '不参与新辨识更新；新辨识实际采用 `identification_window/identification_threshold/identification_rate` '
             '以及共同的 `filter_rate`、先验盒。', '',
             '## 实际权重误差', '',
             '以下均为八维权重误差二范数，左右臂用“左 / 右”表示。真值仅用于离线评估。', '',
             '| 工况 | 初始误差 | IE 时刻 s | 20 s 真值误差 | 数据目标误差底限 | 最大分量相对误差 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, (s, learning, m) in cases.items():
        lines.append(f'| {NAMES[name]} | {pair(m["initial_parameter_error_norms"])} | '
                     f'{pair(s["ie_detection_times_s"])} | {pair(m["final_parameter_error_norms"])} | '
                     f'{pair(m["data_target_error_norms"])} | '
                     f'{m["final_maximum_relative_parameter_component_error_percent"]:.6g}% |')
    lines += ['', '到数据目标的指数关系由原始权重逐点核验，而不是根据对数图直线拟合后宣称。'
              '拟合速率只用于额外交叉检查。', '',
              '| 工况 | 指数流逐分量最大偏差（左 / 右） | 实测对数斜率 s⁻¹（左 / 右） | 真值误差持续小于 0.001 的时刻 s |',
              '|---|---:|---:|---:|']
    for name, (s, learning, m) in cases.items():
        lines.append(f'| {NAMES[name]} | {pair(m["post_freeze_flow_maximum_errors"])} | '
                     f'{pair(m["post_freeze_fitted_rates_s_inv"])} | '
                     f'{pair(m["first_stays_below_1e_3_s"])} |')
    accuracy_pass = all(item[2]['numerical_truth_accuracy_met'] for item in cases.values())
    lines += ['', f'本轮另设数值精度验收：每臂最终八维真值误差范数不超过 `1e-4`。'
              f'三组结果：**{"全部达到" if accuracy_pass else "未全部达到，需继续改进"}**。'
              '该工程精度验收与无偏理论的零误差极限是不同概念。', '',
              '**有限步长下最终误差非零，因此不能写成“仿真已经严格收敛到数学真值零误差”。**'
              '准确表述是：在明确的无偏条件下有真值指数收敛证明；实际 MuJoCo 实验指数趋近到真值附近的数值底限。'
              '这里的“底限”指渐近误差范数，不是全过程不可穿越的下界；短暂抵消可使瞬时误差更小。', '',
              '## 名义工况八参数逐项核对', '',
              '| 参数 | 左真值 | 左最终估计 | 左绝对误差 | 右真值 | 右最终估计 | 右绝对误差 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    labels = [f'连杆{i}密度比例' for i in range(1, 7)]+['公共黏性比例', '负载密度比例']
    truth = l0['true_parameters_offline_only']
    final_theta = np.asarray(s0['final_parameter_estimates'])
    for j, label in enumerate(labels):
        values = [truth[0,j], final_theta[0,j], abs(final_theta[0,j]-truth[0,j]),
                  truth[1,j], final_theta[1,j], abs(final_theta[1,j]-truth[1,j])]
        lines.append('| '+label+' | '+' | '.join(f'{v:.10g}' for v in values)+' |')
    lines += ['', '## 跟踪、内力与独立验证', '',
              '| 工况 | 最后2s位置RMSE mm | 最后2s姿态RMSE ° | 左Fy误差RMS N | 左Fy去均值纹波RMS N | 验证 |',
              '|---|---:|---:|---:|---:|---|']
    for name, (s, learning, m) in cases.items():
        lines.append(f'| {NAMES[name]} | {m["final_2s_position_rmse_mm"]:.6g} | '
                     f'{m["final_2s_orientation_rmse_deg"]:.6g} | '
                     f'{m["internal_force_final_2s_rms"][0][1]:.6g} | '
                     f'{m["internal_force_final_2s_ac_rms"][0][1]:.6g} | '
                     f'{m["validation_checks"]} 项 PASS |')
    lines += ['', '独立验证覆盖：源文件/模型资产 SHA256、原始动量及输入滤波重建、未正则化 Gram/p 重积分、'
              'IE 首次触发、指数流与真值底限、逐 tick 独立 FK/Jacobian/边界/限位、实际物理积分片段重放、'
              '抽样瞬时动量恒等式及冻结后的留出数据残差。'
              'qacc 只在离线验证 oracle 中使用，不进入控制或学习。', '']
    metrics = {name: item[2] for name, item in cases.items()}
    fine_dir = results/'step_refinement'/'nominal'
    if (fine_dir/'validation.json').exists():
        fine_s, fine_l, fine_m = read_case(fine_dir)
        if not fine_m['validation_passed']:
            raise ValueError('Step refinement failed independent validation.')
        metrics['step_refinement_nominal'] = fine_m
        lines += ['## 半步长复算', '',
                  f'控制周期及物理步长均减半至 {fine_s["control_dt_s"]:g} / {fine_s["physics_dt_s"]:g} s，'
                  f'仍为 RK4、同一参考、增益、数据窗与初值。独立验证 {fine_m["validation_checks"]} 项全部通过。', '',
                  f'- 主实验真值误差范数：{pair(m0["final_parameter_error_norms"])}。',
                  f'- 半步长真值误差范数：{pair(fine_m["final_parameter_error_norms"])}。',
                  f'- 半步长最大分量相对误差：{fine_m["final_maximum_relative_parameter_component_error_percent"]:.6g}%。', '',
                  '步长敏感性和原始滤波回归残差支持将剩余量解释为有限步长/滤波精度底限；'
                  '这不意味着真实力传感器带噪声时仍可无条件精确辨识。', '']
    lines += ['## 图与原始数据', '',
              '- `figures/nominal/05_learning.pdf`：八个参数、真实值、先验盒和正确的新 IE 判据。',
              '- `figures/nominal/08_exponential_identification.pdf`：真值误差、数据目标误差、理论指数和数值底限。',
              '- `figures/<case>/01_trajectory.pdf` 与 `02_errors.pdf`：真实物理跟踪及边界。',
              '- 每组 `trajectory.npz`、`learning_state.npz`、`summary.json`、`source_manifest.json`、`validation.json` 可复查。',
              '- `identification_metrics.json` 保存本报告的全部机器可读统计与原始数据校验值。', '',
              '原论文原式和此前降抖结果仍分别保存在 `../results/`、`../results_low_chatter/`，'
              '它们不是本扩展的真值指数辨识结果。调参试跑只用于诊断，正式交付不保留临时原始轨迹。', '']
    (results/'权重指数辨识报告.md').write_text('\n'.join(lines))
    (results/'identification_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n')
    return results/'权重指数辨识报告.md'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE/'results_exponential')
    print(make_report(parser.parse_args().results))


if __name__ == '__main__':
    main()
