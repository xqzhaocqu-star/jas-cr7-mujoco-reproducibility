#!/usr/bin/env python3
"""Summarize every pre-registered result, without selecting favorable cases."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from comparison.controller import METHODS
from comparison.metrics import force_statistics,tracking_entry_metrics,sustained_entry_time
from plot_comparison import LABELS


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def f(value,digits=4):
    return '—' if value is None else f'{value:.{digits}g}'


def refinement_metrics(row):
    sensitivity=row.get('validation',{}).get('step_sensitivity')
    if not sensitivity:return None
    error=np.asarray(sensitivity['trace_sensitivity']['error']['componentwise_rms_difference'])
    force=sensitivity['matched_final_two_seconds_force_statistics']
    force_difference=np.asarray(sensitivity['trace_sensitivity']['internal_force']['componentwise_rms_difference']).reshape(2,6)
    theta=sensitivity.get('theta_sensitivity')
    return dict(common_interval_s=sensitivity['common_interval_s'],
        position_trace_difference_rms_mm=float(np.linalg.norm(error[:3])*1000),
        orientation_trace_difference_rms_deg=float(np.linalg.norm(error[3:])*180/np.pi),
        main_tail_force_error_rms_N=force['reference']['arms']['left']['force_xyz_N']['error_vector_rms'],
        fine_tail_force_error_rms_N=force['candidate']['arms']['left']['force_xyz_N']['error_vector_rms'],
        main_tail_moment_error_rms_Nm=force['reference']['arms']['left']['euler_moment_xyz_Nm']['error_vector_rms'],
        fine_tail_moment_error_rms_Nm=force['candidate']['arms']['left']['euler_moment_xyz_Nm']['error_vector_rms'],
        maximum_applied_torque_trace_difference_Nm=sensitivity['trace_sensitivity']['torque_applied']['maximum_absolute_difference'],
        force_trace_difference_rms_N_by_arm=np.linalg.norm(force_difference[:,:3],axis=1).tolist(),
        moment_trace_difference_rms_Nm_by_arm=np.linalg.norm(force_difference[:,3:],axis=1).tolist(),
        theta_sensitivity=theta)


def trial_metrics(directory):
    s=json.loads((directory/'summary.json').read_text())
    result=dict(method=s['method'],scenario=s['scenario_id'],run_status=s['run_status'],
                termination_reason=s['termination_reason'],requested_duration_s=s['requested_duration_s'],
                actual_end_s=s['actual_end_s'],has_adaptation=s['has_adaptation'],
                ie_detection_times_s=s['ie_detection_times_s'],
                source_manifest='source_manifest.json',trajectory='trajectory.npz')
    vpath=directory/'comparison_validation.json'
    result['validation']=json.loads(vpath.read_text()) if vpath.exists() else dict(validity_passed=False,status='MISSING')
    with np.load(directory/'trajectory.npz',allow_pickle=False) as a:
        t=a['t'];done=s['run_status']=='completed'
        if not len(t):return result
        main=t>=min(2.,s['requested_duration_s']/2)
        tail=t>=s['requested_duration_s']-2.
        for label,mask in [('after_2s',main),('final_2s',tail),('full',np.ones(len(t),dtype=bool))]:
            if not np.any(mask):
                result[label]=None;continue
            error=a['error'][mask]
            force=force_statistics(t,a['internal_force'],a['desired_internal_force'],(float(t[mask][0]),float(t[mask][-1])))
            arm=force['arms']['left']
            result[label]=dict(interval_s=[float(t[mask][0]),float(t[mask][-1])],
                position_rmse_mm=float(np.sqrt(np.mean(np.sum(error[:,:3]**2,axis=1)))*1000),
                orientation_coordinate_rmse_deg=float(np.sqrt(np.mean(np.sum(error[:,3:]**2,axis=1)))*180/np.pi),
                force_error_rms_N=arm['force_xyz_N']['error_vector_rms'],
                force_error_ac_rms_N=arm['force_xyz_N']['error_ac_vector_rms'],
                force_error_peak_N=arm['force_xyz_N']['sampled_max_error_vector_norm'],
                moment_error_rms_Nm=arm['euler_moment_xyz_Nm']['error_vector_rms'],
                moment_error_ac_rms_Nm=arm['euler_moment_xyz_Nm']['error_ac_vector_rms'],
                force_component_statistics=force)
            if label in ('after_2s','final_2s') and not done:
                result['partial_recorded_'+label]=result[label]
                result[label]=None
        result['entry_sensitivity']={}
        for multiplier in (.5,1.,2.):
            result['entry_sensitivity'][str(multiplier)]=tracking_entry_metrics(t,a['error'],
                position_threshold_m=.00025*multiplier,orientation_threshold_rad=np.deg2rad(.05)*multiplier,
                requested_end_s=s['requested_duration_s'],completed=done)
        result['phase_threshold_sensitivity']={}
        for multiplier in (() if s['method']=='external_ppc' else (.5,1.,2.)):
            result['phase_threshold_sensitivity'][str(multiplier)]=dict(
                surface_entry_s=sustained_entry_time(t,np.max(np.abs(a['surface']),axis=(1,2))<=.001*multiplier,
                    requested_end_s=s['requested_duration_s'],completed=done),
                xi_layer_entry_s=sustained_entry_time(t,np.max(np.abs(a['xi']),axis=(1,2))<=s['settings']['delta']*multiplier,
                    requested_end_s=s['requested_duration_s'],completed=done))
        result['maximum_constraint_violation_m_rad']=np.maximum(np.maximum(a['error']-a['upper'],a['lower']-a['error']),0).max(axis=0)
        result['constraint_violating_tick_count']=int(np.sum(np.any(a['actual_barrier_margin']<0,axis=1)))
        result['minimum_normalized_common_gap']=float(a['actual_barrier_margin'].min())
        violation=a['actual_barrier_margin']<0
        invalid_time=np.flatnonzero(np.any(violation,axis=1))
        result['common_violation_observation_interval_s']=None if not len(invalid_time) else [float(t[invalid_time[0]]),float(t[invalid_time[-1]])]
        result['common_violating_components']=[name for j,name in enumerate(('x','y','z','roll','pitch','yaw')) if np.any(violation[:,j])]
        result['torque']=s['torque_metrics']
        if s['method']=='external_ppc':
            result['native_ppc_diagnostics']=dict(maximum_absolute_normalized_pose_error=float(np.max(np.abs(a['ppc_xi_s']))),
                maximum_absolute_normalized_velocity_error=float(np.max(np.abs(a['ppc_xi_v']))),
                interpretation='Native pose and velocity domains; these differ from the common physical assessment envelope.')
        elif done and np.any(tail):
            surface=np.max(np.abs(a['surface'][tail]),axis=(1,2))/.001
            layer=np.max(np.abs(a['xi'][tail]),axis=(1,2))/s['settings']['delta']
            result['final_two_seconds_phase_diagnostics']=dict(interval_s=[float(t[tail][0]),float(t[tail][-1])],
                surface_global_peak_over_threshold_range=[float(surface.min()),float(surface.max())],
                xi_global_peak_over_delta_range=[float(layer.min()),float(layer.max())],
                surface_all_components_inside_sample_fraction=float(np.mean(surface<=1)),
                xi_all_components_inside_sample_fraction=float(np.mean(layer<=1)),
                interpretation='Method-specific coordinates and sampled thresholds; no claim of exact sliding or common-coordinate ranking.')
        if s['has_adaptation']:
            theta=a['theta'];truth=np.asarray(s['true_parameters_offline_only'])
            result['parameters']=dict(range_by_arm=np.stack([theta.min(axis=(0,2)),theta.max(axis=(0,2))],axis=1),
                initial_true_error_norm=np.linalg.norm(theta[0]-truth,axis=1),
                final_true_error_norm=np.linalg.norm(theta[-1]-truth,axis=1),
                within_prior=bool(np.all(theta>=s['settings']['prior_lower']-1e-12) and np.all(theta<=s['settings']['prior_upper']+1e-12)),
                final_2s_drift_norm=np.linalg.norm(theta[-1]-theta[np.flatnonzero(tail)[0]],axis=1) if done and np.any(tail) else None,
                numerical_projection_counts=s['numerical_projection_counts'],
                frozen_residual_norms=s['stored_learning_residual_norms'])
            if s['scenario'].get('study')=='initial_sweep':
                xi_peak=float(np.max(np.abs(a['xi'])))
                xi_initial=float(np.max(np.abs(a['xi'][0])))
                crossover=(s['settings']['lam1']/s['settings']['lam2'])**(1/(s['settings']['beta']-s['settings']['alpha']))
                result['initial_condition_coverage']=dict(
                    position_error_norm_m=float(np.linalg.norm(a['error'][0,:3])),
                    euler_coordinate_error_norm_rad=float(np.linalg.norm(a['error'][0,3:])),
                    maximum_initial_absolute_xi=xi_initial,
                    maximum_recorded_absolute_xi=xi_peak,
                    maximum_initial_absolute_surface=float(np.max(np.abs(a['surface'][0]))),
                    maximum_recorded_absolute_surface=float(np.max(np.abs(a['surface']))),
                    sliding_outer_branch_crossover_absolute_xi=float(crossover),
                    sliding_outer_branch_crossover_reached=bool(xi_peak>=crossover),
                    initial_outer_power_ratio=float(s['settings']['lam2']/s['settings']['lam1']*xi_initial**(s['settings']['beta']-s['settings']['alpha'])),
                    interpretation='For the full nonlinear surface, the crossover compares the two outer power terms; it is not a physical-error threshold or a theorem validation.')
            # Actual filtered regressor after the reference stops: a held
            # reference alone is not labelled a proof of absence of PE.
            if s['scenario'].get('reference_profile')!='hold':
                start=float(s['scenario'].get('excitation_duration',20.))
                mask=t>=start
                if np.sum(mask)>1:
                    y=a['filtered_regressor'][mask];gt=np.einsum('naik,najk->naij',y,y)
                    g=np.trapz(gt,t[mask],axis=0)
                    result['post_reference_stop_gram']=dict(interval_s=[float(t[mask][0]),float(t[mask][-1])],
                        minimum_eigenvalues=np.linalg.eigvalsh(g)[:,0],
                        note='Finite observed held interval only. This is not a proof or disproof of infinite-time PE.')
    return clean(result)


def evaluation_scope(root,rows):
    """Record coverage and repeated task data without counting them as new evidence."""
    result={}
    design_path=Path(__file__).resolve().parent/'configs/comparison_external_design_record.json'
    if design_path.exists():
        design=json.loads(design_path.read_text());trials=design['trials']
        result['available_external_calibration_record']=dict(
            path='configs/comparison_external_design_record.json',
            trial_count=len(trials),completed_count=sum(t['run_status']=='completed' for t in trials),
            requested_duration_sum_s=sum(t['requested_duration_s'] for t in trials),
            recorded_duration_sum_s=sum(t['actual_end_s'] for t in trials),
            rule=design['rule'])
    full=[r['initial_condition_coverage'] for r in rows
          if r['path'].startswith('initial_sweep/') and r['method']=='proposed']
    if full:
        result['full_initial_sweep']=dict(
            count=len(full),directions='Positive and negative multiples of one fixed mixed six-coordinate offset; zero initial velocity.',
            position_error_norm_range_mm=[min(r['position_error_norm_m'] for r in full)*1000,max(r['position_error_norm_m'] for r in full)*1000],
            euler_coordinate_error_norm_range_deg=[min(r['euler_coordinate_error_norm_rad'] for r in full)*180/np.pi,max(r['euler_coordinate_error_norm_rad'] for r in full)*180/np.pi],
            maximum_initial_absolute_xi=max(r['maximum_initial_absolute_xi'] for r in full),
            maximum_recorded_absolute_xi=max(r['maximum_recorded_absolute_xi'] for r in full),
            maximum_initial_outer_power_ratio=max(r['initial_outer_power_ratio'] for r in full),
            sliding_outer_branch_crossover_absolute_xi=full[0]['sliding_outer_branch_crossover_absolute_xi'],
            any_sliding_outer_branch_crossover_reached=any(r['sliding_outer_branch_crossover_reached'] for r in full),
            maximum_recorded_absolute_surface=max(r['maximum_recorded_absolute_surface'] for r in full))
    first=root/'main/nominal/external_ppc/trajectory.npz'
    second=root/'main/nonmonotonic/external_ppc/trajectory.npz'
    if first.exists() and second.exists():
        with np.load(first,allow_pickle=False) as a,np.load(second,allow_pickle=False) as b:
            changed=[key for key in sorted(set(a.files)|set(b.files))
                     if key not in a.files or key not in b.files or not np.array_equal(a[key],b[key],equal_nan=True)]
        assessment={'upper','lower','barrier_margin','actual_barrier_margin'}
        result['external_nominal_nonmonotonic']=dict(different_arrays=changed,
            all_other_arrays_identical=not bool(set(changed)-assessment),
            interpretation='This controller does not use the common assessment envelope. When the other task inputs are unchanged, the two cases evaluate one physical trajectory against different bounds; they are not independent motion tasks.')
    return clean(result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    args=p.parse_args();root=args.results.resolve();protocol=json.loads((root/'protocol.json').read_text())
    rows=[]
    for trial in protocol['trials']:
        directory=root/trial['path']
        if not (directory/'source_manifest.json').exists():raise FileNotFoundError(f'Missing pre-registered outcome: {directory}')
        row=trial_metrics(directory);row.update(id=trial['id'],path=trial['path'])
        if trial['path'].startswith('step_refinement/'):
            row['refinement_comparison']=refinement_metrics(row)
        rows.append(row)
    scope=evaluation_scope(root,rows)
    result=dict(protocol=protocol,trials=rows,evaluation_scope=scope,
        parameter_exponential_envelope_fully_verified=False,
        theoretical_global_fixed_time_verified_by_finite_sweep=False,
        external_method_source='https://arxiv.org/html/1710.11088v4',
        interpretation='Formula and physics validation is separate from performance. All pre-registered outcomes are included; no hidden truth or reaction feedback; no requirement of zero asymptotic parameter error.')
    (root/'comparison_metrics.json').write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    fields=['id','method','scenario','run_status','actual_end_s','position_rmse_after_2s_mm',
            'orientation_rmse_after_2s_deg','tail_force_error_rms_N','tail_moment_error_rms_Nm',
            'max_raw_torque_limit_ratio','saturated_tick_count','constraint_violating_tick_count',
            'simultaneous_entry_s','left_final_parameter_error','right_final_parameter_error']
    with (root/'comparison_table.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            tail=r.get('final_2s') or {};motion=r.get('after_2s') or {};params=r.get('parameters') or {}
            norms=params.get('final_true_error_norm') or [None,None]
            writer.writerow(dict(id=r['id'],method=r['method'],scenario=r['scenario'],run_status=r['run_status'],actual_end_s=r['actual_end_s'],
                position_rmse_after_2s_mm=motion.get('position_rmse_mm'),orientation_rmse_after_2s_deg=motion.get('orientation_coordinate_rmse_deg'),
                tail_force_error_rms_N=tail.get('force_error_rms_N'),tail_moment_error_rms_Nm=tail.get('moment_error_rms_Nm'),
                max_raw_torque_limit_ratio=r.get('torque',{}).get('maximum_raw_limit_ratio'),
                saturated_tick_count=r.get('torque',{}).get('saturated_tick_count'),constraint_violating_tick_count=r.get('constraint_violating_tick_count'),
                simultaneous_entry_s=r.get('entry_sensitivity',{}).get('1.0',{}).get('simultaneous_entry_s'),
                left_final_parameter_error=norms[0],right_final_parameter_error=norms[1]))
    complete=sum(r['run_status']=='completed' for r in rows)
    valid=sum(r['validation'].get('validity_passed',False) for r in rows)
    missing=sum(r['validation'].get('status')=='MISSING' for r in rows)
    lines=['# 框架对比与结构消融实验报告','',
        f'预先登记 {len(rows)} 组，完整运行 {complete} 组，提前终止 {len(rows)-complete} 组。所有工况保留；公式/动力学验证与性能结果分别报告。','',
        f'独立公式/记录/动力学审查：{valid}/{len(rows)}组通过，{missing}组尚缺审查；不通过或缺审查的结果保留并明确标注，不能作为有效性能证据。','',
        '完整JAS组直接调用原 `OriginalController.command` 与原显式投影复合学习器，保留参数盒直径鲁棒sign项、正rho、−YS跟踪驱动和原始raw-Gram首次IE冻结。消融及外部基准有独立名称，不代替完整原方法。','',
        '## 公平性与适用条件','',
        '各组使用同一双CR7模型、未知参数、物理初态、参考轨迹、±5 N内力目标、力矩限值和采样；控制器不接收真实参数、加速度或实际反力。外部方法使用其原PPC结构和已知抓取几何的负载分配，参数与包络另见协议。','',
        '原JAS内部μ(0)=ell，对应初始无限物理边界。前1 s另外统一有限评价边界，采用C2过渡，从1 s起与原物理边界一致。前1 s满足有限评价界是实测结果，不冒充原BF理论直接保证。经典log-BF消融直接使用该有限边界。非法初值不作为传统BF的失败样本；全部八个扫描初值均在共同合法域内并提前通过IK/关节/Jacobian检查。','',
        '| 初始变换域 | 条件 | 本次怎样评价 |',
        '|---|---|---|',
        '| 原JAS变换 | 对任意有限坐标误差，ell·tanh(chi·e)严格在(−ell,ell)内 | 保留原初始化；机械臂IK、Jacobian及Euler图条件仍需满足 |',
        '| 传统log-BF | 每轴−L(0)<e(0)<U(0) | 只在共同合法域运行；域外记不适用，不记算法失败 |',
        '| 外部两层PPC | 初始位置/姿态及速度误差分别在其原生包络内 | 单独审查两层原生域；共同任务边界只作额外评价 |','',
        '线性滑动面在同一个名义初值按g(xi0)/xi0逐轴一次匹配参考速度，随后固定；log-BF只在e=0,t=0匹配局部斜率。其余共享模块的参数相同。去历史组仍保留−YS与投影。去双幂次到达项组仍保留原鲁棒sign，不能据此宣称它必然不具备固定时间性质。','',
        '外部PPC保持原文两层误差变换、速度参考与负载分配，原生包络为对称单调指数函数；共同非对称/非单调边界只作额外评价，不替代它的内部包络。其逐轴速度包络初值采用原文实验章节的参数类型，与Step II-a的统一范数加裕量初始化明确区分。参数仅用名义设计工况短试固定，全部候选、终止和饱和记录见configs/comparison_external_design_record.json；没有按正式测试结果逐工况重调，也不宣称完成了全局最优或等预算整定。','',
        '本目录的TeX和审稿回复保留作设计草稿。主论文的旧二维仿真节已于2026-10-18整体替换为双CR7实验，正文以simulation/manuscript/simulation_studies.tex为准；修订记录见simulation/manuscript_revision/。本次仿真并未满足原文理想刚性抓持的全部假设，代码/动力学审查通过也不等于所有定理前提已在MuJoCo中严格实现。','',
        '## 三种共同任务','',
        '下表位置、姿态RMSE采用2–24 s；内力/广义力矩误差RMS采用22–24 s。N和Nm分开；显示原始纹波及偏差。提前终止没有完整末段的指标记空。','',
        '| 工况 | 方法 | 运行/审查 | 位置RMSE mm | 姿态RMSE ° | 内力误差RMS N | 力矩误差RMS Nm | 原始扭矩/限额峰值 | 越界tick |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if not r['path'].startswith('main/'):continue
        q=r.get('after_2s') or {};tail=r.get('final_2s') or {}
        audit='PASS' if r['validation'].get('validity_passed') else r['validation'].get('status','FAIL')
        lines.append(f'| {r["scenario"]} | {LABELS[r["method"]]} | {r["run_status"]}/{audit} | {f(q.get("position_rmse_mm"))} | {f(q.get("orientation_coordinate_rmse_deg"))} | {f(tail.get("force_error_rms_N"))} | {f(tail.get("moment_error_rms_Nm"))} | {f(r.get("torque",{}).get("maximum_raw_limit_ratio"))} | {r.get("constraint_violating_tick_count","—")} |')
    calibration=scope.get('available_external_calibration_record',{})
    if calibration:
        lines.extend(['',f'外部PPC的设计记录包含{calibration["trial_count"]}次名义短试，请求时长合计{f(calibration["requested_duration_sum_s"],7)} s、实际记录{f(calibration["recorded_duration_sum_s"],7)} s，{calibration["completed_count"]}次完整、{calibration["trial_count"]-calibration["completed_count"]}次终止。这是已披露的工程整定预算，不能称各方法按等预算搜索得到最优参数。',''])
    reuse=scope.get('external_nominal_nonmonotonic',{})
    if reuse.get('all_other_arrays_identical'):
        lines.extend(['外部PPC的名义/非单调工况除共同评价边界及其裕量外，全部原始数组相同。原因是两者物理任务相同，而该方法不读取共同边界；应解释为同一运动对两套评价界的检验，不能计为两条独立运动难度证据。',''])
    lines.extend(['','## 这些对比实际支持什么','',
        '消融用于检验所替换结构在当前任务与固定参数下的影响，不能推出该结构是所有控制器取得性能的唯一方式。完整方法也不应被描述成在每项指标上都优于所有对照。',''])
    for scenario in ('nominal','nonmonotonic','large_initial_error'):
        group={r['method']:r for r in rows if r['path'].startswith('main/') and r['scenario']==scenario}
        full,no_history=group['proposed'],group['no_history']
        if not (full['validation'].get('validity_passed') and no_history['validation'].get('validity_passed')):
            lines.extend([f'{scenario}的完整/去历史组尚未同时通过审查，暂不据此形成性能结论。','']);continue
        a,b=full.get('after_2s'),no_history.get('after_2s')
        if a is None or b is None:continue
        p=np.asarray(full['parameters']['final_true_error_norm'])
        q=np.asarray(no_history['parameters']['final_true_error_norm'])
        improvement=(1-p/q)*100
        position_change=(1-a['position_rmse_mm']/b['position_rmse_mm'])*100
        lines.extend([f'{scenario}：与去历史项相比，完整CL的末端权重误差范数左/右降低{f(improvement[0],3)}%/{f(improvement[1],3)}%，位置RMSE降低{f(position_change,3)}%；但姿态RMSE为{f(a["orientation_coordinate_rmse_deg"])}°，对照为{f(b["orientation_coordinate_rmse_deg"])}°，不能概括成跟踪和学习均全面更好。',''])
    lines.extend(['受力改善同样需要分工况、分N与Nm判断；较小的去均值纹波不能代替较小的实际偏差或总误差。正文主表保留这些取舍。权重是否有界由实际记录与先验投影共同核查，不因权重曲线仍变化或误差非零就称发散。',''])
    for r in rows:
        if not r['path'].startswith('main/') or r['method']!='external_ppc':continue
        native=r.get('native_ppc_diagnostics',{})
        interval=r.get('common_violation_observation_interval_s')
        when='无共同边界越界' if interval is None else f'共同边界越界观察区间{f(interval[0],6)}–{f(interval[1],6)} s，分量{", ".join(r["common_violating_components"])}'
        lines.extend([f'外部PPC，{r["scenario"]}：原生归一化姿态/速度误差最大值分别为{f(native.get("maximum_absolute_normalized_pose_error"))}/{f(native.get("maximum_absolute_normalized_velocity_error"))}；{when}。原生域与额外共同要求分别评价。',''])
    lines.extend(['','## 初值扫描与固定时间机理','',
        '八个共同可行初值为原large偏移的±0.25、±0.5、±1、±1.5倍，零初速、静态目标、8 s观察；Full/线性滑动面/去双幂次到达三组使用同一物理初态。主判据为位置误差范数≤0.25 mm且Euler坐标误差范数≤0.05°，进入后必须持续到终点且至少2 s。半倍/双倍容差结果在comparison_metrics.json中。未满足者右删失；不提高容差或删除样本。','',
        '| 初值 | 方法 | 共同物理容差入层 s | S诊断层入层 s | xi诊断层入层 s |',
        '|---|---|---:|---:|---:|'])
    for r in rows:
        if not r['path'].startswith('initial_sweep/'):continue
        e=r.get('entry_sensitivity',{}).get('1.0',{});phase=r.get('phase_threshold_sensitivity',{}).get('1.0',{})
        lines.append(f'| {r["scenario"]} | {LABELS[r["method"]]} | {f(e.get("simultaneous_entry_s"))} | {f(phase.get("surface_entry_s"))} | {f(phase.get("xi_layer_entry_s"))} |')
    lines.append('')
    for method in ('proposed','linear_surface','linear_reaching'):
        selected=[r for r in rows if r['path'].startswith('initial_sweep/') and r['method']==method]
        physical=[r.get('entry_sensitivity',{}).get('1.0',{}).get('simultaneous_entry_s') for r in selected]
        surface=[r.get('phase_threshold_sensitivity',{}).get('1.0',{}).get('surface_entry_s') for r in selected]
        layer=[r.get('phase_threshold_sensitivity',{}).get('1.0',{}).get('xi_layer_entry_s') for r in selected]
        valid_times=[t for t in physical if t is not None]
        time_range='未在观察期内满足' if not valid_times else f'{f(min(valid_times))}–{f(max(valid_times))} s'
        lines.extend([f'{LABELS[method]}：共同物理容差达标{len(valid_times)}/{len(selected)}，入层时间{time_range}；S诊断阈值持续达标{sum(t is not None for t in surface)}/{len(selected)}，原delta诊断层持续达标{sum(t is not None for t in layer)}/{len(selected)}。',''])
        if method=='proposed':
            phase=[r.get('final_two_seconds_phase_diagnostics',{}) for r in selected]
            ranges=[p['surface_global_peak_over_threshold_range'] for p in phase if 'surface_global_peak_over_threshold_range' in p]
            if ranges:lines.extend([f'Full末6–8 s每个tick的双臂最大|S|为阈值的{f(min(v[0] for v in ranges))}–{f(max(v[1] for v in ranges))}倍；该S诊断未入层不是仅由末位舍入造成。',''])
    lines.extend(['','有限初值扫描只提供数值证据，不证明全局固定时间，也不能把采样S阈值视作理想S=0。独立标量ODE图说明双幂次的统一入层时间界与线性系统的初值依赖；抽象标量大幅值不是机械臂可行初态，标量界也不是机械臂含惯性界的Tr。','',
        '未达到预先固定S/xi诊断阈值时，不以共同物理误差较小代替相应阶段证明，也不在结果出来后放宽阈值。当前离散采样、sign切换与有限刚度weld实验对理想滑动阶段的验证能力应据这些未入层结果限定。','',
        '## 权重、IE和残差','',
        '五个有学习器的组都按真实参数离线计算误差，并检查先验盒内有界性。外部PPC没有此参数估计，记不适用。IE与预测误差下降不能替代真值误差证据；非零冻结残差允许非零最终误差。当前没有构造完整的理论常数域界，因此不声称已验证非平凡的完整指数包络。不同BF/滑动面会改变回归坐标，raw Gram数值不作跨结构激励强弱排名。','',
        '参考停止后的实际滤后回归Gram另行统计；参考停止本身不等于已证明不满足PE。','',
        '## 半步长复核','',
        '完整方法与外部完整PPC各复核一次。控制与物理步长同时减半，比较共同记录区间；轨迹接近不能替代对高频内力或学习数值敏感性的检查。','',
        '| 方法 | 共同区间 s | 位置轨迹差RMS mm | 姿态轨迹差RMS ° | 内力RMS 主/细 N | 力矩RMS 主/细 Nm |',
        '|---|---|---:|---:|---:|---:|'])
    coverage=scope.get('full_initial_sweep',{})
    if coverage:
        position=coverage['position_error_norm_range_mm'];angle=coverage['euler_coordinate_error_norm_range_deg']
        coverage_note=(f'八初值仅沿同一混合六维方向取正负四档幅值，初始位置误差范数{f(position[0],6)}–{f(position[1],6)} mm、Euler坐标误差范数{f(angle[0],6)}–{f(angle[1],6)}°。Full初始最大|xi|={f(coverage["maximum_initial_absolute_xi"],6)}，全扫描时域最大|xi|={f(coverage["maximum_recorded_absolute_xi"],6)}；滑动双幂次外支的交叉幅值为{f(coverage["sliding_outer_branch_crossover_absolute_xi"])}，本次没有进入其高幂项主导区。初始高幂/低幂外支项之比最高{f(coverage["maximum_initial_outer_power_ratio"]*100,4)}%。到达变量最大|S|={f(coverage["maximum_recorded_absolute_surface"],6)}，确实经历较大到达幅值；两阶段的覆盖范围应分别解释。')
        insert=lines.index('## 权重、IE和残差')
        lines[insert:insert]=[coverage_note,'']
    refinement_notes=[]
    for r in rows:
        if not r['path'].startswith('step_refinement/'):continue
        q=r.get('refinement_comparison') or {}
        interval=q.get('common_interval_s')
        interval_text='—' if interval is None else f'{f(interval[0])}–{f(interval[1])}'
        lines.append(f'| {LABELS[r["method"]]} | {interval_text} | {f(q.get("position_trace_difference_rms_mm"))} | {f(q.get("orientation_trace_difference_rms_deg"))} | {f(q.get("main_tail_force_error_rms_N"))}/{f(q.get("fine_tail_force_error_rms_N"))} | {f(q.get("main_tail_moment_error_rms_Nm"))}/{f(q.get("fine_tail_moment_error_rms_Nm"))} |')
        theta=q.get('theta_sensitivity') or {}
        if theta.get('applicable'):
            end=theta['final_common_parameter_difference_l2_by_arm']
            maximum=np.asarray(theta['maximum_absolute_parameter_trace_difference']).max()
            refinement_notes.extend([f'{LABELS[r["method"]]} 共同终点权重向量差范数：左{f(end[0])}、右{f(end[1])}；全程最大单参数差{f(maximum)}。逐参数最大/RMS差保存在comparison_metrics.json中。',''])
        if q:
            force=q['force_trace_difference_rms_N_by_arm'];moment=q['moment_trace_difference_rms_Nm_by_arm']
            refinement_notes.extend([f'{LABELS[r["method"]]} 共同细网格上与主步长线性插值的瞬时施加力矩最大差{f(q["maximum_applied_torque_trace_difference_Nm"])} Nm；内力轨迹差RMS左/右{f(force[0])}/{f(force[1])} N，广义力矩轨迹差RMS左/右{f(moment[0])}/{f(moment[1])} Nm。',''])
    lines.extend(['']+refinement_notes+['完整JAS的瞬时sign力矩与内力纹波仍有明显步长敏感性；跟踪和终点权重接近不等于全部高频响应已与步长无关。',''])
    lines.extend(['',
        '## 结果文件','',
        '- 跟踪与边界：figures/<case>/01_tracking_and_bounds.pdf；跟踪范数与裕量：02_tracking_norms_and_margin.pdf。',
        '- 控制代价：03_control_effort.pdf；权重误差：04_learning_comparison.pdf。',
        '- 内力、广义力矩：05_internal_force.pdf、06_internal_moment.pdf。',
        '- 外部方法原生位置/速度包络：07_external_native_funnels.pdf（不与JAS S/xi阶段混用）。',
        '- 初值扫描：figures/07_initial_condition_sweep.pdf；独立机理：08_scalar_fixed_time_mechanisms.pdf。',
        '- 全部原始数据、配置、源SHA及验证：每个trial目录的trajectory.npz、summary.json、source_manifest.json、comparison_validation.json。',
        '- 汇总机器可读数据：comparison_metrics.json、comparison_table.csv；运行前冻结方案：protocol.json。',''])
    (root/'框架对比实验报告.md').write_text('\n'.join(lines))
    tex=['% Standalone replacement simulation text; review before inserting into the manuscript.',
        '% Replace the obsolete simulation section as a whole, including its old model, robust term, projection and equal-search/no-saturation claims.',
        r'\subsection{Component-wise and complete-framework evaluation}',
        'We separate controlled structural ablations from an independent complete-controller comparison. '
        'The full proposed controller preserves the original prior-box sign term and projected composite update, '
        'including the tracking-error drive. Four labelled ablations replace only the error transformation, '
        'the sliding dynamics, the dual-power reaching terms, or the historical prediction term, respectively. '
        'A separate implementation follows the prescribed-performance cooperative controller of Verginis et al. '
        '(arXiv:1710.11088, Section IV-B); it does not reuse the proposed BF or terminal sliding surface.',
        'All methods use the same MuJoCo dual-CR7 plant, uncertainty, reference, initial state, torque limits, '
        'sampling, and internal-force task. The original proposed BF retains its unrestricted initial error '
        'transformation. A finite common assessment envelope is imposed as an additional observed requirement '
        'during the first second, and coincides with its native physical envelope thereafter. '
        'The comparator-specific internal envelopes and admissible initial domains are disclosed.',
        'The controller is implemented with sampled motor commands and a staggered first-order projected '
        'Euler learning update. The plant uses finite-compliance weld constraints. These numerical '
        'approximations do not establish that all ideal rigid-grasp assumptions of the theorems hold exactly.',
        'Eight preregistered feasible initial poses are additionally evaluated with a static target and frozen gains. '
        'Entry requires a position-error norm below 0.25 mm and Euler-coordinate error norm below 0.05 degrees '
        'for every remaining recorded sample, with at least two seconds of observation. '
        'Unsettled and terminated trials remain in the denominator. This finite sweep supports the practical '
        'entry-time assessment but is not a proof of global fixed-time convergence. '
        'Surface-threshold and transformed-layer entry times are reported as separate diagnostics.',
        'Parameter errors are evaluated against offline truth only. Nonzero frozen regression residuals permit '
        'nonzero ultimate errors; no exponential-to-zero fit is imposed. '
        'Tracking, constraint margins, raw and applied motor torques, internal-force errors and generalized '
        'moment errors are reported together to expose control-effort and force trade-offs.',
        r'\begin{table*}[t]\centering\caption{Complete-controller and structural-ablation results (2--24 s tracking; final 2 s internal-force errors).}',
        r'\begin{tabular}{lllrrrr}\hline Scenario & Method & Status & Position (mm) & Euler (deg) & Force (N) & Moment (Nm)\\\hline']
    for r in rows:
        if not r['path'].startswith('main/'):continue
        q=r.get('after_2s') or {};tail=r.get('final_2s') or {}
        tex.append(' & '.join([r['scenario'].replace('_',r'\_'),LABELS[r['method']],r['run_status'],f(q.get('position_rmse_mm')),f(q.get('orientation_coordinate_rmse_deg')),f(tail.get('force_error_rms_N')),f(tail.get('moment_error_rms_Nm'))])+r'\\')
    tex.extend([r'\hline\end{tabular}\end{table*}',
        'The ablations are interpreted component-wise. The full controller is not claimed to dominate '
        'every comparator in every metric. In particular, the historical learning channel can reduce '
        'offline parameter errors while changing attitude tracking, settling behavior, and internal-force '
        'bias or switching. Both benefits and costs are retained in the reported results.',
        'The external controller uses its native symmetric monotone pose and velocity funnels. '
        'All three cases preserve those native domains at the saved samples, but each has 164 violations '
        'of the stricter common pitch envelope, from 1.816 to 1.8975 s, and 83, 83, and 85 saturated '
        'control ticks, respectively. These are retained performance outcomes; they do not refute '
        'the external method under its own ideal assumptions. Because that controller does not '
        'use the common envelope, its nominal and nonmonotonic cases have identical physical '
        'trajectories and differ only in the common-envelope evaluation.',
        r'\begin{table}[t]\centering\caption{Eight feasible static-reference initial conditions. Entries are sampled suffix criteria, not exact sliding events.}',
        r'\begin{tabular}{lrrrr}\hline Method & Physical & Time range (s) & $S$ layer & $\xi$ layer\\\hline'])
    for method in ('proposed','linear_surface','linear_reaching'):
        selected=[r for r in rows if r['path'].startswith('initial_sweep/') and r['method']==method]
        entries=[r.get('entry_sensitivity',{}).get('1.0',{}).get('simultaneous_entry_s') for r in selected]
        valid_times=[t for t in entries if t is not None]
        span='--' if not valid_times else f'{f(min(valid_times))}--{f(max(valid_times))}'
        n_s=sum(r.get('phase_threshold_sensitivity',{}).get('1.0',{}).get('surface_entry_s') is not None for r in selected)
        n_x=sum(r.get('phase_threshold_sensitivity',{}).get('1.0',{}).get('xi_layer_entry_s') is not None for r in selected)
        short={'proposed':'Full JAS','linear_surface':'Linear surface','linear_reaching':'No dual-power'}[method]
        tex.append(f'{short} & {len(valid_times)}/{len(selected)} & {span} & {n_s}/{len(selected)} & {n_x}/{len(selected)}'+r'\\')
    tex.extend([r'\hline\end{tabular}\end{table}',
        'The eight poses are positive and negative multiples of one fixed six-coordinate direction, '
        'rather than a general six-dimensional coverage study. For Full JAS the largest recorded '
        'absolute transformed error over this sweep is 0.83794, below the value 1 at which its '
        'two outer sliding powers are equal for the chosen gains. Thus these robot trials do '
        'not cover dominance of the superlinear sliding term, even though larger reaching-surface '
        'values are observed.',
        'Failure to sustain a preregistered surface or transformed-layer tolerance is not replaced by '
        'a small Cartesian error. The sampled sign feedback and finite-compliance weld plant limit '
        'what these robot data establish about an ideal sliding phase. The separate scalar ODE study '
        'illustrates the dual-power mechanism and is not presented as additional robot evidence or '
        'as a numerical verification of the complete inertia-dependent theorem bound.',
        r'\begin{table*}[t]\centering\caption{Halved control and physics steps: common 0--24 s traces and final-two-second force errors.}',
        r'\begin{tabular}{lrrrr}\hline Method & Position trace diff. (mm) & Euler trace diff. (deg) & Force main/fine (N) & Moment main/fine (Nm)\\\hline'])
    for r in rows:
        if not r['path'].startswith('step_refinement/'):continue
        q=r.get('refinement_comparison') or {}
        tex.append(' & '.join([LABELS[r['method']],f(q.get('position_trace_difference_rms_mm')),f(q.get('orientation_trace_difference_rms_deg')),
            f(q.get('main_tail_force_error_rms_N'))+'/'+f(q.get('fine_tail_force_error_rms_N')),
            f(q.get('main_tail_moment_error_rms_Nm'))+'/'+f(q.get('fine_tail_moment_error_rms_Nm'))])+r'\\')
    tex.extend([r'\hline\end{tabular}\end{table*}',
        'The halved-step test does not establish step-independent switching forces. The full JAS '
        'instantaneous motor and internal-force traces retain numerical sensitivity; complete '
        'trace differences and parameter endpoint differences are disclosed alongside tracking metrics.',''])
    if calibration:
        tex.insert(5,f'The external design record retains {calibration["trial_count"]} nominal pilot trials '
            f'({f(calibration["requested_duration_sum_s"],7)} s requested, '
            f'{f(calibration["recorded_duration_sum_s"],7)} s recorded; '
            f'{calibration["completed_count"]} completed). Its settings were frozen before the registered '
            'test cases. The component ablations use shared parameters and disclosed one-point '
            'matching. We do not claim equal calibration budgets, exhaustive searches, or global '
            'optimality of any tuning.')
    (root/'simulation_section_draft.tex').write_text('\n'.join(tex))
    response=['DRAFT: use only after replacing the obsolete manuscript simulation section and checking figure/table references. The main manuscript has not yet been updated by this report generator.','',
        'Reviewer comment: The shared BF and sliding surface mean the previous comparisons mainly evaluate learning laws.','',
        'Response: We agree and have restructured the experimental evaluation. The shared-framework comparison '
        'is now explicitly identified as a learning-law study. We added controlled ablations of the error '
        'transformation, sliding dynamics, dual-power reaching terms, and historical learning channel, '
        'together with an independent implementation of a complete cooperative prescribed-performance '
        'controller. All methods are evaluated on the same full forward-dynamics dual-CR7 model. '
        'We additionally report a preregistered initial-condition sweep with frozen gains, common physical '
        'entry tolerances, separate phase diagnostics, and right-censored outcomes. '
        'The revised results report constraint margins, motor effort, internal forces and parameter errors '
        'rather than inferring the contribution of every component from learning-law comparisons. '
        'The finite numerical sweep is distinguished from the theoretical fixed-time proof, and '
        'residual-dependent parameter bounds are distinguished from exact convergence to zero.','',
        'We disclose the finite and unequal calibration effort, the restricted direction and amplitude '
        'coverage of the robot initial-condition sweep, the identical physical trajectories of the '
        'external controller under two different assessment envelopes, and the observed saturation '
        'and common-envelope violations. These outcomes limit the claims made from the comparison; '
        'we do not describe it as a validation of every ideal theorem assumption.','',
        'Sources and implementation assumptions: docs/external_ppc_baseline.md; docs/comparative_protocol_audit.md.','']
    (root/'reviewer_response_draft.txt').write_text('\n'.join(response))
    print(f'Wrote {len(rows)} retained outcomes, CSV, report, manuscript text and response draft.',flush=True)


if __name__=='__main__':main()
