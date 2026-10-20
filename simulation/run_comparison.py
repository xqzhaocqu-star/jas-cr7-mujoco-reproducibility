#!/usr/bin/env python3
"""Full MuJoCo forward-dynamics comparisons with failures retained.

Reaction forces and plant truth are evaluation only. Existing models and the
original controller/learner are not edited. All trial configurations are frozen
in protocol.json before production; termination is recorded, never discarded.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from jas.dynamics import LinkDynamics
from run_experiments import HERE, CASES, initialize, kinematics
from design_excitation import multisine_reference
from comparison.controller import ComparisonController, ComparisonSettings, common_bounds, METHODS
from comparison.metrics import tracking_entry_metrics, force_statistics, torque_statistics, sustained_entry_time


def reference_for(scenario):
    if scenario.get('reference_profile') == 'hold':
        return lambda t, center: (np.asarray(center).copy(), np.zeros(6), np.zeros(6))
    return lambda t, center: multisine_reference(t, center, scenario.get('excitation_duration', 20.), 'moderate')


def controllers_for(method, settings, kins, center, payload, nonmonotonic, reference, external_settings=None):
    cls = ComparisonController
    if method == 'external_ppc':
        from comparison.external_ppc import ExternalPPC
        cls = ExternalPPC
    return [cls(kins[i], LinkDynamics(HERE/'models/arm_nominal.xml'), center,
                payload['mass_kg'], payload['diagonal_inertia_kg_m2'], i,
                settings, nonmonotonic, reference=reference,
                **({'external_settings': external_settings} if method == 'external_ppc' else {})) for i in range(2)]


def source_manifest():
    paths = [Path(__file__), HERE/'run_comparison_all.py', HERE/'run_experiments.py', HERE/'design_excitation.py',
             HERE.parent/'REPRODUCIBILITY.md']
    for folder in ('jas', 'original_retrial', 'comparison'):
        paths.extend(sorted((HERE/folder).glob('*.py')))
    paths.extend(sorted((HERE/'models').glob('*.xml')))
    paths.extend(HERE/'models'/f for f in ('scene_metadata.json', 'plant_truth.json', 'asset_sha256.json'))
    return {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_trial(method, scenario, output, settings_dict, control_dt=.0005, physics_dt=.000125,
              external_settings=None, progress=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'source_manifest.json').exists():
        raise FileExistsError(f'Refusing to overwrite completed or terminated trial: {output}')
    method_settings = dict(settings_dict, method=method)
    cfg = ComparisonSettings(**method_settings)
    duration = float(scenario.get('duration_s', 24.))
    ratio = round(control_dt/physics_dt)
    if ratio < 1 or abs(ratio*physics_dt-control_dt) > 1e-12:
        raise ValueError('Control interval must contain an integer number of physics steps.')
    nsteps = round(duration/control_dt)
    if abs(nsteps*control_dt-duration) > 1e-10:
        raise ValueError('Duration must align to control grid.')
    meta = json.loads((HERE/'models/scene_metadata.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(HERE/'models/dual_cr7.xml'))
    model.opt.timestep = physics_dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    data = mujoco.MjData(model)
    kins = [kinematics(meta, hand) for hand in ('left', 'right')]
    center = np.r_[meta['payload_nominal']['initial_position_world_m'], np.zeros(3)]
    start = initialize(model, data, meta, kins, center, scenario['offset'])
    reference = reference_for(scenario)
    ctls = controllers_for(method, cfg, kins, center, meta['payload_nominal'],
                           scenario.get('nonmonotonic', False), reference, external_settings)
    limits = np.tile(meta['torque_limits_Nm'], 2)
    speeds = np.tile(meta['joint_velocity_limits_rad_s'], 2)
    positions = np.tile(meta['joint_position_limits_rad'], (2, 1))
    payload_body = model.body('payload').id
    rows = []
    started, next_progress = time.perf_counter(), 0.
    status, reason, stage, attempted = 'completed', None, None, None
    min_joint, max_speed, max_contact = float('inf'), 0., 0
    failure_state = None
    for step in range(nsteps+1):
        t = step*control_dt
        try:
            commands, infos = zip(*(ctl.command(t, data.qpos[6*i:6*i+6], data.qvel[6*i:6*i+6])
                                   for i, ctl in enumerate(ctls)))
            command = np.concatenate(commands)
            if not np.isfinite(command).all():
                raise FloatingPointError('Controller produced nonfinite torque.')
        except Exception as exc:
            status, reason, stage, attempted = 'terminated', f'{type(exc).__name__}: {exc}', 'command', t
            failure_state = dict(t=t, qpos=data.qpos.copy(), qvel=data.qvel.copy())
            break
        adapted = hasattr(ctls[0], 'learner')
        theta_control = (np.array([c.learner.theta.copy() for c in ctls]) if adapted
                         else np.full((2, 8), np.nan))
        applied = np.clip(command, -limits, limits)
        known_left, known_right = [], []
        for i, ctl in enumerate(ctls):
            previous = getattr(ctl, 'previous_applied', None)
            held = applied[6*i:6*i+6] if previous is None else previous
            info = infos[i]
            known_left.append(info['A'].T@(held+info['jacobian'].T@ctl.fd))
            known_right.append(info['A'].T@(applied[6*i:6*i+6]+info['jacobian'].T@ctl.fd))
        data.ctrl[:] = applied
        try:
            for i, ctl in enumerate(ctls):
                ctl.update(t, applied[6*i:6*i+6], infos[i])
        except Exception as exc:
            status, reason, stage, attempted = 'terminated', f'{type(exc).__name__}: {exc}', 'learning_update', t
            failure_state = dict(t=t, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                                 failed_torque_command=command, failed_torque_applied=applied)
            break
        # Online control and learning are finished before this diagnostics block.
        mujoco.mj_forward(model, data)
        actual = np.r_[data.xpos[payload_body], Rotation.from_matrix(
            data.xmat[payload_body].reshape(3, 3)).as_euler('xyz')]
        ref = reference(t, center)[0]
        forces = np.array([np.linalg.solve(info['jacobian'].T, data.qfrc_constraint[6*i:6*i+6])
                           for i, info in enumerate(infos)])
        internal = forces-.5*np.sum(forces, axis=0)
        upper, lower_mag, *_ = common_bounds(t, cfg, scenario.get('nonmonotonic', False))
        local_error = np.array([i['error'] for i in infos])
        margins = np.minimum((upper-local_error)/upper, (lower_mag+local_error)/lower_mag)
        actual_margin = np.minimum((upper-(actual-ref))/upper, (lower_mag+(actual-ref))/lower_mag)
        rows.append(dict(t=t, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                         object_pose=actual, reference=ref, error=actual-ref, local_error=local_error,
                         torque_command=command, torque_applied=applied,
                         surface=np.array([i['surface'] for i in infos]), xi=np.array([i['xi'] for i in infos]),
                         upper=upper, lower=-lower_mag, barrier_margin=margins, actual_barrier_margin=actual_margin,
                         native_upper=np.array([i['upper'] for i in infos]),
                         native_lower=np.array([i['lower'] for i in infos]),
                         native_barrier_margin=np.array([i['barrier_margin'] for i in infos]),
                         sigma_min=np.array([i['sigma_min'] for i in infos]),
                         theta=np.array([c.learner.theta.copy() for c in ctls]) if adapted else np.full((2,8), np.nan),
                         theta_control=theta_control,
                         filtered_regressor=np.array([c.learner.filtered_regressor for c in ctls]) if adapted else np.full((2,8,6), np.nan),
                         filtered_torque=np.array([c.learner.tauf for c in ctls]) if adapted else np.full((2,6), np.nan),
                         lambda_min=np.array([c.learner.minimum_eigenvalue for c in ctls]) if adapted else np.full(2, np.nan),
                         prediction_error=np.array([c.learner.epsilon for c in ctls]) if adapted else np.full((2,8), np.nan),
                         regressor=np.array([i['regressor'] for i in infos]),
                         momentum=np.array([i['momentum'] for i in infos]),
                         remainder=np.array([i['remainder'] for i in infos]),
                         A=np.array([i['A'] for i in infos]),
                         learning_drive=np.array([i['regressor']@i['surface'] for i in infos]),
                         known_torque_left=np.array(known_left), known_torque_right=np.array(known_right),
                         internal_force=internal, interaction_force=forces,
                         desired_internal_force=np.array([c.fd for c in ctls]),
                         weld_residual=data.efc_pos[:12].copy(), contact_count=data.ncon,
                         control_valid_mask=True, barrier_applicable=True))
        if method == 'external_ppc':
            for key in ('ppc_rho_s', 'ppc_rho_v', 'ppc_xi_s', 'ppc_xi_v',
                        'ppc_epsilon_s', 'ppc_epsilon_v', 'ppc_v_reference',
                        'ppc_object_velocity', 'ppc_object_feedback', 'ppc_load_allocation',
                        'ppc_grasp_map', 'ppc_palm_wrench', 'ppc_internal_palm_wrench',
                        'ppc_allocation_identity_error', 'ppc_internal_null_error'):
                if key in infos[0]:
                    rows[-1][key] = np.array([info[key] for info in infos])
        if progress and t >= next_progress:
            print(f'{method}/{scenario["id"]} t={t:g}/{duration:g} '
                  f'error={np.linalg.norm(actual[:3]-ref[:3])*1000:.4g}mm '
                  f'tau={np.max(np.abs(command)/limits):.3g} '
                  f'wall={time.perf_counter()-started:.1f}s', flush=True)
            next_progress = t+4.
        if step == nsteps:
            break
        if np.min([i['sigma_min'] for i in infos]) < .001:
            status, reason, stage, attempted = 'terminated', 'Jacobian singularity guard <0.001', 'domain_guard', t
            failure_state = dict(t=t, qpos=data.qpos.copy(), qvel=data.qvel.copy())
            break
        for substep in range(ratio):
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                status, reason, stage, attempted = 'terminated', 'Nonfinite physics state', 'physics', float(data.time)
                failure_state = dict(t=float(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy())
                break
            expected_time = t+(substep+1)*physics_dt
            bad_warning = any(data.warning[int(kind)].number for kind in (
                mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC))
            if abs(data.time-expected_time)>1e-7 or bad_warning:
                status, reason, stage, attempted = 'terminated', 'MuJoCo bad-state warning or automatic clock reset', 'physics_guard', expected_time
                failure_state = dict(t=float(data.time), expected_time=expected_time,
                                     qpos=data.qpos.copy(), qvel=data.qvel.copy())
                break
            min_joint = min(min_joint, float(np.min(np.minimum(data.qpos[:12]-positions[:,0], positions[:,1]-data.qpos[:12]))))
            max_speed = max(max_speed, float(np.max(np.abs(data.qvel[:12])/speeds)))
            max_contact = max(max_contact, int(data.ncon))
        if status != 'completed':
            break
    if not rows:
        # Keep a zero-length schema and explicit failure state, not a fake command.
        arrays = dict(t=np.empty(0), qpos=np.empty((0,model.nq)), qvel=np.empty((0,model.nv)))
    else:
        arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    np.savez_compressed(output/'trajectory.npz', **arrays)
    if failure_state is not None:
        np.savez_compressed(output/'failure_state.npz', **failure_state)
    # First truth access: physics rollout has ended. It never enters a command.
    true_data = json.loads((HERE/'models/plant_truth.json').read_text())
    truth = np.array([true_data[hand+'_density_multipliers']+[true_data[hand+'_viscous_multiplier'],
                     true_data['payload_density_multiplier']] for hand in ('left','right')])
    adapted = hasattr(ctls[0], 'learner')
    if adapted:
        gram, p = np.array([c.learner.gram for c in ctls]), np.array([c.learner.p for c in ctls])
        residual = p-np.einsum('aij,aj->ai', gram, truth)
        np.savez_compressed(output/'learning_state.npz', gram=gram, p=p,
                            true_parameters_offline_only=truth, stored_residual=residual)
    actual_end = float(arrays['t'][-1]) if rows else 0.
    summary = dict(method=method, case=scenario['id'], scenario=scenario, scenario_id=scenario['id'],
                   requested_duration_s=duration, duration_s=duration, actual_end_s=actual_end,
                   run_status=status, termination_reason=reason, termination_stage=stage,
                   failed_attempt_time_s=attempted, initial_offset=scenario['offset'],
                   initial_object_pose=start.tolist(), has_adaptation=adapted,
                   control_dt_s=control_dt, physics_dt_s=physics_dt, integrator='RK4',
                   settings=asdict(cfg), external_settings=external_settings,
                   controller_configurations=[ctl.configuration() for ctl in ctls],
                   wall_seconds=time.perf_counter()-started,
                   algorithm_variant=('Complete original JAS sign controller and composite learning' if method=='proposed'
                                      else 'Independent published PPC protocol' if method=='external_ppc' else f'Explicit structural ablation: {method}'),
                   comparison_boundary_note='Common finite C2 assessment envelope during first second; proposed internal envelope remains original with mu(0)=ell. Bounds coincide from t=1 s.',
                   sensing='Local encoders and actual held motor torque. No reaction forces, qacc or truth enter control/learning.',
                   minimum_joint_limit_margin_rad=min_joint if np.isfinite(min_joint) else None,
                   max_joint_speed_limit_ratio=max_speed, max_contact_count=max_contact,
                   mujoco_warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(data.warning) if w.number},
                   true_parameters_offline_only=truth.tolist(),
                   ie_detection_times_s=[c.learner.frozen_time for c in ctls] if adapted else [None,None],
                   final_parameter_error_norms=np.linalg.norm(arrays['theta'][-1]-truth,axis=1).tolist() if rows and adapted else None,
                   numerical_projection_counts=[c.learner.numerical_projection_count for c in ctls] if adapted else None,
                   stored_learning_residual_norms=np.linalg.norm(residual,axis=1).tolist() if adapted else None,
                   phase_entry_thresholds=dict(position_m=.00025, orientation_rad=float(np.deg2rad(.05)),
                                               surface_component_abs=.001, xi_component_abs=cfg.delta, hold_s=2.),
                   jas_phase_criteria_applicable=method!='external_ppc',
                   claim_limit='Finite physical-domain and discrete-sample comparison; no empirical claim of global fixed time or zero parameter-error limit.')
    if rows:
        t = arrays['t']
        summary['tracking_entry'] = tracking_entry_metrics(t, arrays['error'], requested_end_s=duration,
                                                           completed=status=='completed')
        summary['surface_entry_s'] = (sustained_entry_time(t, np.max(np.abs(arrays['surface']),axis=(1,2))<=.001,
                                                        requested_end_s=duration, completed=status=='completed')
                                      if method!='external_ppc' else None)
        summary['xi_layer_entry_s'] = (sustained_entry_time(t, np.max(np.abs(arrays['xi']),axis=(1,2))<=cfg.delta,
                                                         requested_end_s=duration, completed=status=='completed')
                                       if method!='external_ppc' else None)
        summary['torque_metrics'] = torque_statistics(t, arrays['torque_command'], arrays['torque_applied'], limits)
        summary['minimum_common_normalized_gap'] = float(arrays['actual_barrier_margin'].min())
        summary['constraint_violating_samples'] = int(np.sum(np.any(arrays['actual_barrier_margin']<0,axis=1)))
        summary['internal_force_metrics_final_2s'] = force_statistics(t, arrays['internal_force'], arrays['desired_internal_force'],
                                                                    interval=(max(0., actual_end-2.),actual_end))
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda value: value.tolist() if isinstance(value, np.ndarray) else float(value))+'\n')
    (output/'source_manifest.json').write_text(json.dumps(source_manifest(),indent=2)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--trial', required=True)
    p.add_argument('--output', type=Path, default=HERE/'results_comparison')
    p.add_argument('--control-dt', type=float)
    p.add_argument('--physics-dt', type=float)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    trial = next(r for r in protocol['trials'] if r['id']==args.trial)
    scenario = next(s for s in protocol['scenarios'] if s['id']==trial['scenario'])
    run_trial(trial['method'], scenario, args.output/trial['path'], protocol['settings'],
              args.control_dt or trial.get('control_dt_s', protocol['control_dt_s']),
              args.physics_dt or trial.get('physics_dt_s', protocol['physics_dt_s']),
              protocol.get('external_settings'))


if __name__ == '__main__':
    main()
