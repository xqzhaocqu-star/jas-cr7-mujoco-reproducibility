#!/usr/bin/env python3
"""Full MuJoCo rollout of a declared measured-force identification extension.

JAS motion feedback is retained. The identifier adds ideal grasp-force and
payload pose/twist sensing, shared payload data, and a pure-history SPD
preconditioner after IE. It must not be labelled the unmodified paper method.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from pathlib import Path
import argparse
import hashlib
import json
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from jas.dynamics import LinkDynamics
from jas.reference import desired
from run_experiments import HERE, CASES, initialize, kinematics
from design_excitation import multisine_reference
from exponential_identification.controller import IdentificationController, IdentificationSettings
from exponential_identification.learning import HistoricalIdentifier
from exponential_identification.regression import observe


def reference_for(settings, duration):
    if settings.excitation_profile == 'baseline':
        return desired
    motion_duration = min(duration, settings.excitation_duration)
    return lambda t, center: multisine_reference(t, center, motion_duration, settings.excitation_profile)


def run_case(name='nominal', duration=20., control_dt=.0005, physics_dt=.000125,
             output=None, settings=None, progress=True, integrator='RK4'):
    cfg = settings or IdentificationSettings(robust_mode='boundary_layer', excitation_profile='moderate')
    if min(duration, control_dt, physics_dt, cfg.filter_rate, cfg.identification_rate,
           cfg.identification_window, cfg.identification_threshold) <= 0:
        raise ValueError('Durations, sample periods, filter and identification settings must be positive.')
    if duration < max(4., cfg.identification_window) or cfg.excitation_duration < 4.:
        raise ValueError('The run must include the complete identification window and reference startup/stop.')
    if abs(duration/control_dt-round(duration/control_dt)) > 1e-8:
        raise ValueError('Duration must be an integer number of control periods.')
    output = Path(output or HERE/'results_exponential'/name)
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((HERE/'models/scene_metadata.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(HERE/'models/dual_cr7.xml'))
    model.opt.timestep = physics_dt
    integrators = {'implicitfast': mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                   'RK4': mujoco.mjtIntegrator.mjINT_RK4}
    model.opt.integrator = integrators[integrator]
    ratio = round(control_dt/physics_dt)
    if ratio < 1 or abs(ratio*physics_dt-control_dt) > 1e-12:
        raise ValueError('The control period must be an integer number of physical steps.')
    data = mujoco.MjData(model)
    kins = [kinematics(meta, n) for n in ('left', 'right')]
    center = np.r_[meta['payload_nominal']['initial_position_world_m'], np.zeros(3)]
    initialize(model, data, meta, kins, center, CASES[name]['offset'])
    payload = meta['payload_nominal']
    learners = [HistoricalIdentifier(filter_rate=cfg.filter_rate, window=cfg.identification_window,
                                    normalized_threshold=cfg.identification_threshold,
                                    rate=cfg.identification_rate,
                                    lower=cfg.prior_lower, upper=cfg.prior_upper) for _ in range(2)]
    reference = reference_for(cfg, duration)
    controllers = [IdentificationController(kins[i], LinkDynamics(HERE/'models/arm_nominal.xml'),
                                            center, payload['mass_kg'], payload['diagonal_inertia_kg_m2'],
                                            i, cfg, CASES[name]['nonmonotonic'],
                                            reference=reference, identifier=learners[i]) for i in range(2)]
    limits = np.tile(meta['torque_limits_Nm'], 2)
    velocity_limits = np.tile(meta['joint_velocity_limits_rad_s'], 2)
    position_limits = np.tile(np.asarray(meta['joint_position_limits_rad']), (2, 1))
    payload_body = model.body('payload').id
    rows = []
    min_margin, min_sigma, min_joint = 1., 1., 10.
    max_torque, max_speed, max_weld, max_contacts, clipped = 0., 0., 0., 0, 0
    start = time.perf_counter()
    next_update = 0.
    for step in range(round(duration/control_dt)+1):
        t = step*control_dt
        # Synchronous force samples on both sides of a motor-command jump.
        # State integration is exclusively mj_step; no tracking-state resets.
        mujoco.mj_forward(model, data)
        commands, infos = zip(*(ctl.command(t, data.qpos[6*i:6*i+6], data.qvel[6*i:6*i+6])
                               for i, ctl in enumerate(controllers)))
        bases = [info['robot_bases'] for info in infos]
        left = observe(model, data, bases, payload['mass_kg'], payload['diagonal_inertia_kg_m2'])
        theta_control = np.array([learner.theta.copy() for learner in learners])
        command = np.concatenate(commands)
        applied = np.clip(command, -limits, limits)
        clipped += int(np.any(applied != command))
        data.ctrl[:] = applied
        mujoco.mj_forward(model, data)
        right = observe(model, data, bases, payload['mass_kg'], payload['diagonal_inertia_kg_m2'])
        for i, learner in enumerate(learners):
            learner.advance(t, right.momentum[i], right.remainder[i],
                            left.applied_force[i], right.applied_force[i])
        actual = np.r_[data.xpos[payload_body], Rotation.from_matrix(
            data.xmat[payload_body].reshape(3, 3)).as_euler('xyz')]
        ref = reference(t, center)[0]
        forces = np.array([np.linalg.solve(info['jacobian'].T, data.qfrc_constraint[6*i:6*i+6])
                           for i, info in enumerate(infos)])
        internal = forces-.5*np.sum(forces, axis=0)
        margins = np.array([info['barrier_margin'] for info in infos])
        sigma = np.array([info['sigma_min'] for info in infos])
        min_margin = min(min_margin, float(margins.min()))
        min_sigma = min(min_sigma, float(sigma.min()))
        max_torque = max(max_torque, float(np.max(np.abs(command)/limits)))
        min_joint = min(min_joint, float(np.min(np.minimum(data.qpos[:12]-position_limits[:,0],
                                                          position_limits[:,1]-data.qpos[:12]))))
        max_speed = max(max_speed, float(np.max(np.abs(data.qvel[:12])/velocity_limits)))
        max_weld = max(max_weld, float(np.max(np.abs(data.efc_pos[:12]))))
        max_contacts = max(max_contacts, int(data.ncon))
        rows.append(dict(t=t, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                         object_pose=actual, reference=ref, error=actual-ref,
                         local_error=np.array([info['error'] for info in infos]),
                         torque_command=command, torque_applied=applied,
                         surface=np.array([info['surface'] for info in infos]),
                         xi=np.array([info['xi'] for info in infos]), barrier_margin=margins,
                         upper=infos[0]['upper'], lower=infos[0]['lower'], sigma_min=sigma,
                         theta=np.array([learner.theta.copy() for learner in learners]),
                         theta_control=theta_control,
                         filtered_regressor=np.array([learner.regressor_f for learner in learners]),
                         filtered_torque=np.array([learner.force_f for learner in learners]),
                         lambda_min=np.array([learner.minimum_eigenvalue for learner in learners]),
                         normalized_lambda_min=np.array([learner.normalized_minimum_eigenvalue for learner in learners]),
                         prediction_error=np.array([learner.epsilon for learner in learners]),
                         measured_momentum=right.momentum.copy(),
                         measured_remainder=right.remainder.copy(),
                         measured_force_left=left.applied_force.copy(), measured_force_right=right.applied_force.copy(),
                         internal_force=internal, interaction_force=forces,
                         desired_internal_force=np.array([ctl.fd for ctl in controllers]),
                         weld_residual=data.efc_pos[:12].copy(), contact_count=data.ncon))
        if progress and t >= next_update:
            print(f'{name} t={t:.2f}/{duration:g}s, frozen={[z.frozen_time for z in learners]}, '
                  f'pose={np.linalg.norm(actual[:3]-ref[:3])*1000:.3f}mm, '
                  f'torque={max_torque:.3f}, wall={time.perf_counter()-start:.1f}s', flush=True)
            next_update = t+2.
        if step == round(duration/control_dt):
            break
        for _ in range(ratio):
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError('Nonfinite physical state.')
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(output/'trajectory.npz', **arrays)
    # Truth is first opened after the physical rollout; evaluation only.
    truth_data = json.loads((HERE/'models/plant_truth.json').read_text())
    truth = np.array([truth_data[n+'_density_multipliers']+[truth_data[n+'_viscous_multiplier'],
                     truth_data['payload_density_multiplier']] for n in ('left', 'right')])
    gram = np.array([learner.gram for learner in learners])
    p = np.array([learner.p for learner in learners])
    residual = p-np.einsum('aij,aj->ai', gram, truth)
    target = np.array([learner.data_target if learner.data_target is not None else np.full(8, np.nan)
                       for learner in learners])
    np.savez_compressed(output/'learning_state.npz', gram=gram, p=p,
                        true_parameters_offline_only=truth, stored_residual=residual,
                        data_target=target,
                        theta_at_freeze=np.array([z.theta_at_freeze if z.theta_at_freeze is not None
                                                 else np.full(8, np.nan) for z in learners]))
    tracking = arrays['t'] >= 2.
    final = arrays['t'] >= duration-2.
    summary = dict(case=name, duration_s=duration, control_dt_s=control_dt, physics_dt_s=physics_dt,
                   integrator=integrator,
                   wall_seconds=time.perf_counter()-start, settings=controllers[0].configuration(),
                   scenario=CASES[name], algorithm_variant='Measured-force, preconditioned historical identification extension of JAS',
                   sensing_extension='Ideal grasp reaction and actual payload pose/twist; shared payload observation; no accelerations or true parameters',
                   learning_extension='Pure historical prediction after IE, positive definite data preconditioner; original -YS learning term replaced',
                   reference_profile=cfg.excitation_profile,
                   position_rmse_after_2s_mm=float(np.sqrt(np.mean(np.sum(arrays['error'][tracking,:3]**2, axis=1)))*1000),
                   orientation_rmse_after_2s_deg=float(np.sqrt(np.mean(np.sum(arrays['error'][tracking,3:]**2, axis=1)))*180/np.pi),
                   minimum_normalized_barrier_gap=min_margin, minimum_jacobian_singular_value=min_sigma,
                   minimum_joint_limit_margin_rad=min_joint, max_joint_speed_limit_ratio=max_speed,
                   max_command_torque_limit_ratio=max_torque, saturated_control_samples=clipped,
                   max_weld_residual=max_weld, max_contact_count=max_contacts,
                   mujoco_warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(data.warning) if w.number},
                   ie_detection_times_s=[z.frozen_time for z in learners],
                   final_excitation_min_eigenvalues=[z.minimum_eigenvalue for z in learners],
                   final_normalized_excitation_min_eigenvalues=[z.normalized_minimum_eigenvalue for z in learners],
                   final_parameter_estimates=arrays['theta'][-1].tolist(),
                   true_parameters_offline_only=truth.tolist(),
                   final_parameter_error_norms=np.linalg.norm(arrays['theta'][-1]-truth, axis=1).tolist(),
                   final_maximum_parameter_component_errors=np.max(np.abs(arrays['theta'][-1]-truth), axis=1).tolist(),
                   data_target_error_norms=np.linalg.norm(target-truth, axis=1).tolist(),
                   data_target=target.tolist(), stored_learning_residual_norms=np.linalg.norm(residual, axis=1).tolist(),
                   excitation_condition_numbers=[float(np.linalg.cond(g)) for g in gram],
                   numerical_projection_counts=[z.projection_count for z in learners],
                   rejected_outside_prior_counts=[z.rejected_outside_prior for z in learners],
                   internal_force_error_final_rms=np.sqrt(np.mean((arrays['internal_force'][final]
                       -arrays['desired_internal_force'][final])**2, axis=0)).tolist(),
                   claim_limit='Exact exponential parameter convergence requires unbiased history; finite-step measured regression leaves an explicitly evaluated numerical floor.')
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    sources = [HERE.parent/'bare_jrnl_new_sample4.tex', Path(__file__), HERE/'run_experiments.py',
               HERE/'design_excitation.py']
    sources += sorted((HERE/'jas').glob('*.py'))+sorted((HERE/'exponential_identification').glob('*.py'))
    sources += sorted((HERE/'models').glob('*.xml'))+[HERE/'models/scene_metadata.json', HERE/'models/plant_truth.json']
    manifest = {str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=[*CASES, 'all'], default='nominal')
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--control-dt', type=float, default=.0005)
    parser.add_argument('--physics-dt', type=float, default=.000125)
    parser.add_argument('--integrator', choices=['implicitfast', 'RK4'], default='RK4')
    parser.add_argument('--window', type=float, default=6.)
    parser.add_argument('--rate', type=float, default=1.5)
    parser.add_argument('--threshold', type=float, default=1e-5)
    parser.add_argument('--filter-rate', type=float, default=8.)
    parser.add_argument('--profile', choices=['baseline', 'moderate', 'rich', 'rotation_rich'], default='moderate')
    parser.add_argument('--motion-duration', type=float, default=16.)
    parser.add_argument('--robust-mode', choices=['paper_sign', 'boundary_layer'], default='boundary_layer')
    parser.add_argument('--output', type=Path, default=HERE/'results_exponential')
    args = parser.parse_args()
    cfg = IdentificationSettings(identification_window=args.window, identification_rate=args.rate,
                                 identification_threshold=args.threshold, filter_rate=args.filter_rate,
                                 excitation_profile=args.profile, excitation_duration=args.motion_duration,
                                 robust_mode=args.robust_mode)
    for case in (CASES if args.case == 'all' else [args.case]):
        summary = run_case(case, args.duration, args.control_dt, args.physics_dt, args.output/case, cfg,
                           integrator=args.integrator)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
