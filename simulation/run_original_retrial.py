#!/usr/bin/env python3
"""Repeat the paper's original sign controller and projected composite law.

Only constant gains, prescribed reference, filter/window/IE threshold, and
numerical step/integrator settings change. No measured force enters control
or learning. All truth access is postponed until the rollout has ended.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from jas.dynamics import LinkDynamics
from jas.reference import desired
from run_experiments import HERE, CASES, initialize, kinematics
from design_excitation import multisine_reference
from original_retrial.controller import OriginalController, OriginalSettings


def reference_for(settings):
    if settings.excitation_profile == 'baseline':
        return desired
    return lambda t, center: multisine_reference(t, center, settings.excitation_duration,
                                                 settings.excitation_profile)


def run_case(name='nominal', duration=24., control_dt=.0005, physics_dt=.000125,
             output=None, settings=None, progress=True, integrator='RK4'):
    cfg = settings or OriginalSettings()
    if min(duration, control_dt, physics_dt, cfg.filter_rate, cfg.window, cfg.omega, cfg.gamma) <= 0:
        raise ValueError('Positive fixed gains, periods, window and IE threshold required.')
    if abs(duration/control_dt-round(duration/control_dt)) > 1e-8:
        raise ValueError('Duration must align with the control grid.')
    output = Path(output or HERE/'results_original_retrial'/name)
    output.mkdir(parents=True, exist_ok=True)
    # A completed run is marked by source_manifest.json, written last.
    if (output/'source_manifest.json').exists():
        raise FileExistsError(f'Completed result already exists: {output}')
    meta = json.loads((HERE/'models/scene_metadata.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(HERE/'models/dual_cr7.xml'))
    model.opt.timestep = physics_dt
    model.opt.integrator = {'implicitfast': mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                            'RK4': mujoco.mjtIntegrator.mjINT_RK4}[integrator]
    ratio = round(control_dt/physics_dt)
    if ratio < 1 or abs(ratio*physics_dt-control_dt) > 1e-12:
        raise ValueError('Control period must be an integer number of physics steps.')
    data = mujoco.MjData(model)
    kins = [kinematics(meta, hand) for hand in ('left', 'right')]
    center = np.r_[meta['payload_nominal']['initial_position_world_m'], np.zeros(3)]
    initial = initialize(model, data, meta, kins, center, CASES[name]['offset'])
    payload = meta['payload_nominal']
    reference = reference_for(cfg)
    controllers = [OriginalController(kins[i], LinkDynamics(HERE/'models/arm_nominal.xml'), center,
                                      payload['mass_kg'], payload['diagonal_inertia_kg_m2'],
                                      i, cfg, CASES[name]['nonmonotonic'], reference=reference)
                   for i in range(2)]
    limits = np.tile(meta['torque_limits_Nm'], 2)
    speed_limits = np.tile(meta['joint_velocity_limits_rad_s'], 2)
    position_limits = np.tile(np.asarray(meta['joint_position_limits_rad']), (2, 1))
    payload_body = model.body('payload').id
    rows = []
    started, next_update = time.perf_counter(), 0.
    min_margin, min_sigma, min_joint = 1., 1., 10.
    max_torque, max_speed, max_weld, max_contacts, saturated = 0., 0., 0., 0, 0
    for step in range(round(duration/control_dt)+1):
        t = step*control_dt
        commands, infos = zip(*(ctl.command(t, data.qpos[6*i:6*i+6], data.qvel[6*i:6*i+6])
                               for i, ctl in enumerate(controllers)))
        theta_control = np.array([ctl.learner.theta.copy() for ctl in controllers])
        command = np.concatenate(commands)
        applied = np.clip(command, -limits, limits)
        saturated += int(np.any(applied != command))
        known_left, known_right = [], []
        for i, ctl in enumerate(controllers):
            held = applied[6*i:6*i+6] if ctl.previous_applied is None else ctl.previous_applied
            info = infos[i]
            known_left.append(info['A'].T@(held+info['jacobian'].T@ctl.fd))
            known_right.append(info['A'].T@(applied[6*i:6*i+6]+info['jacobian'].T@ctl.fd))
        data.ctrl[:] = applied
        for i, ctl in enumerate(controllers):
            ctl.update(t, applied[6*i:6*i+6], infos[i])
        # Reaction forces below are DIAGNOSTICS ONLY and are sampled after the
        # controller and learning update. They have no route into either law.
        mujoco.mj_forward(model, data)
        actual = np.r_[data.xpos[payload_body], Rotation.from_matrix(
            data.xmat[payload_body].reshape(3, 3)).as_euler('xyz')]
        ref = reference(t, center)[0]
        forces = np.array([np.linalg.solve(info['jacobian'].T, data.qfrc_constraint[6*i:6*i+6])
                           for i, info in enumerate(infos)])
        internal = forces-.5*np.sum(forces, axis=0)
        margins = np.array([info['barrier_margin'] for info in infos])
        sigmas = np.array([info['sigma_min'] for info in infos])
        min_margin, min_sigma = min(min_margin, float(margins.min())), min(min_sigma, float(sigmas.min()))
        max_torque = max(max_torque, float(np.max(np.abs(command)/limits)))
        rows.append(dict(t=t, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                         object_pose=actual, reference=ref, error=actual-ref,
                         local_error=np.array([info['error'] for info in infos]),
                         torque_command=command, torque_applied=applied,
                         surface=np.array([info['surface'] for info in infos]),
                         xi=np.array([info['xi'] for info in infos]), barrier_margin=margins,
                         upper=infos[0]['upper'], lower=infos[0]['lower'], sigma_min=sigmas,
                         theta=np.array([ctl.learner.theta.copy() for ctl in controllers]),
                         theta_control=theta_control,
                         filtered_regressor=np.array([ctl.learner.filtered_regressor for ctl in controllers]),
                         filtered_torque=np.array([ctl.learner.tauf for ctl in controllers]),
                         lambda_min=np.array([ctl.learner.minimum_eigenvalue for ctl in controllers]),
                         prediction_error=np.array([ctl.learner.epsilon for ctl in controllers]),
                         regressor=np.array([info['regressor'] for info in infos]),
                         momentum=np.array([info['momentum'] for info in infos]),
                         remainder=np.array([info['remainder'] for info in infos]),
                         known_torque_left=np.array(known_left), known_torque_right=np.array(known_right),
                         learning_drive=np.array([info['regressor']@info['surface'] for info in infos]),
                         update_fixed_point_error=np.array([getattr(ctl.learner, 'last_fixed_point_error', 0.)
                                                            for ctl in controllers]),
                         internal_force=internal, interaction_force=forces,
                         desired_internal_force=np.array([ctl.fd for ctl in controllers]),
                         weld_residual=data.efc_pos[:12].copy(), contact_count=data.ncon))
        if progress and t >= next_update:
            print(f'{name} t={t:.2f}/{duration:g}s, IE={[ctl.learner.frozen_time for ctl in controllers]}, '
                  f'pose={np.linalg.norm(actual[:3]-ref[:3])*1000:.3f}mm, '
                  f'torque={max_torque:.3f}, wall={time.perf_counter()-started:.1f}s', flush=True)
            next_update = t+2.
        if min_margin <= 0 or min_sigma < .001 or max_torque > 1.2:
            raise RuntimeError('Pilot left the admissible unsaturated domain; no state resets or hidden gain changes.')
        if step == round(duration/control_dt):
            break
        for substep in range(ratio):
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError('Nonfinite physical state; no resets performed.')
            if abs(data.time-(t+(substep+1)*physics_dt)) > 1e-7:
                raise RuntimeError('Unexpected MuJoCo clock change.')
            min_joint = min(min_joint, float(np.min(np.minimum(data.qpos[:12]-position_limits[:, 0],
                                                               position_limits[:, 1]-data.qpos[:12]))))
            max_speed = max(max_speed, float(np.max(np.abs(data.qvel[:12])/speed_limits)))
            max_weld = max(max_weld, float(np.max(np.abs(data.efc_pos[:12]))))
            max_contacts = max(max_contacts, int(data.ncon))
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(output/'trajectory.npz', **arrays)
    # Evaluation only: this is the FIRST access to the unknown plant truth.
    true_data = json.loads((HERE/'models/plant_truth.json').read_text())
    truth = np.array([true_data[hand+'_density_multipliers']+[true_data[hand+'_viscous_multiplier'],
                     true_data['payload_density_multiplier']] for hand in ('left', 'right')])
    gram, p = np.array([ctl.learner.gram for ctl in controllers]), np.array([ctl.learner.p for ctl in controllers])
    residual = p-np.einsum('aij,aj->ai', gram, truth)
    np.savez_compressed(output/'learning_state.npz', gram=gram, p=p,
                        true_parameters_offline_only=truth, stored_residual=residual)
    final = arrays['t'] >= max(0., duration-2.)
    tracking = arrays['t'] >= min(2., duration/2.)
    error = arrays['error']
    summary = dict(case=name, duration_s=duration, control_dt_s=control_dt, physics_dt_s=physics_dt,
                   integrator=integrator, wall_seconds=time.perf_counter()-started,
                   initial_object_pose=initial.tolist(), settings=controllers[0].configuration(), scenario=CASES[name],
                   algorithm_variant='Original JAS paper sign controller and original projected composite learning law',
                   learning_law='Upsilon*Pr(-Y*S+gamma*(p-G*theta)); constant positive diagonal Upsilon; original raw-Gram IE/freeze',
                   sensing='Local encoders and held motor torque only; no reaction force, acceleration or true parameters supplied',
                   reference_profile=cfg.excitation_profile,
                   position_rmse_after_2s_mm=float(np.sqrt(np.mean(np.sum(error[tracking, :3]**2, axis=1)))*1000),
                   orientation_rmse_after_2s_deg=float(np.sqrt(np.mean(np.sum(error[tracking, 3:]**2, axis=1)))*180/np.pi),
                   final_position_rmse_mm=float(np.sqrt(np.mean(np.sum(error[final, :3]**2, axis=1)))*1000),
                   final_orientation_rmse_deg=float(np.sqrt(np.mean(np.sum(error[final, 3:]**2, axis=1)))*180/np.pi),
                   minimum_normalized_barrier_gap=min_margin, minimum_jacobian_singular_value=min_sigma,
                   minimum_joint_limit_margin_rad=min_joint, max_joint_speed_limit_ratio=max_speed,
                   max_command_torque_limit_ratio=max_torque, saturated_control_samples=saturated,
                   max_weld_residual=max_weld, max_contact_count=max_contacts,
                   mujoco_warnings={mujoco.mjtWarning(i).name:int(w.number) for i, w in enumerate(data.warning) if w.number},
                   ie_detection_times_s=[ctl.learner.frozen_time for ctl in controllers],
                   final_excitation_min_eigenvalues=[ctl.learner.minimum_eigenvalue for ctl in controllers],
                   final_parameter_estimates=arrays['theta'][-1].tolist(), true_parameters_offline_only=truth.tolist(),
                   initial_parameter_error_norms=np.linalg.norm(arrays['theta'][0]-truth, axis=1).tolist(),
                   final_parameter_error_norms=np.linalg.norm(arrays['theta'][-1]-truth, axis=1).tolist(),
                   stored_learning_residual_norms=np.linalg.norm(residual, axis=1).tolist(),
                   excitation_condition_numbers=[float(np.linalg.cond(g)) for g in gram],
                   numerical_projection_counts=[ctl.learner.numerical_projection_count for ctl in controllers],
                   active_projection_samples=[getattr(ctl.learner, 'active_projection_samples', None) for ctl in controllers],
                   maximum_update_fixed_point_error=[getattr(ctl.learner, 'maximum_fixed_point_error', None) for ctl in controllers],
                   internal_force_error_final_rms=np.sqrt(np.mean((arrays['internal_force'][final]
                       -arrays['desired_internal_force'][final])**2, axis=0)).tolist(),
                   claim_limit='Original-law sampled simulation; exact truth exponential convergence requires the theorem hypotheses including zero frozen residual. Numerical weld and sampled sign dynamics are explicitly approximations.')
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    sources = [HERE.parent/'bare_jrnl_new_sample4.tex', Path(__file__), HERE/'run_experiments.py',
               HERE/'design_excitation.py']
    sources += sorted((HERE/'jas').glob('*.py'))+sorted((HERE/'original_retrial').glob('*.py'))
    sources += sorted((HERE/'models').glob('*.xml'))+[HERE/'models/scene_metadata.json', HERE/'models/plant_truth.json']
    manifest = {str(path.relative_to(HERE.parent)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (output/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=[*CASES, 'all'], default='nominal')
    parser.add_argument('--duration', type=float, default=24.)
    parser.add_argument('--control-dt', type=float, default=.0005)
    parser.add_argument('--physics-dt', type=float, default=.000125)
    parser.add_argument('--integrator', choices=['RK4', 'implicitfast'], default='RK4')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=HERE/'results_original_retrial')
    args = parser.parse_args()
    settings = OriginalSettings(**(json.loads(args.config.read_text()) if args.config else {}))
    for case in CASES if args.case == 'all' else [args.case]:
        result = run_case(case, args.duration, args.control_dt, args.physics_dt,
                          args.output/case, settings, integrator=args.integrator)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
