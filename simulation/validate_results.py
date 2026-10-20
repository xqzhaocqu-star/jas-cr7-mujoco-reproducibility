#!/usr/bin/env python3
"""Independently validate saved dual-CR7 simulation artifacts.

Only validation.json and validation_summary.json are written. Sources, models,
states, trajectories, and learning records are never changed. Checks concern
saved samples plus explicitly reported short MuJoCo replay intervals; they do
not establish continuous-time or physical-robot guarantees.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CASES = ('nominal', 'nonmonotonic', 'large_initial_error')


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


class Audit:
    def __init__(self, case):
        self.report = dict(case=case, passed=False, checks={}, failures=[],
                           scope='All saved control ticks; independently reconstructed '
                                 'FK/Jacobians; sampled short full-MuJoCo replays. '
                                 'No continuous-time or physical-robot guarantee.')

    def check(self, name, passed, **details):
        passed = bool(passed)
        self.report['checks'][name] = dict(passed=passed, **plain(details))
        if not passed:
            self.report['failures'].append(name)

    def finish(self, output):
        self.report['passed'] = not self.report['failures']
        (output/'validation.json').write_text(
            json.dumps(plain(self.report), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        return self.report


def maximum_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))


def skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def independent_reference(times, center):
    """Vectorized evaluation of the independently specified task equations."""
    times = np.asarray(times)
    phase = times.copy()
    transition = (times > 8.) & (times < 10.)
    z = (times[transition]-8.)/2.
    phase[transition] = 8.+2*(z-2.5*z**4+3*z**5-z**6)
    phase[times >= 10.] = 9.
    z = np.clip(times, 0., 1.)
    gate = 10*z**3-15*z**4+6*z**5
    amplitude = np.array([.035, .025, .025, .08, .07, .08])
    frequency = np.array([.83, 1.11, .67, .92, .76, 1.21])
    shift = np.array([0., .7, 1.2, .4, 1., 1.8])
    return center+gate[:, None]*amplitude*np.sin(phase[:, None]*frequency+shift)


def independent_bounds(times, chi, ell, nonmonotonic):
    final_a = np.array([.012, .014, .010, .045, .040, .050])
    final_b = np.array([.018, .010, .016, .060, .055, .035])
    a = ell*np.tanh(chi*final_a)
    b = ell*np.tanh(chi*final_b)
    upper = a+(ell-a)*np.exp(-.65*times[:, None])
    lower = b+(ell-b)*np.exp(-.65*times[:, None])
    if nonmonotonic:
        z = np.clip((times-3.)/4., 0., 1.)
        bump = 64*z**3*(1-z)**3
        upper += .23*ell*bump[:, None]
        lower += .8*.23*ell*bump[:, None]
    return upper, lower


def verify_sources(audit, manifest):
    mismatches = []
    for relative, expected in manifest.items():
        path = PROJECT/relative
        if not path.is_file():
            mismatches.append(dict(path=relative, reason='missing'))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(dict(path=relative, reason='SHA256 mismatch'))
    required = {'bare_jrnl_new_sample4.tex', 'simulation/run_experiments.py',
                'simulation/models/dual_cr7.xml', 'simulation/models/arm_nominal.xml',
                'simulation/models/scene_metadata.json', 'simulation/models/plant_truth.json'}
    required.update(str(p.relative_to(PROJECT)) for p in (HERE/'jas').glob('*.py'))
    missing = sorted(required-set(manifest))
    audit.check('source_manifest_sha256', not mismatches and not missing,
                checked_files=len(manifest), mismatches=mismatches,
                missing_required_sources=missing)
    assets_path = HERE/'models'/'asset_sha256.json'
    assets = json.loads(assets_path.read_text())
    bad_assets = []
    for relative, expected in assets.items():
        path = HERE/'models'/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            bad_assets.append(relative)
    audit.check('imported_asset_sha256', not bad_assets,
                checked_files=len(assets), mismatches=bad_assets)


def reconstruct_kinematics(model, arrays, metadata):
    """Use the full scene FK, independently of jas.geometry and its local model."""
    n = len(arrays['t'])
    local = np.empty((n, 2, 6))
    sigmas = np.empty((n, 2))
    object_pose = np.empty((n, 6))
    data = mujoco.MjData(model)
    payload = model.body('payload').id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for tick in range(n):
        data.qpos[:] = arrays['qpos'][tick]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        object_pose[tick, :3] = data.xpos[payload]
        object_pose[tick, 3:] = Rotation.from_matrix(
            data.xmat[payload].reshape(3, 3)).as_euler('xyz')
        for arm, name in enumerate(('left', 'right')):
            frame = metadata['grasp_frames'][name]
            site = model.site(name+'_palm').id
            palm_rotation = data.site_xmat[site].reshape(3, 3)
            rotation = palm_rotation@np.asarray(frame['palm_rotation_object']).T
            shift = -rotation@np.asarray(frame['palm_position_object_m'])
            local[tick, arm, :3] = data.site_xpos[site]+shift
            rpy = Rotation.from_matrix(rotation).as_euler('xyz')
            local[tick, arm, 3:] = rpy
            mujoco.mj_jacSite(model, data, jp, jr, site)
            _, pitch, yaw = rpy
            angular = np.array([[np.cos(yaw)*np.cos(pitch), -np.sin(yaw), 0.],
                                [np.sin(yaw)*np.cos(pitch), np.cos(yaw), 0.],
                                [-np.sin(pitch), 0., 1.]])
            columns = slice(6*arm, 6*arm+6)
            jacobian = np.vstack((jp[:, columns]-skew(shift)@jr[:, columns],
                                   np.linalg.solve(angular, jr[:, columns])))
            sigmas[tick, arm] = np.linalg.svd(jacobian, compute_uv=False)[-1]
    return object_pose, local, sigmas


def replay_intervals(model, arrays, control_dt, physics_dt, sample_count):
    """Replay short intervals, without depending on logged summary assertions."""
    model.opt.timestep = physics_dt
    substeps = int(round(control_dt/physics_dt))
    indices = np.unique(np.linspace(0, len(arrays['t'])-2,
                                    min(sample_count, len(arrays['t'])-1)).astype(int))
    state_error, velocity_error, time_error = [], [], []
    contacts, warnings = 0, 0
    for index in indices:
        data = mujoco.MjData(model)
        data.qpos[:] = arrays['qpos'][index]
        data.qvel[:] = arrays['qvel'][index]
        data.time = arrays['t'][index]
        data.ctrl[:] = arrays['torque_applied'][index]
        mujoco.mj_forward(model, data)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            contacts = max(contacts, data.ncon)
        state_error.append(maximum_error(data.qpos, arrays['qpos'][index+1]))
        velocity_error.append(maximum_error(data.qvel, arrays['qvel'][index+1]))
        time_error.append(abs(data.time-arrays['t'][index+1]))
        warnings += sum(int(w.number) for w in data.warning)
    return dict(intervals=len(indices), maximum_qpos_error=float(max(state_error)),
                maximum_qvel_error=float(max(velocity_error)),
                maximum_time_error_s=float(max(time_error)),
                maximum_contacts=contacts, warning_count=warnings)


def validate_learning(audit, arrays, saved, summary, truth):
    times = arrays['t']
    yf, torque = arrays['filtered_regressor'], arrays['filtered_torque']
    # Batched matmul performs the raw outer products; no positive diagonal,
    # regularization, singular-value truncation, or learner implementation used.
    integrand_g = yf@np.swapaxes(yf, -1, -2)
    integrand_p = np.einsum('naik,nak->nai', yf, torque)
    delta = np.diff(times)
    cg = np.zeros_like(integrand_g)
    cp = np.zeros_like(integrand_p)
    cg[1:] = np.cumsum(.5*delta[:, None, None, None]
                       *(integrand_g[1:]+integrand_g[:-1]), axis=0)
    cp[1:] = np.cumsum(.5*delta[:, None, None]
                       *(integrand_p[1:]+integrand_p[:-1]), axis=0)
    window = float(summary['settings']['window'])
    omega = float(summary['settings']['omega'])
    # The recorded fixed-step learner retains quadrature intervals whose right
    # endpoint is newer than the cutoff. This differs from a continuously cut
    # first interval by at most one control sample, reported explicitly below.
    cutoff = np.maximum(0., times-window)
    left = np.searchsorted(times, cutoff, side='right')-1
    rolling_g = cg-cg[left]
    rolling_p = cp-cp[left]
    rolling_g = .5*(rolling_g+np.swapaxes(rolling_g, -1, -2))
    eigenvalues = np.linalg.eigvalsh(rolling_g)[..., 0]
    recovered = []
    frozen_g, frozen_p = [], []
    expected_lambda = np.empty_like(arrays['lambda_min'])
    expected_prediction = np.empty_like(arrays['prediction_error'])
    for arm in range(2):
        eligible = np.flatnonzero((times >= window) & (eigenvalues[:, arm] >= omega))
        detected_index = int(eligible[0]) if len(eligible) else None
        detected_time = float(times[detected_index]) if detected_index is not None else None
        recovered.append(detected_time)
        final_index = detected_index if detected_index is not None else len(times)-1
        frozen_g.append(rolling_g[final_index, arm])
        frozen_p.append(rolling_p[final_index, arm])
        active_g = rolling_g[:, arm].copy()
        active_p = rolling_p[:, arm].copy()
        expected_lambda[:, arm] = eigenvalues[:, arm]
        if detected_index is not None:
            active_g[detected_index:] = active_g[detected_index]
            active_p[detected_index:] = active_p[detected_index]
            expected_lambda[detected_index:, arm] = eigenvalues[detected_index, arm]
        expected_prediction[:, arm] = active_p-np.einsum(
            'nij,nj->ni', active_g, arrays['theta_control'][:, arm])
    frozen_g, frozen_p = np.array(frozen_g), np.array(frozen_p)
    reported = summary['ie_detection_times_s']
    timing_ok = all((a is None and b is None) or
                    (a is not None and b is not None and abs(a-b) <= 1e-9)
                    for a, b in zip(recovered, reported))
    audit.check('ie_first_threshold_crossing_from_raw_data', timing_ok,
                recomputed_times_s=recovered, reported_times_s=reported,
                threshold=omega, window_s=window,
                quadrature='Raw-sample trapezoids, endpoint window; at most one '
                           'control-interval edge discrepancy from a continuous window.')
    gram_error = maximum_error(frozen_g, saved['gram'])
    gram_tolerance = 5e-8+1e-14*float(np.max(np.abs(frozen_g)))
    p_error = maximum_error(frozen_p, saved['p'])
    p_tolerance = 5e-8+1e-12*float(np.max(np.abs(frozen_p)))
    audit.check('stored_gram_and_p_from_unregularized_raw_integrals',
                gram_error <= gram_tolerance and p_error <= p_tolerance,
                maximum_gram_error=gram_error, gram_tolerance=gram_tolerance,
                maximum_p_error=p_error, p_tolerance=p_tolerance,
                diagonal_difference=np.diagonal(saved['gram']-frozen_g, axis1=1, axis2=2),
                recomputed_minimum_eigenvalues=np.linalg.eigvalsh(frozen_g)[:, 0])
    lambda_error = maximum_error(expected_lambda, arrays['lambda_min'])
    audit.check('excitation_eigenvalue_preserved_after_freeze', lambda_error <= 5e-8,
                maximum_logged_lambda_error=lambda_error, absolute_tolerance=5e-8)
    prediction_error = maximum_error(expected_prediction, arrays['prediction_error'])
    prediction_tolerance = 1e-7+1e-11*float(np.max(np.abs(expected_prediction)))
    audit.check('prediction_error_uses_current_control_estimate',
                prediction_error <= prediction_tolerance,
                maximum_error=prediction_error, tolerance=prediction_tolerance)
    residual = frozen_p-np.einsum('aij,aj->ai', frozen_g, truth)
    residual_error = maximum_error(residual, saved['stored_residual'])
    audit.check('stored_residual_equals_p_minus_gram_times_truth',
                residual_error <= p_tolerance+8*gram_tolerance,
                maximum_error=residual_error, residual_norms=np.linalg.norm(residual, axis=1),
                interpretation='A nonzero stored residual permits biased parameter estimates.')
    audit.check('learning_truth_matches_plant_truth_file',
                maximum_error(saved['true_parameters_offline_only'], truth) <= 1e-13,
                maximum_error=maximum_error(saved['true_parameters_offline_only'], truth))


def compare_reference(arrays, path):
    with np.load(path/'trajectory.npz', allow_pickle=False) as source:
        reference = {k: source[k] for k in ('t', 'object_pose', 'error')}
    valid = (arrays['t'] >= max(2., reference['t'][0])) & (arrays['t'] <= reference['t'][-1])
    times = arrays['t'][valid]
    if not len(times):
        raise ValueError('Reference has no shared interval after t=2 s.')
    interp = np.column_stack([np.interp(times, reference['t'], reference['object_pose'][:, i])
                              for i in range(6)])
    difference = arrays['object_pose'][valid]-interp
    return dict(reference_directory=str(path.resolve()), common_samples=len(times),
                position_difference_rms_mm=float(np.sqrt(np.mean(np.sum(difference[:, :3]**2, axis=1)))*1000),
                orientation_coordinate_difference_rms_deg=float(np.sqrt(np.mean(np.sum(difference[:, 3:]**2, axis=1)))*180/np.pi),
                maximum_position_difference_mm=float(np.max(np.linalg.norm(difference[:, :3], axis=1)))*1000,
                interpretation='Descriptive numerical step comparison; no extrapolation '
                               'to exact continuous-time sign dynamics.')


def validate_case(directory, expected_duration=14., samples=101, reference=None):
    audit = Audit(directory.name)
    required_files = ('trajectory.npz', 'learning_state.npz', 'summary.json', 'source_manifest.json')
    missing = [name for name in required_files if not (directory/name).is_file()]
    audit.check('complete_artifact_files', not missing, missing=missing)
    if missing:
        return audit.finish(directory)
    summary = json.loads((directory/'summary.json').read_text())
    manifest = json.loads((directory/'source_manifest.json').read_text())
    verify_sources(audit, manifest)
    metadata = json.loads((HERE/'models/scene_metadata.json').read_text())
    plant_truth = json.loads((HERE/'models/plant_truth.json').read_text())
    with np.load(directory/'trajectory.npz', allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    with np.load(directory/'learning_state.npz', allow_pickle=False) as source:
        learned = {key: source[key] for key in source.files}
    required_arrays = {'t', 'qpos', 'qvel', 'object_pose', 'reference', 'error',
                       'local_error', 'torque_command', 'torque_applied', 'surface', 'xi',
                       'barrier_margin', 'upper', 'lower', 'sigma_min', 'theta',
                       'theta_control', 'filtered_regressor', 'filtered_torque', 'lambda_min',
                       'prediction_error', 'weld_residual', 'contact_count'}
    absent = sorted(required_arrays-set(arrays))
    audit.check('required_trajectory_arrays', not absent, missing=absent)
    if absent:
        return audit.finish(directory)
    times = arrays['t']
    n = len(times)
    dt = float(summary['control_dt_s'])
    physics_dt = float(summary['physics_dt_s'])
    intervals = np.diff(times)
    audit.check('monotonic_complete_time_grid',
                n > 1 and np.all(intervals > 0) and abs(times[0]) <= 1e-12
                and abs(times[-1]-expected_duration) <= 1e-9
                and n == int(round(expected_duration/dt))+1
                and np.max(np.abs(intervals-dt)) <= 1e-10,
                samples=n, first_time_s=float(times[0]), last_time_s=float(times[-1]),
                expected_duration_s=expected_duration, control_dt_s=dt,
                maximum_step_error_s=float(np.max(np.abs(intervals-dt))))
    shapes = {'qpos': (n, 19), 'qvel': (n, 18), 'object_pose': (n, 6),
              'local_error': (n, 2, 6), 'theta': (n, 2, 8),
              'filtered_regressor': (n, 2, 8, 6), 'filtered_torque': (n, 2, 6)}
    wrong_shapes = {k: dict(actual=arrays[k].shape, expected=v)
                    for k, v in shapes.items() if arrays[k].shape != v}
    bad_finite = [key for key in required_arrays-{'upper', 'lower'}
                  if not np.all(np.isfinite(arrays[key]))]
    audit.check('finite_state_and_expected_array_shapes', not wrong_shapes and not bad_finite,
                wrong_shapes=wrong_shapes, nonfinite_arrays=bad_finite)
    if wrong_shapes or bad_finite:
        return audit.finish(directory)
    quaternion_error = float(np.max(np.abs(np.linalg.norm(arrays['qpos'][:, 15:19], axis=1)-1)))
    audit.check('payload_unit_quaternion', quaternion_error <= 1e-10,
                maximum_norm_error=quaternion_error)
    center = np.r_[metadata['payload_nominal']['initial_position_world_m'], np.zeros(3)]
    ref = independent_reference(times, center)
    audit.check('reference_matches_independently_evaluated_task',
                maximum_error(ref, arrays['reference']) <= 2e-12,
                maximum_error=maximum_error(ref, arrays['reference']))
    model = mujoco.MjModel.from_xml_path(str(HERE/'models/dual_cr7.xml'))
    object_pose, local_pose, sigmas = reconstruct_kinematics(model, arrays, metadata)
    object_error, local_error = object_pose-ref, local_pose-ref[:, None, :]
    audit.check('object_pose_and_errors_reconstructed_from_qpos',
                maximum_error(object_pose, arrays['object_pose']) <= 2e-11
                and maximum_error(object_error, arrays['error']) <= 2e-11,
                maximum_pose_error=maximum_error(object_pose, arrays['object_pose']),
                maximum_tracking_error_discrepancy=maximum_error(object_error, arrays['error']))
    audit.check('local_encoder_errors_reconstructed_from_full_scene_fk',
                maximum_error(local_error, arrays['local_error']) <= 2e-10,
                maximum_error=maximum_error(local_error, arrays['local_error']), checked_ticks=n)
    position_residual = np.linalg.norm(local_pose[:, :, :3]-object_pose[:, None, :3], axis=2)
    angle_residual = np.empty((n, 2))
    object_rotation = Rotation.from_euler('xyz', object_pose[:, 3:])
    for arm in range(2):
        angle_residual[:, arm] = (Rotation.from_euler('xyz', local_pose[:, arm, 3:])
                                  *object_rotation.inv()).magnitude()
    audit.check('numerical_rigid_grasp_pose_consistency',
                np.max(position_residual) <= 5e-4 and np.max(angle_residual) <= .002,
                maximum_position_residual_m=float(np.max(position_residual)),
                maximum_rotation_residual_rad=float(np.max(angle_residual)),
                thresholds=dict(position_m=5e-4, rotation_rad=.002),
                interpretation='Finite MuJoCo weld compliance, not an exact ideal constraint.')
    audit.check('weld_equality_residual', np.max(np.abs(arrays['weld_residual'])) <= 1e-3,
                maximum_absolute_residual=float(np.max(np.abs(arrays['weld_residual']))),
                threshold=1e-3)
    chi, ell = np.array(summary['settings']['chi']), np.array(summary['settings']['ell'])
    mu_a, mu_b = independent_bounds(times, chi, ell, summary['scenario']['nonmonotonic'])
    eta_local = ell*np.tanh(chi*local_error)
    gap_a = (mu_a[:, None, :]-eta_local)/ell
    gap_b = (mu_b[:, None, :]+eta_local)/ell
    local_margin = np.minimum(gap_a, gap_b)
    eta_object = ell*np.tanh(chi*object_error)
    true_margin = np.minimum((mu_a-eta_object)/ell, (mu_b+eta_object)/ell)
    audit.check('both_local_and_actual_object_barriers_admissible_every_tick',
                np.min(local_margin) > 0 and np.min(true_margin) > 0,
                minimum_local_normalized_gap=float(np.min(local_margin)),
                minimum_actual_object_normalized_gap=float(np.min(true_margin)),
                first_bad_local_indices=np.argwhere(local_margin <= 0)[:5],
                first_bad_actual_indices=np.argwhere(true_margin <= 0)[:5])
    audit.check('logged_normalized_barrier_gaps',
                maximum_error(local_margin, arrays['barrier_margin']) <= 2e-10,
                maximum_error=maximum_error(local_margin, arrays['barrier_margin']))
    expected_xi = eta_local/((mu_a[:, None, :]-eta_local)*(mu_b[:, None, :]+eta_local))
    audit.check('logged_barrier_transform', maximum_error(expected_xi, arrays['xi']) <= 2e-9,
                maximum_error=maximum_error(expected_xi, arrays['xi']))
    with np.errstate(divide='ignore'):
        expected_upper = np.arctanh(mu_a/ell)/chi
        expected_lower = -np.arctanh(mu_b/ell)/chi
    bounds_ok = (np.all(np.isposinf(arrays['upper'][0]))
                 and np.all(np.isneginf(arrays['lower'][0]))
                 and np.all(np.isfinite(arrays['upper'][1:]))
                 and np.all(np.isfinite(arrays['lower'][1:]))
                 and maximum_error(expected_upper[1:], arrays['upper'][1:]) <= 2e-10
                 and maximum_error(expected_lower[1:], arrays['lower'][1:]) <= 2e-10)
    constraints_ok = (np.all(local_error < arrays['upper'][:, None, :])
                       and np.all(local_error > arrays['lower'][:, None, :])
                       and np.all(object_error < arrays['upper'])
                       and np.all(object_error > arrays['lower']))
    audit.check('physical_error_envelopes_at_every_tick', bounds_ok and constraints_ok,
                initial_physical_envelope='infinite', later_bound_samples=n-1,
                all_local_and_actual_errors_strictly_inside=constraints_ok)
    sigma_error = maximum_error(sigmas, arrays['sigma_min'])
    min_cos = float(np.min(np.abs(np.cos(local_pose[:, :, 4]))))
    audit.check('independent_jacobian_and_euler_chart_validity',
                np.min(sigmas) > .001 and sigma_error <= 2e-10 and min_cos > .2,
                minimum_singular_value=float(np.min(sigmas)),
                maximum_logged_sigma_error=sigma_error, minimum_absolute_cos_pitch=min_cos,
                thresholds=dict(singular_value=.001, absolute_cos_pitch=.2))
    limits = np.tile(metadata['torque_limits_Nm'], 2)
    raw_ratio = float(np.max(np.abs(arrays['torque_command'])/limits))
    applied_ratio = float(np.max(np.abs(arrays['torque_applied'])/limits))
    saturated = int(np.count_nonzero(np.any(np.abs(arrays['torque_applied']
                                                 -arrays['torque_command']) > 1e-12, axis=1)))
    motor_error = maximum_error(np.max(np.abs(model.actuator_ctrlrange), axis=1), limits)
    audit.check('raw_applied_and_model_motor_limits_no_saturation',
                raw_ratio <= 1+1e-12 and applied_ratio <= 1+1e-12
                and saturated == 0 and motor_error <= 1e-12,
                maximum_raw_limit_ratio=raw_ratio, maximum_applied_limit_ratio=applied_ratio,
                saturated_saved_ticks=saturated, model_limit_discrepancy=motor_error)
    position_limits = np.tile(np.array(metadata['joint_position_limits_rad']), (2, 1))
    q = arrays['qpos'][:, :12]
    qmargin = np.minimum(q-position_limits[:, 0], position_limits[:, 1]-q)
    speed_limits = np.tile(metadata['joint_velocity_limits_rad_s'], 2)
    speed_ratio = float(np.max(np.abs(arrays['qvel'][:, :12])/speed_limits))
    audit.check('joint_position_and_speed_limits', np.min(qmargin) > 0 and speed_ratio <= 1+1e-12,
                minimum_position_limit_margin_rad=float(np.min(qmargin)),
                maximum_speed_limit_ratio=speed_ratio)
    step_ratio = float(np.max(np.abs(np.diff(q, axis=0))/(intervals[:, None]*speed_limits)))
    audit.check('saved_states_have_no_discontinuous_joint_jumps', step_ratio <= 1.05,
                maximum_step_to_velocity_limit_ratio=step_ratio, threshold=1.05,
                interpretation='Sampled consistency evidence; supplemented by interval replays.')
    truth = np.array([plant_truth[name+'_density_multipliers']
                      +[plant_truth[name+'_viscous_multiplier'], plant_truth['payload_density_multiplier']]
                      for name in ('left', 'right')])
    lower = np.asarray(metadata['parameterization']['local_prior_lower'])
    upper = np.asarray(metadata['parameterization']['local_prior_upper'])
    parameter_ok = (np.all(arrays['theta'] >= lower-1e-12)
                     and np.all(arrays['theta'] <= upper+1e-12)
                     and np.all(arrays['theta_control'] >= lower-1e-12)
                     and np.all(arrays['theta_control'] <= upper+1e-12)
                     and np.all(truth >= lower) and np.all(truth <= upper)
                     and maximum_error(summary['true_parameters_offline_only'], truth) <= 1e-13)
    audit.check('all_estimates_and_true_parameters_within_prior_box', parameter_ok,
                minimum_estimate=float(np.min(arrays['theta'])),
                maximum_estimate=float(np.max(arrays['theta'])), true_parameters=truth,
                lower=lower, upper=upper)
    continuity_error = maximum_error(arrays['theta_control'][1:], arrays['theta'][:-1])
    audit.check('parameter_log_respects_control_then_update_order', continuity_error <= 1e-13,
                maximum_error=continuity_error)
    audit.check('no_logged_contacts_or_mujoco_warnings',
                np.max(arrays['contact_count']) == 0 and not summary['mujoco_warnings']
                and summary['max_contact_count'] == 0,
                maximum_saved_contacts=int(np.max(arrays['contact_count'])),
                physics_substep_contact_maximum=summary['max_contact_count'],
                cumulative_warnings=summary['mujoco_warnings'])
    final = times >= max(0., times[-1]-2.)
    final_position = float(np.sqrt(np.mean(np.sum(object_error[final, :3]**2, axis=1)))*1000)
    final_orientation = float(np.sqrt(np.mean(np.sum(object_error[final, 3:]**2, axis=1)))*180/np.pi)
    audit.check('final_two_second_rmse_independently_recomputed',
                abs(final_position-summary['final_position_rmse_mm']) <= 1e-8
                and abs(final_orientation-summary['final_orientation_rmse_deg']) <= 1e-8,
                position_rmse_mm=final_position, orientation_coordinate_rmse_deg=final_orientation,
                saved_ticks_in_final_window=int(np.count_nonzero(final)))
    audit.check('sampled_final_tracking_accuracy', final_position <= .25 and final_orientation <= .05,
                position_rmse_mm=final_position, orientation_coordinate_rmse_deg=final_orientation,
                thresholds=dict(position_rmse_mm=.25, orientation_rmse_deg=.05))
    substep_ok = (summary['saturated_control_samples'] == 0
                   and summary['minimum_joint_limit_margin_rad'] > 0
                   and summary['max_joint_speed_limit_ratio'] <= 1+1e-12
                   and summary['max_weld_residual'] <= 1e-3)
    audit.check('reported_physics_substep_diagnostics', substep_ok,
                source='Cumulative runner diagnostics, independently supplemented by saved states/replays',
                saturated_samples=summary['saturated_control_samples'],
                minimum_joint_margin_rad=summary['minimum_joint_limit_margin_rad'],
                maximum_speed_ratio=summary['max_joint_speed_limit_ratio'],
                maximum_weld_residual=summary['max_weld_residual'])
    validate_learning(audit, arrays, learned, summary, truth)
    replay = replay_intervals(model, arrays, dt, physics_dt, samples)
    audit.check('independent_full_physics_replay_no_reset_evidence',
                replay['maximum_qpos_error'] <= 1e-7
                and replay['maximum_qvel_error'] <= 1e-5
                and replay['maximum_time_error_s'] <= 1e-10
                and replay['maximum_contacts'] == 0 and replay['warning_count'] == 0,
                **replay, thresholds=dict(qpos_error=1e-7, qvel_error=1e-5, time_error_s=1e-10))
    if reference is not None:
        refcase = reference if (reference/'trajectory.npz').is_file() else reference/directory.name
        audit.report['step_comparison'] = compare_reference(arrays, refcase)
    return audit.finish(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE/'results')
    parser.add_argument('--case', default='all', choices=CASES+('all',))
    parser.add_argument('--reference', type=Path, default=None,
                        help='Optional result root or case directory for numerical step comparison.')
    parser.add_argument('--duration-s', type=float, default=14.,
                        help='Expected complete duration; production experiment default is 14 s.')
    parser.add_argument('--replay-samples', type=int, default=101)
    args = parser.parse_args()
    results = args.results.resolve()
    if not results.is_dir():
        parser.error(f'Results directory does not exist: {results}')
    reports = []
    for name in (CASES if args.case == 'all' else (args.case,)):
        directory = results/name
        if not directory.is_dir():
            reports.append(dict(case=name, passed=False, failures=['missing_case_directory']))
            print(f'{name}: FAIL missing case directory', flush=True)
            continue
        try:
            report = validate_case(directory, args.duration_s, args.replay_samples, args.reference)
        except Exception as exc:
            audit = Audit(name)
            audit.check('validator_completed', False, exception_type=type(exc).__name__, message=str(exc))
            report = audit.finish(directory)
        reports.append(report)
        status = 'PASS' if report['passed'] else 'FAIL '+', '.join(report['failures'])
        print(f'{name}: {status}', flush=True)
    aggregate = dict(passed=all(r['passed'] for r in reports),
                     cases=[dict(case=r['case'], passed=r['passed'], failures=r['failures']) for r in reports],
                     expected_duration_s=args.duration_s,
                     interpretation='Saved-sample and sampled-replay numerical validation only.')
    (results/'validation_summary.json').write_text(json.dumps(aggregate, indent=2)+'\n')
    return 0 if aggregate['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
